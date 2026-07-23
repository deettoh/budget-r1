"""Unit tests for the SFT data builder pure functions."""

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "build_sft_data",
    os.path.join(
        os.path.dirname(__file__), "..", "scripts", "build_sft_data.py"
    ),
)
bsd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bsd)


def _trace(question, response, calls=2, em=1.0, source="2wikimultihopqa",
           gold=2):
    prompt = (
        f"<|im_start|>user\nAnswer the question. Question: {question}\n"
        "<|im_end|>\n"
    )
    text = prompt + bsd.ASSISTANT_MARKER + response + "<|im_end|>"
    return {
        "sequences_str": text,
        "em": em,
        "valid_search_calls": calls,
        "data_source": source,
        "gold_budget": gold,
    }


class SplitAndExtractTest(unittest.TestCase):
    def test_split_prompt_response(self):
        prompt, resp = bsd.split_prompt_response(
            "userstuff" + bsd.ASSISTANT_MARKER + "the answer"
        )
        self.assertEqual(prompt, "userstuff")
        self.assertEqual(resp, "the answer")

    def test_split_missing_marker_raises(self):
        with self.assertRaises(ValueError):
            bsd.split_prompt_response("no marker here")

    def test_extract_question(self):
        self.assertEqual(
            bsd.extract_question("blah Question: Who wrote Hamlet?\n<|im_end|>"),
            "Who wrote Hamlet?",
        )

    def test_extract_question_missing_raises(self):
        with self.assertRaises(ValueError):
            bsd.extract_question("no question marker")

    def test_clean_response_strips_specials(self):
        self.assertEqual(
            bsd.clean_response("<answer>x</answer><|im_end|>\n"),
            "<answer>x</answer>",
        )


class BudgetHelpersTest(unittest.TestCase):
    def test_clamp_budget(self):
        self.assertEqual(bsd.clamp_budget(7), 5)
        self.assertEqual(bsd.clamp_budget(-1), 0)
        self.assertEqual(bsd.clamp_budget(3), 3)

    def test_treatment_response_prepends_tag(self):
        self.assertEqual(
            bsd.treatment_response(2, "body"), "<budget>2</budget>\nbody"
        )


class BuildRowsTest(unittest.TestCase):
    def test_filters_em_correct_only(self):
        records = [
            _trace("Q one?", "<answer>a</answer>", em=1.0),
            _trace("Q two?", "<answer>b</answer>", em=0.0),
        ]
        treatment, control = bsd.build_rows(records)
        self.assertEqual(len(treatment), 1)
        self.assertEqual(len(control), 1)

    def test_symmetric_and_budget_tag_treatment_only(self):
        records = [_trace("Who wrote Hamlet?", "<answer>x</answer>", calls=2)]
        treatment, control = bsd.build_rows(records)
        self.assertTrue(
            treatment[0]["response"].startswith("<budget>2</budget>\n")
        )
        self.assertNotIn("<budget>", control[0]["response"])
        self.assertIn("Who wrote Hamlet?", treatment[0]["prompt"])
        self.assertIn("Who wrote Hamlet?", control[0]["prompt"])

    def test_missing_sequences_str_raises(self):
        with self.assertRaises(ValueError):
            bsd.build_rows([{"em": 1.0, "valid_search_calls": 1}])


class GoldBudgetLabelTest(unittest.TestCase):
    def test_gold_label_used_in_tag(self):
        # trace searched 2 but gold says 4: gold mode declares 4
        records = [_trace("Q?", "<answer>x</answer>", calls=2, gold=4)]
        treatment, _ = bsd.build_rows(records, budget_label="gold")
        self.assertTrue(
            treatment[0]["response"].startswith("<budget>4</budget>\n")
        )

    def test_default_mode_still_uses_calls(self):
        records = [_trace("Q?", "<answer>x</answer>", calls=2, gold=4)]
        treatment, _ = bsd.build_rows(records)
        self.assertTrue(
            treatment[0]["response"].startswith("<budget>2</budget>\n")
        )

    def test_gold_clamped_to_max(self):
        records = [_trace("Q?", "<answer>x</answer>", gold=9)]
        treatment, _ = bsd.build_rows(records, budget_label="gold")
        self.assertTrue(
            treatment[0]["response"].startswith("<budget>5</budget>\n")
        )

    def test_gold_missing_raises(self):
        rec = _trace("Q?", "<answer>x</answer>")
        del rec["gold_budget"]
        with self.assertRaises(ValueError):
            bsd.build_rows([rec], budget_label="gold")

    def test_gold_negative_raises(self):
        records = [_trace("Q?", "<answer>x</answer>", gold=-1)]
        with self.assertRaises(ValueError):
            bsd.build_rows(records, budget_label="gold")

    def test_unknown_label_mode_raises(self):
        records = [_trace("Q?", "<answer>x</answer>")]
        with self.assertRaises(ValueError):
            bsd.build_rows(records, budget_label="bogus")


class ThinkFirstTemplateTest(unittest.TestCase):
    RESPONSE = (
        "I need two facts to answer this.\n"
        "<search>first query</search>\n"
        "<information>Doc 1</information>\n"
        "<search>second query</search>\n"
        "<information>Doc 2</information>\n"
        "<answer>x</answer>"
    )

    def test_budget_spliced_before_first_search(self):
        out = bsd.treatment_response(
            3, self.RESPONSE, budget_template="think_first"
        )
        self.assertIn("<budget>3</budget>\n<search>first query</search>", out)
        self.assertTrue(out.startswith("I need two facts"))
        self.assertEqual(out.count("<budget>"), 1)

    def test_zero_search_trace_splices_before_answer(self):
        out = bsd.treatment_response(
            1, "No search needed.\n<answer>x</answer>",
            budget_template="think_first",
        )
        self.assertIn("<budget>1</budget>\n<answer>x</answer>", out)
        self.assertTrue(out.startswith("No search needed."))

    def test_no_action_tag_raises(self):
        with self.assertRaises(ValueError):
            bsd.treatment_response(
                2, "just prose, no tags", budget_template="think_first"
            )

    def test_unknown_template_raises(self):
        with self.assertRaises(ValueError):
            bsd.treatment_response(2, self.RESPONSE, budget_template="soft")

    def test_build_rows_think_first_prompt_and_response(self):
        records = [
            _trace("Who wrote Hamlet?", self.RESPONSE, calls=2, gold=4)
        ]
        treatment, control = bsd.build_rows(
            records, budget_label="gold", budget_template="think_first"
        )
        expected_prompt = bsd.make_search_prefix(
            "Who wrote Hamlet?", require_budget=True, max_budget=5,
            budget_template="think_first",
        )
        self.assertEqual(treatment[0]["prompt"], expected_prompt)
        self.assertIn(
            "<budget>4</budget>\n<search>first query</search>",
            treatment[0]["response"],
        )
        self.assertFalse(
            treatment[0]["response"].startswith("<budget>")
        )
        self.assertNotIn("<budget>", control[0]["response"])

    def test_build_rows_default_template_unchanged(self):
        records = [_trace("Q?", self.RESPONSE, calls=2)]
        treatment, _ = bsd.build_rows(records)
        self.assertTrue(
            treatment[0]["response"].startswith("<budget>2</budget>\n")
        )


class BalanceRecordsTest(unittest.TestCase):
    def _records(self):
        return [
            _trace("Q1?", "<answer>a</answer>", source="2wikimultihopqa",
                   gold=2),
            _trace("Q2?", "<answer>b</answer>", source="2wikimultihopqa",
                   gold=2),
            _trace("Q3?", "<answer>c</answer>", source="2wikimultihopqa",
                   gold=2),
            _trace("Q4?", "<answer>d</answer>", source="2wikimultihopqa",
                   gold=4),
            _trace("Q5?", "<answer>e</answer>", source="hotpotqa", gold=2),
            _trace("Q6?", "<answer>f</answer>", source="hotpotqa", gold=2,
                   em=0.0),
        ]

    def test_caps_per_source_and_budget(self):
        kept = bsd.balance_records(
            self._records(), budget_label="gold", cap=2, seed=1
        )
        by_group = {}
        for rec in kept:
            key = (rec["data_source"], rec["gold_budget"])
            by_group[key] = by_group.get(key, 0) + 1
        self.assertEqual(by_group[("2wikimultihopqa", 2)], 2)
        self.assertEqual(by_group[("2wikimultihopqa", 4)], 1)
        self.assertEqual(by_group[("hotpotqa", 2)], 1)

    def test_drops_non_em_records(self):
        kept = bsd.balance_records(
            self._records(), budget_label="gold", cap=10, seed=1
        )
        self.assertEqual(len(kept), 5)

    def test_deterministic_for_seed(self):
        first = bsd.balance_records(
            self._records(), budget_label="gold", cap=1, seed=7
        )
        second = bsd.balance_records(
            self._records(), budget_label="gold", cap=1, seed=7
        )
        self.assertEqual(
            [r["sequences_str"] for r in first],
            [r["sequences_str"] for r in second],
        )

    def test_nonpositive_cap_is_noop(self):
        records = self._records()
        self.assertEqual(
            bsd.balance_records(
                records, budget_label="gold", cap=0, seed=1
            ),
            records,
        )


class SplitTrainValTest(unittest.TestCase):
    def test_holds_out_val_fraction(self):
        train, val = bsd.split_train_val(100, seed=1)
        self.assertEqual(len(val), 5)
        self.assertEqual(len(train), 95)
        self.assertEqual(set(train) | set(val), set(range(100)))


if __name__ == "__main__":
    unittest.main()
