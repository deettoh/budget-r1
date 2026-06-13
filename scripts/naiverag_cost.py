"""Correct naive-RAG TTC/CES by counting pre-injected passage tokens.

naive-RAG injects the top-3 passages at data-prep so the harness logs
retrieved_tokens=0 and undercounts cost. Counts the injected tokens
per data_source and reports TTC = generated + injected, comparable to
the agentic baseline.
"""

import argparse
import json
import re
from collections import defaultdict

_INFO = re.compile(r"<information>(.*?)</information>", re.DOTALL)


def last_information_span(content: str) -> str:
    """Return the last <information> block (the injected passages).

    The naive-RAG instruction line literally mentions "<information>"
    and "</information>", so the first regex match captures that
    instruction text, not the docs. The injected passages are the
    last span.
    """
    spans = _INFO.findall(content)
    return spans[-1] if spans else ""


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    import pandas as pd
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    df = pd.read_parquet(a.parquet)
    inj = defaultdict(list)
    for prompt, src in zip(df["prompt"], df["data_source"]):
        text = last_information_span(prompt[0]["content"])
        ntok = len(tok(text, add_special_tokens=False)["input_ids"])
        inj[src].append(ntok)
    inj_mean = {s: _mean(v) for s, v in inj.items()}
    print("per-source injected-token means:", {
        s: round(v, 1) for s, v in inj_mean.items()})

    recs = json.load(open(a.eval_json))
    bysrc = defaultdict(list)
    for r in recs:
        bysrc[r["data_source"]].append(r)
    groups = [("all", recs)] + [(s, bysrc[s]) for s in sorted(bysrc)]

    head = (f"{'group':<18}{'n':>5}{'EM':>7}{'F1':>7}{'gen':>8}"
            f"{'inj':>8}{'TTC':>9}{'CES':>7}")
    print(head)
    print("-" * len(head))
    for g, rs in groups:
        em = _mean([r["em"] for r in rs])
        f1 = _mean([r["f1"] for r in rs])
        gen = _mean([r["generated_tokens"] for r in rs])
        if g == "all":
            injm = _mean([inj_mean.get(r["data_source"], 0.0)
                          for r in rs])
        else:
            injm = inj_mean.get(g, 0.0)
        ttc = gen + injm
        ces = f1 * 1000 / ttc if ttc else 0.0
        print(f"{g:<18}{len(rs):>5}{em:>7.3f}{f1:>7.3f}{gen:>8.1f}"
              f"{injm:>8.1f}{ttc:>9.1f}{ces:>7.3f}")


if __name__ == "__main__":
    main()
