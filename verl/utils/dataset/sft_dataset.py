"""SFT dataset with retrieved-span loss masking.

Prompt/response parquet loader for the SFT cold-start. The chat
template is applied to the prompt so tokens match the RL rollout; loss
covers response tokens only, and with mask_information_spans the
<information>..</information> spans are excluded so SFT obeys the same
never-train-on-retrieved-tokens rule as RL.
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask

from search_r1.sft_masking import build_loss_mask, information_span_flags


class SFTDataset(Dataset):
    """Prompt/response SFT dataset with retrieved-span loss masking."""

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key: str = "prompt",
        prompt_dict_keys=None,
        response_key: str = "response",
        response_dict_keys=None,
        max_length: int = 1024,
        truncation: str = "error",
        mask_information_spans: bool = False,
    ) -> None:
        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_length = max_length
        self.truncation = truncation
        self.mask_information_spans = mask_information_spans
        self._open_ids = tokenizer.encode(
            "<information>", add_special_tokens=False
        )
        self._close_ids = tokenizer.encode(
            "</information>", add_special_tokens=False
        )
        self._download_and_read()

    def _download_and_read(self) -> None:
        for i, path in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(path)
        frames = [pd.read_parquet(p) for p in self.parquet_files]
        self.dataframe = pd.concat(frames)
        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.responses = self.dataframe[self.response_key].tolist()

    def __len__(self) -> int:
        return len(self.prompts)

    def _pad_or_truncate(self, input_ids, attention_mask):
        """Right-pad to max_length, else truncate per the mode."""
        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            pad = self.max_length - seq_len
            input_ids = torch.cat(
                (
                    input_ids,
                    torch.full(
                        (pad,), self.tokenizer.pad_token_id,
                        dtype=input_ids.dtype,
                    ),
                ),
                dim=-1,
            )
            attention_mask = torch.cat(
                (attention_mask, torch.zeros(pad, dtype=attention_mask.dtype)),
                dim=-1,
            )
        elif seq_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_length "
                    f"{self.max_length}"
                )
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
        return input_ids, attention_mask

    def __getitem__(self, item: int) -> dict:
        tokenizer = self.tokenizer
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": self.prompts[item]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        response_str = self.responses[item] + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        response_ids = tokenizer(
            response_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.ones_like(input_ids)

        info_keep = None
        if self.mask_information_spans:
            info_keep = information_span_flags(
                response_ids.tolist(), self._open_ids, self._close_ids
            )

        input_ids, attention_mask = self._pad_or_truncate(
            input_ids, attention_mask
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        loss_mask = torch.tensor(
            build_loss_mask(
                prompt_length, response_length, self.max_length, info_keep
            ),
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
