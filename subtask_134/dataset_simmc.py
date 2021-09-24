from pathlib import Path
import linecache
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BartTokenizer
# from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.file_utils import cached_property


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class SimmcDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        src_file,
        tgt_file=None,
        max_source_length=512,
        max_target_length=512,
        prefix="",
        is_lm=True,
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.is_lm = is_lm

        self.src_lens = self.get_char_lens(self.src_file)
        
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        if self.tgt_file is None:
            return 0
        return self.get_char_lens(self.tgt_file)

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n") # + " =>"
        assert source_line, f"empty source line for index {index}"
        if self.tgt_file is None:
            return {"tgt_texts": None, "src_texts": source_line, "id": index - 1}
        
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert tgt_line, f"empty tgt line for index {index}"
        
        if not tgt_line.endswith(self.tokenizer.eos_token):
            tgt_line = tgt_line + " " + self.tokenizer.eos_token        

        # source and tgt same for language modeling
        if self.is_lm:
            source_line = source_line + " " + tgt_line
            tgt_line = source_line

        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}


    

class SimmcDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}


    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"] if "labels" in batch else None,
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch]) if batch[0]["labels"] is not None else None

            if labels is not None:
                labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if isinstance(self.tokenizer, BartTokenizer):
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.tokenizer.bos_token_id) if labels is not None else None
            if decoder_input_ids is not None:
                batch["decoder_input_ids"] = decoder_input_ids

        return batch


    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        # encode input texts
        batch_encoding = self.tokenizer(
            [x["src_texts"] for x in batch],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt", 
            **self.dataset_kwargs,
        )

        if batch[0]["tgt_texts"] is None:
            return batch_encoding

        # Encode tgt_texts
        labels = self.tokenizer(
            [x["tgt_texts"] for x in batch],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
            **self.dataset_kwargs,
        )["input_ids"]
        batch_encoding["labels"] = labels


        return batch_encoding




