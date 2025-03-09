import os
import regex as re
import torch
from torch.utils.data import IterableDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader


# Define CJK character pattern
CJK_PATTERN = r"[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]"

def add_noise_to_cjk(text, tokenizer, mask_prob=0.15, noise_prob=0.1):
    """
    Tokenize text, then randomly replace some CJK characters with:
    - [MASK] with probability `mask_prob`
    - A random incorrect token with probability `noise_prob`
    Outputs:
    - raw token IDs (original text with [CLS] and [SEP])
    - noised token IDs (corrupted text with [CLS] and [SEP])
    - cls_labels (1 for correct, 0 for replaced)
    """
    # Tokenize the text (without special tokens yet)
    tokens = tokenizer.tokenize(text)
    raw_ids = tokenizer.convert_tokens_to_ids(tokens)
    noised_ids = raw_ids.copy()

    # Find indices of CJK characters
    cjk_indices = [i for i, token in enumerate(tokens) if re.search(CJK_PATTERN, token)]

    for i in cjk_indices:
        rand_val = torch.rand(1).item()
        if rand_val < mask_prob:
            # Replace with [MASK]
            noised_ids[i] = tokenizer.mask_token_id
        elif rand_val < mask_prob + noise_prob:
            # Replace with a random token
            noised_ids[i] = torch.randint(671, 7992, (1,)).item()

    # Add special tokens ([CLS] at start, [SEP] at end)
    raw_ids = [tokenizer.cls_token_id] + raw_ids + [tokenizer.sep_token_id]
    noised_ids = [tokenizer.cls_token_id] + noised_ids + [tokenizer.sep_token_id]

    # Compute cls_labels (0 if changed, 1 otherwise)
    cls_labels = [1] + [1 if raw_ids[i] == noised_ids[i] else 0 for i in range(1, len(raw_ids) - 1)] + [1]

    return raw_ids, noised_ids, cls_labels



class StreamingTextDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer_name, max_length=128):
        """
        Args:
        - data_dir (str): Path to folder containing .txt files
        - tokenizer: Hugging Face tokenizer
        - max_length (int): Max sequence length for padding/truncation
        """
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def _line_generator(self):
        """Yield one line at a time from multiple files."""
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()

    def __iter__(self):
        for line in self._line_generator():
            raw_ids, noised_ids, cls_labels = add_noise_to_cjk(line, self.tokenizer)
            
            # Truncate if too long
            raw_ids = raw_ids[:self.max_length]
            noised_ids = noised_ids[:self.max_length]
            cls_labels = cls_labels[:self.max_length]
            
            # Pad if too short
            pad_len = self.max_length - len(raw_ids)
            if pad_len > 0:
                raw_ids += [self.tokenizer.pad_token_id] * pad_len
                noised_ids += [self.tokenizer.pad_token_id] * pad_len
                cls_labels += [1] * pad_len  # Assume padding is "correct"

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in noised_ids]

            yield {
                "input_ids": torch.tensor(noised_ids, dtype=torch.long),  # Noised version
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "mlm_labels": torch.tensor(raw_ids, dtype=torch.long),  # MLM target is the raw input
                "cls_labels": torch.tensor(cls_labels, dtype=torch.float),  # 1 = correct, 0 = replaced
            }


if __name__ == "__main__":

    data_dir = "data/txt/trial"
    max_length = 128  # Set max sequence length
    tokenizer_name = "data/bert"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("data/bert")


    # Create dataset and DataLoader
    dataset = StreamingTextDataset(data_dir, tokenizer_name, max_length)
    dataloader = DataLoader(dataset, batch_size=4)

    # Inspect one batch
    batch = next(iter(dataloader))
    print("Input IDs:", batch["input_ids"].shape)
    print("Attention Mask:", batch["attention_mask"].shape)
    print("MLM Labels:", batch["mlm_labels"].shape)
    print("CLS Labels:", batch["cls_labels"].shape)

    print("Input IDs:", batch["input_ids"][0])
    print("Attention Mask:", batch["attention_mask"][0])
    print("MLM Labels:", batch["mlm_labels"][0])
    print("CLS Labels:", batch["cls_labels"][0])



    # Convert back to text for debugging
    sample_noised = batch["input_ids"][0].tolist()
    sample_raw = batch["mlm_labels"][0].tolist()
    print("Noised Tokens:", tokenizer.convert_ids_to_tokens(sample_noised, skip_special_tokens=False))
    print("Raw Tokens:", tokenizer.convert_ids_to_tokens(sample_raw, skip_special_tokens=True))

    