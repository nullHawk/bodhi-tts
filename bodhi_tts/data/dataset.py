import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


class BodhiDataset(Dataset):
    """Dataset that loads from a preprocessed manifest.jsonl."""

    def __init__(self, manifest_path: str, tokenizer, min_mel_frames: int = 20,
                 max_mel_frames: int = 2000, max_text_len: int = 400):
        self.tokenizer = tokenizer
        self.samples = []

        with open(manifest_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                n_mel = entry["n_mel_frames"]
                n_chars = entry["n_chars"]
                if min_mel_frames <= n_mel <= max_mel_frames and n_chars <= max_text_len:
                    self.samples.append(entry)

        print(f"BodhiDataset: {len(self.samples)} samples after filtering")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        text_ids = torch.tensor(self.tokenizer.encode(entry["text"]), dtype=torch.long)
        mel = torch.load(entry["mel_path"], weights_only=True)  # [80, T_mel]
        desc_embed = torch.load(entry["desc_embed_path"], weights_only=True)  # [384]

        return {
            "text_ids": text_ids,
            "mel": mel,
            "desc_embed": desc_embed,
            "text_length": text_ids.shape[0],
            "mel_length": mel.shape[1],
        }


class BodhiCollator:
    """Pads batch to max lengths."""

    def __call__(self, batch: List[dict]) -> dict:
        text_lengths = torch.tensor([b["text_length"] for b in batch], dtype=torch.long)
        mel_lengths = torch.tensor([b["mel_length"] for b in batch], dtype=torch.long)

        max_text_len = text_lengths.max().item()
        max_mel_len = mel_lengths.max().item()
        batch_size = len(batch)

        text_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        mel = torch.zeros(batch_size, 80, max_mel_len)
        desc_embed = torch.stack([b["desc_embed"] for b in batch])  # [B, 384]

        for i, b in enumerate(batch):
            tl = b["text_length"]
            ml = b["mel_length"]
            text_ids[i, :tl] = b["text_ids"]
            mel[i, :, :ml] = b["mel"]

        return {
            "text_ids": text_ids,
            "mel": mel,
            "desc_embed": desc_embed,
            "text_lengths": text_lengths,
            "mel_lengths": mel_lengths,
        }
