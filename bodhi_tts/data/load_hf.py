import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List
from tqdm import tqdm


def load_hf_dataset(dataset_id: str, split: str, cache_dir: str,
                    text_field: str = "text",
                    description_field: str = "description",
                    source_name: str = "hf") -> List[dict]:
    """Load HF dataset, save audio arrays as WAV files, return sample dicts."""
    from datasets import load_dataset

    print(f"Loading HF dataset: {dataset_id} (split={split})")
    ds = load_dataset(dataset_id, split=split)
    print(f"Loaded {len(ds)} samples")

    audio_dir = Path(cache_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    skipped = 0
    for idx in tqdm(range(len(ds)), desc=f"Processing {source_name}"):
        row = ds[idx]
        text = row.get(text_field, "")
        if not text or not text.strip():
            skipped += 1
            continue

        sample_id = f"{source_name}_{idx}"
        audio_path = audio_dir / f"{sample_id}.wav"

        # Save audio array to WAV if not cached
        if not audio_path.exists():
            audio_data = row.get("audio", {})
            if isinstance(audio_data, dict):
                array = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]
            else:
                skipped += 1
                continue
            sf.write(str(audio_path), array, sr)

        samples.append({
            "id": sample_id,
            "source": source_name,
            "text": text.strip(),
            "description": row.get(description_field, "neutral"),
            "audio_path": str(audio_path),
        })

    print(f"[{source_name}] Extracted {len(samples)} samples ({skipped} skipped)")
    return samples
