"""Standalone preprocessing: download audio, compute mels, embed descriptions, write manifest."""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from bodhi_tts.config import load_config
from bodhi_tts.data.mel import MelProcessor
from bodhi_tts.data.text import build_vocab
from bodhi_tts.data.load_gcs import load_gcs_metadata, download_audio_parallel, extract_gcs_samples
from bodhi_tts.data.load_hf import load_hf_dataset


def compute_desc_embeddings(samples, cache_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Pre-compute MiniLM description embeddings and cache as .pt files."""
    from sentence_transformers import SentenceTransformer

    desc_dir = Path(cache_dir)
    desc_dir.mkdir(parents=True, exist_ok=True)

    # Check which need computing
    to_compute = []
    for s in samples:
        path = desc_dir / f"{s['id']}.pt"
        if not path.exists():
            to_compute.append(s)

    if not to_compute:
        print(f"All {len(samples)} description embeddings already cached")
        return

    print(f"Computing {len(to_compute)} description embeddings (MiniLM)...")
    model = SentenceTransformer(model_name)

    batch_size = 256
    for i in tqdm(range(0, len(to_compute), batch_size), desc="Desc embeddings"):
        batch = to_compute[i:i + batch_size]
        texts = [s["description"] for s in batch]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        for s, emb in zip(batch, embeddings):
            path = desc_dir / f"{s['id']}.pt"
            torch.save(torch.from_numpy(emb.astype(np.float32)), path)

    del model
    torch.cuda.empty_cache()
    print(f"Description embeddings complete: {len(to_compute)} computed")


def main():
    parser = argparse.ArgumentParser(description="Bodhi-TTS Data Preprocessing")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    model_cfg, _, data_cfg = load_config(args.model_config, args.train_config, args.data_config)
    cache = data_cfg.cache

    # Create cache dirs
    for d in [cache.base_dir, cache.audio_dir, cache.mel_dir, cache.desc_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Phase 1: Download/extract audio from all sources
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Loading audio from all sources")
    print("=" * 60)

    all_samples = []
    for source in data_cfg.sources:
        if source.type == "gcs":
            entries = load_gcs_metadata(source.metadata_path)
            audio_map = download_audio_parallel(entries, cache.base_dir)
            samples = extract_gcs_samples(
                entries, audio_map, source.name,
                text_field=source.text_field,
                text_field_fallback=source.text_field_fallback,
                description_field=source.description_field,
            )
            all_samples.extend(samples)
        elif source.type == "hf":
            samples = load_hf_dataset(
                source.dataset_id, source.split, cache.base_dir,
                text_field=source.text_field,
                description_field=source.description_field,
                source_name=source.name,
            )
            all_samples.extend(samples)

    print(f"\nTotal samples after Phase 1: {len(all_samples)}")

    # ================================================================
    # Phase 2: Compute mel spectrograms
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Computing mel spectrograms")
    print("=" * 60)

    mel_proc = MelProcessor(
        sr=model_cfg.mel.sr,
        n_fft=model_cfg.mel.n_fft,
        hop_length=model_cfg.mel.hop_length,
        n_mels=model_cfg.mel.n_mels,
        f_min=model_cfg.mel.f_min,
        f_max=model_cfg.mel.f_max,
    )

    valid_samples = []
    for s in tqdm(all_samples, desc="Computing mels"):
        try:
            mel, mel_path = mel_proc.compute_and_cache(s["audio_path"], cache.mel_dir, s["id"])
            s["mel_path"] = mel_path
            s["n_mel_frames"] = mel.shape[1]
            valid_samples.append(s)
        except Exception as e:
            pass  # skip corrupted audio

    print(f"Valid samples after mel computation: {len(valid_samples)}")

    # ================================================================
    # Phase 3: Pre-compute description embeddings
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Computing description embeddings")
    print("=" * 60)

    compute_desc_embeddings(valid_samples, cache.desc_dir, model_cfg.description_encoder.minilm_model)

    for s in valid_samples:
        s["desc_embed_path"] = str(Path(cache.desc_dir) / f"{s['id']}.pt")

    # ================================================================
    # Phase 4: Build vocab + write manifest
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Building vocab and writing manifest")
    print("=" * 60)

    all_texts = [s["text"] for s in valid_samples]
    vocab = build_vocab(all_texts, cache.vocab_path)

    # Write manifest
    with open(cache.manifest_path, "w") as f:
        for s in valid_samples:
            s["n_chars"] = len(s["text"])
            # Don't persist audio_path in manifest — not needed during training
            entry = {k: v for k, v in s.items() if k != "audio_path"}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nPreprocessing complete!")
    print(f"  Manifest: {cache.manifest_path} ({len(valid_samples)} samples)")
    print(f"  Vocab: {cache.vocab_path} ({len(vocab)} tokens)")
    print(f"  Mel cache: {cache.mel_dir}")
    print(f"  Desc cache: {cache.desc_dir}")


if __name__ == "__main__":
    main()
