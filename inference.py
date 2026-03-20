"""Standalone inference script for Bodhi-TTS."""
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path

from bodhi_tts.config import load_config
from bodhi_tts.data.text import CharTokenizer
from bodhi_tts.model.bodhi import BodhiTTS


def main():
    parser = argparse.ArgumentParser(description="Bodhi-TTS Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--description", default="A neutral voice speaking clearly",
                        help="Voice description")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--n-steps", type=int, default=10, help="ODE solver steps")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg, _, data_cfg = load_config(args.model_config, args.train_config, args.data_config)

    # Load tokenizer
    tokenizer = CharTokenizer.from_vocab(data_cfg.cache.vocab_path)
    model_cfg.vocab_size = tokenizer.vocab_size

    # Load model
    model = BodhiTTS(model_cfg)
    ckpt_path = Path(args.checkpoint)
    if (ckpt_path / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path / "model.safetensors"))
        model.load_state_dict(state_dict)
    elif (ckpt_path / "pytorch_model.bin").exists():
        state_dict = torch.load(ckpt_path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # Try accelerate state loading
        from accelerate import load_checkpoint_in_model
        load_checkpoint_in_model(model, str(ckpt_path))

    model = model.to(device).eval()
    print(f"Model loaded from {args.checkpoint}")

    # Encode text
    text_ids = torch.tensor([tokenizer.encode(args.text)], dtype=torch.long, device=device)
    text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.long, device=device)

    # Encode description (load MiniLM for this single use)
    from sentence_transformers import SentenceTransformer
    desc_model = SentenceTransformer(model_cfg.description_encoder.minilm_model)
    desc_np = desc_model.encode([args.description], convert_to_numpy=True)
    desc_embed = torch.from_numpy(desc_np.astype(np.float32)).to(device)
    del desc_model

    # Synthesize
    print(f"Synthesizing: '{args.text}'")
    print(f"Description: '{args.description}'")
    print(f"Steps: {args.n_steps}")

    with torch.no_grad():
        mel, mel_lengths = model.synthesize(text_ids, text_lengths, desc_embed, n_steps=args.n_steps)

    mel_out = mel[0, :, :mel_lengths[0]].cpu()  # [80, T]
    print(f"Generated mel: {mel_out.shape}")

    # Griffin-Lim vocoder
    mel_spec = torch.exp(mel_out)  # undo log
    inv_basis = torchaudio.functional.inverse_mel_scale(
        n_stft=model_cfg.mel.n_fft // 2 + 1,
        n_mels=model_cfg.mel.n_mels,
        sample_rate=model_cfg.mel.sr,
        f_min=model_cfg.mel.f_min,
        f_max=model_cfg.mel.f_max,
    )
    linear_spec = torch.matmul(inv_basis, mel_spec)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=model_cfg.mel.n_fft,
        hop_length=model_cfg.mel.hop_length,
        power=1.0,
    )
    waveform = griffin_lim(linear_spec)

    torchaudio.save(args.output, waveform.unsqueeze(0), model_cfg.mel.sr)
    print(f"Saved: {args.output} ({waveform.shape[0] / model_cfg.mel.sr:.2f}s)")


if __name__ == "__main__":
    main()
