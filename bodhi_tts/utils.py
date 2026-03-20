import os
import sys
import signal
import math
import torch
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR


def create_wsd_scheduler(optimizer, num_training_steps, warmup_ratio=0.05,
                         stable_ratio=0.85, min_lr_ratio=0.01):
    """Warmup-Stable-Decay scheduler."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    stable_steps = int(num_training_steps * stable_ratio)
    decay_steps = num_training_steps - warmup_steps - stable_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        elif current_step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_progress = (current_step - warmup_steps - stable_steps) / max(1, decay_steps)
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * decay_progress)

    return LambdaLR(optimizer, lr_lambda)


def get_wsd_phase(step, num_training_steps, warmup_ratio=0.05, stable_ratio=0.85):
    """Return current WSD phase name."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    stable_steps = int(num_training_steps * stable_ratio)
    if step < warmup_steps:
        return "warmup"
    elif step < warmup_steps + stable_steps:
        return "stable"
    return "decay"


def should_save_checkpoint(step, total_steps, percentages):
    """Check if current step is at any of the percentage milestones."""
    for pct in percentages:
        target_step = int(total_steps * pct / 100)
        if step == target_step:
            return True, pct
    return False, 0


def setup_preemption_handler(accelerator, save_fn):
    """Register SIGTERM handler for GCP preemption.

    Args:
        accelerator: Accelerate accelerator
        save_fn: callable that saves checkpoint to persist dir
    """
    def handler(signum, frame):
        if not accelerator.is_main_process:
            sys.exit(0)
        print("\n" + "!" * 60)
        print("PREEMPTION SIGNAL — saving emergency checkpoint...")
        print("!" * 60)
        try:
            save_fn()
            print("Emergency checkpoint saved.")
        except Exception as e:
            print(f"Emergency save error: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)


def upload_checkpoint_to_gcs(checkpoint_dir, gcs_bucket, run_name, step):
    """Upload checkpoint directory to GCS."""
    if not gcs_bucket:
        return
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        gcs_path = f"{gcs_bucket}/{run_name}/checkpoint-{step}"
        fs.put(str(checkpoint_dir), gcs_path, recursive=True)
        print(f"Uploaded checkpoint to gs://{gcs_path}")
    except Exception as e:
        print(f"GCS upload failed: {e}")


def upload_checkpoint_to_hf(checkpoint_dir, hf_repo, step, pct):
    """Upload checkpoint to HuggingFace Hub."""
    if not hf_repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(hf_repo, exist_ok=True, private=True)
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=hf_repo,
            path_in_repo=f"checkpoint-{step}-{pct}pct",
            commit_message=f"Checkpoint at {pct}% (step {step})",
        )
        print(f"Uploaded checkpoint to HF: {hf_repo}/checkpoint-{step}-{pct}pct")
    except Exception as e:
        print(f"HF upload failed: {e}")


def generate_eval_audio(model, tokenizer, config, step, accelerator, eval_prompts, desc_model=None):
    """Generate evaluation audio with Griffin-Lim and log to W&B.

    Args:
        model: BodhiTTS (unwrapped)
        tokenizer: CharTokenizer
        config: ModelConfig
        step: current training step
        accelerator: Accelerate accelerator
        eval_prompts: list of EvalPrompt
        desc_model: SentenceTransformer for encoding descriptions (loaded on demand)
    """
    import torchaudio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    device = accelerator.device

    # Load description encoder if needed
    if desc_model is None:
        from sentence_transformers import SentenceTransformer
        desc_model = SentenceTransformer(config.description_encoder.minilm_model)

    logs = {}
    for i, prompt in enumerate(eval_prompts):
        text_ids = torch.tensor([tokenizer.encode(prompt.text)], dtype=torch.long, device=device)
        text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.long, device=device)

        desc_np = desc_model.encode([prompt.description], convert_to_numpy=True)
        desc_embed = torch.from_numpy(desc_np.astype(np.float32)).to(device)

        mel, mel_lens = model.synthesize(text_ids, text_lengths, desc_embed)
        mel_out = mel[0, :, :mel_lens[0]].cpu()  # [80, T]

        # Griffin-Lim via pseudo-inverse of mel filterbank
        mel_spec = torch.exp(mel_out)  # undo log
        n_stft = config.mel.n_fft // 2 + 1
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_stft, f_min=config.mel.f_min, f_max=config.mel.f_max,
            n_mels=config.mel.n_mels, sample_rate=config.mel.sr,
        )  # [n_stft, n_mels]
        inv_fb = torch.linalg.pinv(mel_fb.T)  # [n_stft, n_mels]
        linear_spec = torch.matmul(inv_fb, mel_spec).clamp(min=0)  # [n_stft, T]
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config.mel.n_fft, hop_length=config.mel.hop_length, power=1.0,
        )
        waveform = griffin_lim(linear_spec)

        try:
            import wandb
            logs[f"audio/eval_{i}"] = wandb.Audio(
                waveform.numpy(), sample_rate=config.mel.sr,
                caption=f"{prompt.text} | {prompt.description}"
            )

            # Mel spectrogram image
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.imshow(mel_out.numpy(), aspect="auto", origin="lower")
            ax.set_title(f"Step {step}: {prompt.text[:40]}")
            plt.tight_layout()
            logs[f"mel/eval_{i}"] = wandb.Image(fig)
            plt.close(fig)
        except Exception:
            pass

    if logs:
        accelerator.log(logs, step=step)

    model.train()
    del desc_model
