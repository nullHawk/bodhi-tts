"""Bodhi-TTS training script — custom loop with accelerate for OT-CFM."""
import argparse
import os
import json
import time
import torch
from pathlib import Path
from dotenv import load_dotenv

from accelerate import Accelerator
from accelerate.utils import set_seed

from bodhi_tts.config import load_config
from bodhi_tts.data.text import CharTokenizer
from bodhi_tts.data.dataset import BodhiDataset, BodhiCollator
from bodhi_tts.model.bodhi import BodhiTTS
from bodhi_tts.flow.ot_cfm import sample_and_compute_loss
from bodhi_tts.utils import (
    create_wsd_scheduler, get_wsd_phase, should_save_checkpoint,
    setup_preemption_handler, upload_checkpoint_to_gcs,
    upload_checkpoint_to_hf, generate_eval_audio,
)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Bodhi-TTS Training")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--resume-from", default=None, help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    load_dotenv()

    model_cfg, train_cfg, data_cfg = load_config(args.model_config, args.train_config, args.data_config)
    tc = train_cfg.training
    cc = train_cfg.checkpointing
    lc = train_cfg.logging

    # Init accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=tc.grad_accum,
        mixed_precision="bf16" if tc.bf16 else "no",
        log_with="wandb" if lc.wandb_project else None,
    )

    set_seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    is_main = accelerator.is_main_process

    if is_main:
        print("=" * 70)
        print("BODHI-TTS TRAINING")
        print(f"  GPUs: {accelerator.num_processes}x {torch.cuda.get_device_name(0)}")
        print(f"  Effective batch: {tc.batch_size} x {tc.grad_accum} x {accelerator.num_processes} = "
              f"{tc.batch_size * tc.grad_accum * accelerator.num_processes}")
        print(f"  LR: {train_cfg.optimizer.lr}")
        print(f"  Resume: {args.resume_from or 'fresh'}")
        print("=" * 70)

    # HF login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Load tokenizer (vocab must exist from preprocessing)
    tokenizer = CharTokenizer.from_vocab(data_cfg.cache.vocab_path)
    if is_main:
        print(f"Vocab loaded: {tokenizer.vocab_size} tokens")

    # Set vocab size and build model
    model_cfg.vocab_size = tokenizer.vocab_size
    model = BodhiTTS(model_cfg)

    total_params, trainable_params = count_params(model)
    if is_main:
        print(f"Model params: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")

    if tc.compile:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Dataset
    dataset = BodhiDataset(
        manifest_path=data_cfg.cache.manifest_path,
        tokenizer=tokenizer,
        min_mel_frames=data_cfg.filtering.min_mel_frames,
        max_mel_frames=data_cfg.filtering.max_mel_frames,
        max_text_len=data_cfg.filtering.max_text_len,
    )

    # Train/val split
    val_size = max(1, int(len(dataset) * data_cfg.filtering.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(tc.seed),
    )
    if is_main:
        print(f"Dataset: {train_size} train, {val_size} val")

    collator = BodhiCollator()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=tc.batch_size, shuffle=True,
        collate_fn=collator, num_workers=tc.num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=tc.batch_size, shuffle=False,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )

    # Optimizer — 8-bit AdamW
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=train_cfg.optimizer.lr,
        betas=tuple(train_cfg.optimizer.betas),
        weight_decay=train_cfg.optimizer.weight_decay,
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * tc.epochs
    scheduler = create_wsd_scheduler(
        optimizer, total_steps,
        warmup_ratio=train_cfg.scheduler.warmup_ratio,
        stable_ratio=train_cfg.scheduler.stable_ratio,
        min_lr_ratio=train_cfg.scheduler.min_lr_ratio,
    )

    # Prepare with accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume_from:
        if is_main:
            print(f"Resuming from {args.resume_from}")
        accelerator.load_state(args.resume_from)
        # Recover step from checkpoint dir name or state
        ckpt_name = Path(args.resume_from).name
        if ckpt_name.startswith("checkpoint-"):
            global_step = int(ckpt_name.split("-")[1])
            start_epoch = global_step // steps_per_epoch
            if is_main:
                print(f"Resumed at step {global_step}, epoch {start_epoch}")

    # Preemption handler
    def emergency_save():
        persist_dir = Path(cc.persist_dir) / "preempted"
        persist_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(persist_dir))

    setup_preemption_handler(accelerator, emergency_save)

    # W&B init
    if lc.wandb_project:
        accelerator.init_trackers(
            project_name=lc.wandb_project,
            config={
                "model": model_cfg.__dict__,
                "training": tc.__dict__,
                "optimizer": train_cfg.optimizer.__dict__,
            },
            init_kwargs={"wandb": {"name": lc.wandb_run_name}},
        )

    if is_main:
        print(f"\nTraining: {tc.epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")
        print(f"Checkpoint milestones: {cc.percentages}%\n")

    # Training loop
    dur_loss_weight = train_cfg.flow.dur_loss_weight
    step_times = []

    for epoch in range(start_epoch, tc.epochs):
        model.train()
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            t0 = time.time()

            with accelerator.accumulate(model):
                total_loss, flow_loss, dur_loss = sample_and_compute_loss(
                    model, batch, dur_loss_weight
                )
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                    global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_time = time.time() - t0
            step_times.append(step_time)

            # Logging
            if accelerator.sync_gradients and is_main and global_step % lc.log_every == 0:
                progress_pct = global_step / total_steps * 100
                avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])
                phase = get_wsd_phase(global_step, total_steps,
                                      train_cfg.scheduler.warmup_ratio,
                                      train_cfg.scheduler.stable_ratio)

                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/flow_loss": flow_loss.item(),
                    "train/dur_loss": dur_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/epoch": epoch,
                    "wsd/phase_idx": {"warmup": 0, "stable": 1, "decay": 2}[phase],
                    "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "speed/progress_pct": progress_pct,
                    "speed/steps_per_sec": 1.0 / avg_step_time,
                }
                accelerator.log(log_dict, step=global_step)

                print(f"[{global_step}/{total_steps} ({progress_pct:.1f}%)] "
                      f"loss={total_loss.item():.4f} flow={flow_loss.item():.4f} "
                      f"dur={dur_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"phase={phase} {1/avg_step_time:.1f} steps/s")

            # Audio eval
            if accelerator.sync_gradients and is_main and global_step % lc.eval_every == 0 and global_step > 0:
                unwrapped = accelerator.unwrap_model(model)
                try:
                    generate_eval_audio(
                        unwrapped, tokenizer, model_cfg, global_step,
                        accelerator, train_cfg.eval_prompts,
                    )
                except Exception as e:
                    print(f"Eval audio failed: {e}")

            # Checkpoint
            if accelerator.sync_gradients:
                save, pct = should_save_checkpoint(global_step, total_steps, cc.percentages)
                if save:
                    ckpt_dir = Path(cc.output_dir) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.save_state(str(ckpt_dir))

                    if is_main:
                        print(f"\nCheckpoint saved at {pct}%: {ckpt_dir}")

                        # Persist copy
                        persist_ckpt = Path(cc.persist_dir) / f"checkpoint-{global_step}"
                        persist_ckpt.mkdir(parents=True, exist_ok=True)
                        accelerator.save_state(str(persist_ckpt))

                        # GCS upload
                        upload_checkpoint_to_gcs(
                            ckpt_dir, cc.gcs_bucket,
                            lc.wandb_run_name or "bodhi-tts", global_step
                        )

                        # HF upload
                        upload_checkpoint_to_hf(ckpt_dir, cc.hf_repo, global_step, pct)

                        # Validation
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.eval()
                        val_losses = []
                        with torch.no_grad():
                            for val_batch in val_loader:
                                tl, fl, dl = sample_and_compute_loss(
                                    unwrapped, val_batch, dur_loss_weight
                                )
                                val_losses.append((tl.item(), fl.item(), dl.item()))
                        if val_losses:
                            avg_tl = sum(x[0] for x in val_losses) / len(val_losses)
                            avg_fl = sum(x[1] for x in val_losses) / len(val_losses)
                            avg_dl = sum(x[2] for x in val_losses) / len(val_losses)
                            accelerator.log({
                                "eval/total_loss": avg_tl,
                                "eval/flow_loss": avg_fl,
                                "eval/dur_loss": avg_dl,
                            }, step=global_step)
                            print(f"Val: total={avg_tl:.4f} flow={avg_fl:.4f} dur={avg_dl:.4f}")
                        unwrapped.train()

        if is_main:
            print(f"\n--- Epoch {epoch + 1}/{tc.epochs} complete ---\n")

    # Final save
    if is_main:
        final_dir = Path(cc.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(final_dir))
        print(f"Final model saved: {final_dir}")

    accelerator.end_training()
    if is_main:
        print("Training complete!")


if __name__ == "__main__":
    main()
