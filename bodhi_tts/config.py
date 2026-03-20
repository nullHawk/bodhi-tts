from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class TextEncoderConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class DescriptionEncoderConfig:
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    minilm_dim: int = 384
    proj_dim: int = 256
    freeze: bool = True


@dataclass
class DurationPredictorConfig:
    hidden_dim: int = 256
    n_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.5


@dataclass
class DecoderConfig:
    in_channels: int = 80
    d_model: int = 256
    channels_mult: List[int] = field(default_factory=lambda: [1, 2, 4])
    n_res_blocks: int = 2
    n_heads: int = 4


@dataclass
class MelConfig:
    sr: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: int = 0
    f_max: int = 12000


@dataclass
class ModelConfig:
    vocab_size: Optional[int] = None
    char_embed_dim: int = 256
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    description_encoder: DescriptionEncoderConfig = field(default_factory=DescriptionEncoderConfig)
    duration_predictor: DurationPredictorConfig = field(default_factory=DurationPredictorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    mel: MelConfig = field(default_factory=MelConfig)


@dataclass
class OptimizerConfig:
    type: str = "adamw_8bit"
    lr: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    type: str = "wsd"
    warmup_ratio: float = 0.05
    stable_ratio: float = 0.85
    min_lr_ratio: float = 0.01


@dataclass
class TrainingParams:
    epochs: int = 200
    batch_size: int = 32
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    bf16: bool = True
    num_workers: int = 10
    seed: int = 42
    compile: bool = False


@dataclass
class CheckpointConfig:
    percentages: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    output_dir: str = "/workspace/checkpoints/bodhi-tts"
    persist_dir: str = "/workspace/persist/bodhi-tts"
    gcs_bucket: Optional[str] = None
    hf_repo: Optional[str] = None


@dataclass
class LoggingConfig:
    log_every: int = 10
    eval_every: int = 500
    wandb_project: str = "bodhi-tts"
    wandb_run_name: Optional[str] = None


@dataclass
class EvalPrompt:
    text: str = ""
    description: str = ""


@dataclass
class FlowConfig:
    n_inference_steps: int = 10
    dur_loss_weight: float = 0.1


@dataclass
class TrainConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval_prompts: List[EvalPrompt] = field(default_factory=list)
    flow: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class DataSourceConfig:
    name: str = ""
    type: str = ""
    metadata_path: str = ""
    dataset_id: str = ""
    split: str = "train"
    text_field: str = "text"
    text_field_fallback: str = ""
    description_field: str = "description"


@dataclass
class CacheConfig:
    base_dir: str = "/workspace/cache/bodhi-tts"
    audio_dir: str = "/workspace/cache/bodhi-tts/audio"
    mel_dir: str = "/workspace/cache/bodhi-tts/mel"
    desc_dir: str = "/workspace/cache/bodhi-tts/desc_embeddings"
    manifest_path: str = "/workspace/cache/bodhi-tts/manifest.jsonl"
    vocab_path: str = "/workspace/cache/bodhi-tts/vocab.json"


@dataclass
class FilteringConfig:
    min_mel_frames: int = 20
    max_mel_frames: int = 2000
    max_text_len: int = 400
    val_split: float = 0.02


@dataclass
class DataConfig:
    sources: List[DataSourceConfig] = field(default_factory=list)
    cache: CacheConfig = field(default_factory=CacheConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)


def _dict_to_dataclass(cls, d):
    """Recursively convert a dict to a nested dataclass."""
    if d is None:
        return cls()
    import dataclasses
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in fieldtypes:
            continue
        ft = fieldtypes[k]
        # Resolve string type annotations
        if isinstance(ft, str):
            ft = eval(ft)
        origin = getattr(ft, '__origin__', None)
        if origin is list:
            args = getattr(ft, '__args__', ())
            if args and dataclasses.is_dataclass(args[0]):
                kwargs[k] = [_dict_to_dataclass(args[0], item) for item in v]
            else:
                kwargs[k] = v
        elif dataclasses.is_dataclass(ft):
            kwargs[k] = _dict_to_dataclass(ft, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(model_yaml: str, train_yaml: str, data_yaml: str):
    """Load configs from YAML files into dataclasses."""
    with open(model_yaml) as f:
        model_dict = yaml.safe_load(f)
    with open(train_yaml) as f:
        train_dict = yaml.safe_load(f)
    with open(data_yaml) as f:
        data_dict = yaml.safe_load(f)

    model_cfg = _dict_to_dataclass(ModelConfig, model_dict)
    train_cfg = _dict_to_dataclass(TrainConfig, train_dict)
    data_cfg = _dict_to_dataclass(DataConfig, data_dict)

    return model_cfg, train_cfg, data_cfg
