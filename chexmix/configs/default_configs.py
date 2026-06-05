from dataclasses import dataclass, field
import os
from huggingface_hub import hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

HF_REPO = "StanfordMIMI/CheXmix"


@dataclass
class VQGANConfig:
    vq_gan_path: str = os.path.join(BASE_DIR, "vq_gan.yaml")
    vq_gan_ckpt_path: str = field(
        default_factory=lambda: hf_hub_download(repo_id=HF_REPO, filename="vqgan.ckpt", local_dir=REPO_DIR)
    )
    vq_gan_device: str = "cuda:0"


# Default configuration for CheXmix model
@dataclass
class CheXmixConfig:
    model_path: str = field(
        default_factory=lambda: hf_hub_download(repo_id=HF_REPO, filename="model.safetensors", local_dir=REPO_DIR)
    )
    vq_gan_config: VQGANConfig = field(default_factory=VQGANConfig)
    ORIGINAL_VOCAB_SIZE = 50368
    VQGAN_VOCAB_SIZE = 8192
    PADDING_TOKEN = 2
    IMG_TOKEN_START_VAL = 50295
    IMG_TOKEN_END_VAL = 50296
    ENDOFTEXT_TOKEN_VAL = 50256
    max_new_tokens = 200
    min_new_tokens = 5
