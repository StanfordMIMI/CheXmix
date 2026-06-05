import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig
from pathlib import Path

from chexmix.configs.default_configs import CheXmixConfig
from chexmix.utils import (
    ImageTokenizer,
    load_pil_image,
    truncate_at_special,
    load_checkpoint,
)


class CheXmix(nn.Module):
    def __init__(
        self,
        config: CheXmixConfig | None = None,
        ReportGeneration: bool = False,
        ImageEmbeddings: bool = False,
    ):
        super(CheXmix, self).__init__()

        if config is None:
            config = CheXmixConfig()

        self.cfg = config

        model_kwargs = {
            "ReportGeneration": ReportGeneration,
            "ImageEmbeddings": ImageEmbeddings,
        }

        self.tokenizer, self.model = self._load_model(**model_kwargs)
        self.vq_gan = self._load_vq_gan()

    def _load_vq_gan(self):
        return ImageTokenizer(
            self.cfg.vq_gan_config.vq_gan_path,
            self.cfg.vq_gan_config.vq_gan_ckpt_path,
            self.cfg.vq_gan_config.vq_gan_device,
        )

    def _load_model(self, **model_kwargs):
        # Resolve path relative to this file's directory (chexmix/models/)
        _RADPHI2_DIR = Path(__file__).resolve().parent / "radphi2"

        tokenizer = AutoTokenizer.from_pretrained(_RADPHI2_DIR, trust_remote_code=True)
        # Load ONLY the configuration
        config = AutoConfig.from_pretrained(_RADPHI2_DIR)
        if model_kwargs["ImageEmbeddings"]:
            model = AutoModel.from_config(config)
        else:
            model = AutoModelForCausalLM.from_config(config)

        model.resize_token_embeddings(
            self.cfg.ORIGINAL_VOCAB_SIZE
            + self.cfg.VQGAN_VOCAB_SIZE
            + self.cfg.PADDING_TOKEN
        )

        model.to(torch.bfloat16).cuda()
        missing_keys, unexpected_keys = model.load_state_dict(
            load_checkpoint(self.cfg.model_path, **model_kwargs), strict=False
        )
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)

        return tokenizer, model

    """
    Return Image Tokens from Image Path
    """

    def _load_image_tokens(self, image_path):
        if isinstance(image_path, list):
            images = [load_pil_image(img_path) for img_path in image_path]
        else:
            images = [load_pil_image(image_path)]

        image_token_lst = [
            self.vq_gan.img_tokens_from_pil(image).detach().cpu() for image in images
        ]
        incremented_image_token_lst = [
            (image_token + self.cfg.ORIGINAL_VOCAB_SIZE).tolist()
            for image_token in image_token_lst
        ]
        return incremented_image_token_lst

    """
    Generate Embeddings from Image Path
    Return a lst if image_path is a lst input
    @input: Image Path can be a single image path or a list of image paths
    @output: List of embeddings
    """

    def generate_embeddings(self, image_path):
        self.model.eval()
        image_token_lst = self._load_image_tokens(image_path)
        image_token_start_end_lst = [
            [self.cfg.IMG_TOKEN_START_VAL] + image_tensor + [self.cfg.IMG_TOKEN_END_VAL]
            for image_tensor in image_token_lst
        ]
        image_tensor_cuda_lst = [
            torch.tensor(
                image_token_start_end_lst, dtype=torch.long, device="cuda"
            ).unsqueeze(0)
            for image_token_start_end_lst in image_token_start_end_lst
        ]
        with torch.no_grad():
            outputs_lst = [
                self.model(image_tensor_cuda, output_hidden_states=True)
                for image_tensor_cuda in image_tensor_cuda_lst
            ]

        embedding_lst = []
        for outputs in outputs_lst:
            embeddings = {}
            for layer, output in enumerate(outputs.hidden_states):
                image_embeddings = output[0, 1:-1, :].detach().cpu()
                image_embeddings = image_embeddings.mean(dim=0)
                embeddings[layer] = image_embeddings
            embedding_lst.append(embeddings)

        return embedding_lst

    """
    Generate Report from Image Path
    @input: Image Path can be a single image path or a list of image paths
    @output: List of reports
    """

    def generate_reports(self, image_path):
        self.model.eval()
        image_token_lst = self._load_image_tokens(image_path)
        image_token_start_end_lst = [
            [self.cfg.IMG_TOKEN_START_VAL]
            + image_token
            + [self.cfg.IMG_TOKEN_END_VAL]
            + [self.cfg.ENDOFTEXT_TOKEN_VAL]
            for image_token in image_token_lst
        ]
        image_token_start_end_tensor = [
            torch.tensor(image_token_start_end_lst, device="cuda:0")
            for image_token_start_end_lst in image_token_start_end_lst
        ]

        outputs_return_lst = []
        for image_tokens in image_token_start_end_tensor:
            with torch.no_grad():
                outputs = self.model.generate(
                    image_tokens.unsqueeze(0),
                    max_new_tokens=self.cfg.max_new_tokens,
                    min_new_tokens=self.cfg.min_new_tokens,
                    eos_token_id=self.cfg.ENDOFTEXT_TOKEN_VAL,
                    pad_token_id=self.cfg.ENDOFTEXT_TOKEN_VAL,
                )

                model_text_output = outputs[0, 1027:]  # Remove batch dimension

                model_text_output = truncate_at_special(model_text_output)

                model_text_output.clamp_(
                    0, self.cfg.ORIGINAL_VOCAB_SIZE - 1
                )  # Ensure tokens are valid

                model_text_output = self.tokenizer.decode(model_text_output.tolist())

                outputs_return_lst.append(model_text_output)

        return outputs_return_lst
