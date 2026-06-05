# CheXmix Setup & Inference

This guide details how to download the necessary weights, configure the environment, and run the demo for report generation and image embedding extraction.

## 1. Download Weights

Both the CheXmix model weights and VQ-GAN checkpoint are hosted on Hugging Face Hub at [StanfordMIMI/CheXmix](https://huggingface.co/StanfordMIMI/CheXmix) and are **downloaded automatically** when you instantiate the model. No manual download or path configuration is needed.

## 2. Configuration

No configuration is required for default usage — weights are fetched from HF Hub and downloaded to the repo root directory on first run.

To use custom local weight paths instead, pass a `CheXmixConfig` explicitly:

```python
from chexmix.configs.default_configs import CheXmixConfig, VQGANConfig
from chexmix.models.load import CheXmix

config = CheXmixConfig(
    model_path="/your/path/to/model.safetensors",
    vq_gan_config=VQGANConfig(vq_gan_ckpt_path="/your/path/to/vqgan.ckpt"),
)
model = CheXmix(config=config)
```

## 3. Running the Demo

A demonstration script, `demo.py`, is provided to showcase Report Generation and Image Embeddings.

To run the demo on a specific GPU (e.g., GPU 0), use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py

```

## 4. Features & Troubleshooting

### Report Generation

The demo will output a medical report based on the corresponding input image.

- **Troubleshooting:** If the generated report is too short, too long, or cuts off, try adjusting the `max_new_tokens` or `min_new_tokens` parameters in `chexmix/configs/default_configs.py`.
- **Note:** This is a pre-trained model and has not been explicitly instruction-tuned. As a result, it may not follow complex prompt instructions as strictly as an instruction-tuned model would.

### Image Embeddings

The model can output embeddings from each layer of the network.

- **Recommendation:** For most discriminative tasks (such as classification or retrieval), we recommend using embeddings from the **early to middle layers**.
- **Best Layers:** Specifically, layers **7 through 12** tend to yield the best performance.
