"""
CheXmix Demo Script
===================

This script serves as a demonstration of the core capabilities of the CheXmix model:
Medical Report Generation and Image Embedding Extraction.

The script performs two main tasks sequentially:
1. Report Generation: Initializes the model to generate a textual radiology
   report based on a sample chest X-ray image. The output is formatted and
   displayed using the `rich` library.
2. Embedding Extraction: Re-initializes the model to extract latent
   feature representations (embeddings) from the same image across specific
   transformer layers.

Usage
-----
Ensure that `assets/demo_img.png` exists and that your configuration files
point to the correct model checkpoints before running.

Command Line:
    $ CUDA_VISIBLE_DEVICES=0 python demo.py
"""

from chexmix import CheXmix
from rich.console import Console
from rich.panel import Panel
import os

img_path = "assets/demo_img.png"
# create absolute path
img_path = os.path.join(os.path.dirname(__file__), img_path)

print("Generating report...")
model = CheXmix(ReportGeneration=True)

report = model.generate_reports(img_path)

console = Console()
console.print(Panel(report[0].strip(), title="Report", border_style="dim"))

print("--------------------------------")

print("Generating embeddings...")
chexmix = CheXmix(ImageEmbeddings=True)

embeddings = chexmix.generate_embeddings(img_path)

for layer, embedding in embeddings[0].items():
    print(f"Layer {layer}: Embedding Shape {embedding.shape}")
