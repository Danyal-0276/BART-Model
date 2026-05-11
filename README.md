# 📰 BART-Large-CNN Fine-Tuned News Summarizer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-4.40.0-FFD21E?style=flat-square&logo=huggingface)
![Dataset](https://img.shields.io/badge/Dataset-CNN%2FDailyMail-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?style=flat-square&logo=kaggle)

**Fine-tuning `facebook/bart-large-cnn` on CNN/DailyMail for abstractive news summarization.**

[🤗 Model on HuggingFace](#) · [📓 Kaggle Notebook](#) · [📊 Results](#evaluation-results)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Evaluation Results](#evaluation-results)
- [How to Use](#how-to-use)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Key Implementation Details](#key-implementation-details)
- [Requirements](#requirements)
- [Running on Kaggle](#running-on-kaggle)
- [Output Files](#output-files)
- [Authors](#authors)
- [Citation](#citation)

---

## Overview

This project fine-tunes [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) on the **CNN/DailyMail 3.0.0** dataset to improve abstractive news summarization performance. BART is a sequence-to-sequence model pre-trained with a denoising objective, making it well-suited for generation tasks like summarization.

The full training pipeline was built and run on **Kaggle GPU** (T4/P100), with careful attention to the 20 GB disk quota constraint. The pipeline includes exploratory data analysis, zero-shot baseline evaluation, fine-tuning with mixed-precision, ROUGE + BERTScore evaluation, and an interactive training dashboard.

**Key highlights:**
- Trained on 90,000 CNN/DailyMail samples across 3 epochs
- Mixed-precision (FP16) training with gradient accumulation
- Disk-space-aware checkpointing designed for Kaggle's 20 GB limit
- Full ROUGE-1/2/L/Lsum + BERTScore evaluation
- Interactive Plotly dashboard for training diagnostics
- Model published to HuggingFace Hub

---

## Project Structure

```
bart-news-summarizer/
│
├── bart-summarization-kaggle.ipynb   # Main training notebook
│
├── outputs/                          # Generated after training
│   ├── bart_news_summarizer/         # Fine-tuned model (HF format)
│   │   ├── config.json
│   │   ├── pytorch_model.bin / model.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   └── README.md                 # HuggingFace model card
│   ├── checkpoints/
│   │   └── ckpt_best.pt              # Best checkpoint by ROUGE-1
│   ├── figures/
│   │   ├── eda_plots.png             # Dataset analysis plots
│   │   ├── training_curves.png       # Loss + ROUGE per epoch
│   │   ├── quality_analysis.png      # Summary quality distribution
│   │   ├── bertscore_analysis.png    # BERTScore distribution
│   │   └── dashboard.html            # Interactive Plotly dashboard
│   └── results/
│       └── final_results.json        # Baseline vs fine-tuned metrics
│
└── README.md
```

---

## Dataset

**CNN/DailyMail 3.0.0** — a standard benchmark dataset for abstractive summarization consisting of news articles paired with multi-sentence highlights (summaries).

| Split      | Full Size  | Used       |
|------------|-----------|------------|
| Train      | 287,113   | 90,000     |
| Validation | 13,368    | 5,000      |
| Test       | 11,490    | 5,000      |

**Key statistics (computed on 5,000 training samples):**

| Statistic          | Articles  | Summaries |
|--------------------|-----------|-----------|
| Mean length (words)| ~780      | ~56       |
| Median length      | ~700      | ~52       |
| Compression ratio  | ~14× on average      |

The dataset was loaded via HuggingFace `datasets` and shuffled with a fixed seed (42) before sampling.

---

## Model Architecture

**Base:** [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn)

BART (Lewis et al., 2019) is a denoising autoencoder built with a standard Transformer encoder-decoder architecture. It is pre-trained by corrupting text with an arbitrary noising function and then learning to reconstruct the original text. `bart-large-cnn` is the large variant already fine-tuned by Meta on CNN/DailyMail, which we further fine-tune.

| Component         | Details                        |
|-------------------|--------------------------------|
| Architecture      | Transformer Encoder-Decoder    |
| Parameters        | ~406M                          |
| Encoder layers    | 12                             |
| Decoder layers    | 12                             |
| Hidden size       | 1024                           |
| Attention heads   | 16                             |
| Vocabulary size   | 50,265                         |
| Max position embs | 1024                           |

---

## Training Setup

### Hyperparameters

| Parameter              | Value                          |
|------------------------|--------------------------------|
| Epochs                 | 3                              |
| Train batch size       | 4                              |
| Gradient accumulation  | 4 steps (effective batch = 16) |
| Learning rate          | 3e-5                           |
| LR schedule            | Linear warmup (5%) + decay     |
| Weight decay           | 0.01                           |
| Max grad norm          | 1.0 (gradient clipping)        |
| Max input length       | 512 tokens                     |
| Max target length      | 128 tokens                     |
| Precision              | FP16 mixed precision           |
| Optimizer              | AdamW                          |
| Seed                   | 42                             |

### Generation Parameters (Inference)

| Parameter              | Value |
|------------------------|-------|
| `max_length`           | 80    |
| `min_length`           | 30    |
| `num_beams`            | 4     |
| `length_penalty`       | 2.0   |
| `early_stopping`       | True  |
| `no_repeat_ngram_size` | 3     |

### Hardware

- **Platform:** Kaggle (T4 x2 or P100)
- **VRAM:** 16 GB
- **Disk:** 20 GB quota (aggressively managed)

### Disk Space Strategy

Kaggle's 20 GB working directory limit required special handling:
- HuggingFace cache redirected to `/kaggle/working/hf_cache/`
- Only **1 rolling step checkpoint** kept on disk at any time
- Epoch checkpoints saved in **lightweight mode** (model weights only, no optimizer state)
- Best checkpoint saved separately, always preserved
- FP16 checkpoint saving to halve checkpoint size (~1.6 GB → ~0.8 GB)

---

## Evaluation Results

Evaluated on the CNN/DailyMail **test set** (800 samples). ROUGE scores are on a 0–100 scale.

| Metric      | Zero-Shot Baseline | Fine-Tuned    | Δ Improvement |
|-------------|-------------------|---------------|---------------|
| ROUGE-1     | 43.58             | **44.10**     | +0.52         |
| ROUGE-2     | 21.04             | 20.62         | -0.42         |
| ROUGE-L     | 31.15             | 30.46         | -0.69         |
| ROUGE-Lsum  | 37.18             | **41.04**     | +3.86         |

> The base model (`facebook/bart-large-cnn`) was already pre-trained on CNN/DailyMail, so the zero-shot baseline is strong. Fine-tuning improves ROUGE-1 and ROUGE-Lsum notably, while ROUGE-2 and ROUGE-L remain competitive.

**BERTScore** (semantic similarity, 100 test samples):

| Metric       | Score |
|--------------|-------|
| Precision    | —     |
| Recall       | —     |
| F1           | —     |

> BERTScore uses `roberta-large` as the reference model via `bert-score==0.3.13`.

---

## How to Use

### Install Dependencies

```bash
pip install transformers==4.40.0 datasets==2.19.0 evaluate==0.4.1 \
            bert_score==0.3.13 rouge_score==0.1.2 accelerate==0.29.3 \
            sentencepiece==0.2.0 torch
```

### Load from HuggingFace

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "daniB2112/bart-large-cnn-news-summarizer"
tokenizer  = BartTokenizer.from_pretrained(model_name)
model      = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()
```

### Summarize an Article

```python
article = """
Your news article text goes here. The model accepts up to 512 tokens.
Longer articles will be truncated automatically.
"""

inputs = tokenizer(
    article,
    max_length=512,
    truncation=True,
    return_tensors="pt"
)

summary_ids = model.generate(
    **inputs,
    max_length=80,
    min_length=30,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True,
    no_repeat_ngram_size=3,
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

### Using the Pipeline API

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="daniB2112/bart-large-cnn-news-summarizer",
)

result = summarizer(
    article,
    max_length=80,
    min_length=30,
    do_sample=False,
)
print(result[0]["summary_text"])
```

### Batch Inference

```python
import torch
from torch.utils.data import DataLoader

articles = ["Article 1 text...", "Article 2 text...", "Article 3 text..."]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

inputs = tokenizer(
    articles,
    max_length=512,
    truncation=True,
    padding=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    summary_ids = model.generate(
        **inputs,
        max_length=80,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
for i, s in enumerate(summaries):
    print(f"[{i+1}] {s}\n")
```

---

## Notebook Walkthrough

The training notebook is organized into 18 cells:

| Cell | Description |
|------|-------------|
| 0  | Disk audit — check available space before starting |
| 1  | Install dependencies |
| 2  | Imports, config (`CFG` dict), path setup, space utilities |
| 3  | Load CNN/DailyMail dataset and sample splits |
| 4  | Exploratory data analysis — length distributions, compression ratios |
| 5  | Tokenizer + `NewsDataset` class (PyTorch Dataset) |
| 6  | Load base model, detect and resume from checkpoint if available |
| 7  | Disk-space-aware checkpoint utilities |
| 8  | Zero-shot baseline ROUGE evaluation |
| 9  | Main training loop (FP16, grad accum, step checkpointing) |
| 9b | Emergency space recovery (run only if near 20 GB limit) |
| 10 | Training curves (loss + ROUGE per epoch) |
| 11 | Final test set ROUGE evaluation |
| 12 | Summary quality analysis plots |
| 13 | BERTScore computation and analysis |
| 14 | Inference examples on test samples |
| 15 | Save and deploy model to HuggingFace Hub |
| 16 | Interactive Plotly dashboard |
| 17 | Package all outputs into a ZIP for download |
| 18 | Generate HuggingFace model card (README.md) |

---

## Key Implementation Details

### Custom Dataset Class

The `NewsDataset` class tokenizes articles and summaries separately using `as_target_tokenizer()` context to avoid the `max_target_length not recognized` warning in `transformers==4.40.0`. Padding tokens in labels are replaced with `-100` so they are ignored by the cross-entropy loss.

```python
class NewsDataset(Dataset):
    def __getitem__(self, idx):
        encoding = tokenizer(article, max_length=512, ...)
        with tokenizer.as_target_tokenizer():
            target  = tokenizer(summary, max_length=128, ...)
        labels  = target['input_ids'].squeeze().clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {'input_ids': ..., 'attention_mask': ..., 'labels': labels}
```

### Mixed Precision Training

FP16 training via `torch.amp.autocast` + `GradScaler`, with a fallback to `torch.cuda.amp` for older PyTorch versions.

### Gradient Accumulation

Effective batch size of 16 is achieved by accumulating gradients over 4 steps with a physical batch size of 4, keeping VRAM usage manageable on T4/P100.

### Checkpoint Management

- Step checkpoints saved every 500 steps, keeping only the latest 1
- Best checkpoint (by ROUGE-1) always preserved separately
- Epoch checkpoints saved in lightweight mode (model weights only)
- All saves guarded by a minimum free disk space check (3 GB threshold)

### ROUGE Evaluation

```python
def evaluate_rouge(model, loader, tokenizer, num_batches=25):
    # Generates summaries batch by batch
    # Decodes with skip_special_tokens=True
    # Computes ROUGE via evaluate.load('rouge') with stemming
    # Returns rouge1, rouge2, rougeL, rougeLsum (×100)
```

---

## Requirements

```
torch>=2.0
transformers==4.40.0
datasets==2.19.0
evaluate==0.4.1
rouge_score==0.1.2
bert_score==0.3.13
accelerate==0.29.3
sentencepiece==0.2.0
nltk
numpy
pandas
matplotlib
seaborn
plotly
tqdm
huggingface_hub
```

---

## Running on Kaggle

1. Create a new Kaggle notebook and upload `bart-summarization-kaggle.ipynb`
2. **Settings → Accelerator → GPU T4 x2** (or P100)
3. **Settings → Internet → On**
4. **Add-ons → Secrets → Add** `HF_TOKEN` (your HuggingFace write token)
5. Run cells **top to bottom**, skipping Cell 9b unless you run low on disk
6. After training, run Cells 10–18 for evaluation, analysis, and export

> **Tip:** Run Cell 0 first to check your baseline free space. You need at least 15 GB free before starting.

---

## Output Files

After a full run the following files are produced under `/kaggle/working/bart_finetuning/`:

| File | Description |
|------|-------------|
| `bart_news_summarizer/` | Fine-tuned model in HuggingFace format |
| `checkpoints/ckpt_best.pt` | Best model checkpoint (by ROUGE-1) |
| `eda_plots.png` | Dataset exploratory analysis plots |
| `training_curves.png` | Loss and ROUGE curves per epoch |
| `quality_analysis.png` | Per-sample ROUGE-1 distribution + radar chart |
| `bertscore_analysis.png` | BERTScore F1 distribution + ROUGE correlation |
| `dashboard.html` | Interactive Plotly training dashboard |
| `final_results.json` | Baseline vs fine-tuned ROUGE scores |
| `README.md` | HuggingFace model card |

---

## Authors

**Danyal & Nauman Irshad**

---

## Citation

If you use this work, please cite:

```bibtex
@misc{danyal_nauman_bart_summarizer,
  author    = {Danyal and Nauman Irshad},
  title     = {BART-Large-CNN Fine-Tuned on CNN/DailyMail for News Summarization},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/your-username/your-repo}
}
```

### Original BART Paper

```bibtex
@article{lewis2019bart,
  title   = {BART: Denoising Sequence-to-Sequence Pre-training for
             Natural Language Generation, Translation, and Comprehension},
  author  = {Lewis, Mike and Liu, Yinhan and Goyal, Naman and
             Ghahraman, Marjan and Mohamed, Abdelrahman and Chen, Danqi and
             Ott, Myle and Gimpel, Kevin and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal = {arXiv preprint arXiv:1910.13461},
  year    = {2019}
}
```

### CNN/DailyMail Dataset

```bibtex
@inproceedings{see2017get,
  title     = {Get to the point: Summarization with pointer-generator networks},
  author    = {See, Abigail and Liu, Peter J and Manning, Christopher D},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
  year      = {2017}
}
```

---

<div align="center">
Made with ❤️ using HuggingFace Transformers and Kaggle GPU
</div>
