---
language:
- en
license: mit
tags:
- summarization
- abstractive-summarization
- news-summarization
- bart
- cnn-dailymail
- nlp
- pytorch
- transformers
datasets:
- cnn_dailymail
metrics:
- rouge
- bertscore
base_model: facebook/bart-large-cnn
pipeline_tag: summarization
---

# 📰 BART-Large-CNN — Fine-Tuned News Summarizer

**Authors:** Danyal
**Base Model:** [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn)
**Task:** Abstractive News Summarization
**Dataset:** CNN/DailyMail 3.0.0
**Language:** English
**License:** MIT

---

## Model Description

This model is a fine-tuned version of `facebook/bart-large-cnn` on the
**CNN/DailyMail 3.0.0** dataset for abstractive news summarization.
BART (Bidirectional and Auto-Regressive Transformer) is a denoising
autoencoder pre-trained with a corrupted text reconstruction objective,
making it particularly effective for sequence-to-sequence generation tasks
such as summarization.

Fine-tuning was performed on **90,000 training samples** for **3 epochs**
on a Kaggle GPU (T4/P100) using mixed-precision (FP16) training.

---

## Intended Uses & Limitations

### ✅ Intended Uses
- Summarizing English news articles into concise, readable highlights
- Research and benchmarking on abstractive summarization
- Building downstream NLP pipelines that require document compression

### ⚠️ Limitations
- Optimized for **news-style English text** (CNN/DailyMail domain)
- May produce hallucinated facts for out-of-domain or highly technical text
- Not suitable for multilingual summarization without further fine-tuning
- Long documents beyond 512 tokens are truncated

---

## Training Details

| Parameter              | Value                                     |
|------------------------|-------------------------------------------|
| Base model             | facebook/bart-large-cnn                   |
| Dataset                | CNN/DailyMail 3.0.0                       |
| Training samples       | 90,000                                    |
| Validation samples     | 5,000                                     |
| Test samples           | 5,000                                     |
| Epochs                 | 3                                         |
| Train batch size       | 4 (× 4 grad accum = 16 effective)         |
| Learning rate          | 3e-5 (linear warmup 5%)                   |
| Weight decay           | 0.01                                      |
| Max input length       | 512 tokens                                |
| Max target length      | 128 tokens                                |
| Precision              | FP16 (mixed precision)                    |
| Optimizer              | AdamW                                     |
| Hardware               | Kaggle GPU (T4 / P100)                    |
| Framework              | PyTorch + HuggingFace Transformers 4.40.0 |

---

## Evaluation Results

Evaluated on the **CNN/DailyMail test set** (100 batches × 8 = 800 samples).
Scores are on a 0–100 scale.

| Metric     | Zero-Shot Baseline | Fine-Tuned (Ours)       |
|------------|--------------------|-------------------------|
| ROUGE-1    | 43.58   | **44.10**    |
| ROUGE-2    | 21.04   | **20.62**    |
| ROUGE-L    | 31.15   | **30.46**    |
| ROUGE-Lsum | 37.18| **41.04** |

> BERTScore (F1) was also computed on 100 test samples using `bert-score==0.3.13`.

---

## How to Use

### Quick Inference

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "daniB2112/bart-large-cnn-news-summarizer"
tokenizer  = BartTokenizer.from_pretrained(model_name)
model      = BartForConditionalGeneration.from_pretrained(model_name)

article = """
Paste your news article text here. The model accepts up to 512 tokens.
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
    tokenizer="daniB2112/bart-large-cnn-news-summarizer",
)

result = summarizer(article, max_length=80, min_length=30, do_sample=False)
print(result[0]["summary_text"])
```

---

## Generation Parameters

| Parameter              | Value |
|------------------------|-------|
| `max_length`           | 80    |
| `min_length`           | 30    |
| `num_beams`            | 4     |
| `length_penalty`       | 2.0   |
| `early_stopping`       | True  |
| `no_repeat_ngram_size` | 3     |

---

## Training & Evaluation Framework

| Library        | Version  |
|----------------|----------|
| transformers   | 4.40.0   |
| datasets       | 2.19.0   |
| evaluate       | 0.4.1    |
| rouge-score    | 0.1.2    |
| bert-score     | 0.3.13   |
| accelerate     | 0.29.3   |
| sentencepiece  | 0.2.0    |
| PyTorch        | (Kaggle default) |

---

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{daniB2112_bart_summarizer,
  author    = {Danyal},
  title     = {BART-Large-CNN Fine-Tuned on CNN/DailyMail for News Summarization},
  year      = {2024},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/daniB2112/bart-large-cnn-news-summarizer}
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

---

## Model Card Contact

For questions or issues, open a discussion on the HuggingFace model page.
