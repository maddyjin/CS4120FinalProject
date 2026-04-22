# Cross-Linguistic Bias Analysis in Wikipedia Articles

## Overview
The goal of this project is to investigate bias in embeddings across a variety of languages, using text from a variety of platforms and data sources. We recognize that the majority of existing NLP research on algorithmic bias is focused on language models and datasets that are in English, and that there is a gap in research that explores multilingual models. With this project we aim to close that gap and provide answers to questions about whether social, cultural, and demographic biases transfer consistently across languages or emerge in language-specific ways. We will do this by measuring, comparing, and analyzing bias patterns when identical or comparable content is processed in multiple languages. We will explore how various factors such as language and culture influence biased outputs.

---

## Objectives
- Analyze bias in multilingual word embeddings
- Compare embedding structures across languages
- Identify whether bias is language-specific or corpus-driven
- Visualize differences using dimensionality reduction techniques

---

## Required Libraries
- `numpy`
- `pandas`
- `nltk`
- `gensim`
- `scikit-learn`
- `matplotlib`
- `plotly`
- `googletrans`
- `wefe`

Required Python version: 3.12 or below

---

## Data
Wikipedia articles.

---

## Models & Methods
Run the following code in order to get the embeddings:

`python scripts/driver.py --lang [LANGUAGE] --topic [TOPIC OR ALL] --buffer [NUMBER]`

Run the `measuring_bias.ipynb` and `scripts/weat_bias.py` files for analysis

---
