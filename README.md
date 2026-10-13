# Arabic Dialect Classification Using TF-IDF and MARBERT

A comparative NLP project exploring how traditional machine learning and transformer-based models perform in identifying Arabic dialects from text.

## Project Overview

Arabic is characterized by a wide range of regional dialects that vary in vocabulary, expressions, and writing conventions. These variations make automatic dialect identification a challenging natural language processing task.

This project compares two text classification approaches: a TF-IDF representation combined with Logistic Regression, and a fine-tuned MARBERT model. The objective is to evaluate whether contextual language representations provide better dialect classification performance than a traditional machine learning baseline.

## Dataset and Preparation

The experiment uses a dataset of **112,000 Arabic text records** distributed across **16 dialect categories**, covering regional varieties such as Saudi, Egyptian, Iraqi, and Moroccan Arabic, alongside Modern Standard Arabic.

The preprocessing workflow includes:
- Exploring the dataset and its label distribution.
- Removing records with missing text or labels.
- Normalizing Arabic text while retaining relevant dialectal characteristics.
- Splitting the data into training and validation sets using an 85:15 stratified split.

## Modeling Approach

### 1. TF-IDF with Logistic Regression

This model serves as the baseline for comparison. Arabic text is represented through TF-IDF features using unigrams and bigrams, with a maximum vocabulary size of 50,000 features. Logistic Regression with class balancing is then used to predict the dialect category.

### 2. MARBERT Fine-Tuning

MARBERT is a transformer-based language model designed for Arabic and its dialectal variations. It is fine-tuned on the classification dataset to capture contextual and linguistic patterns that may help distinguish between dialects.

The documented configuration includes:
- Learning rate: `1e-5`
- Batch size: `16`
- Gradient accumulation: `2` steps
- Training epochs: `2`
- Maximum sequence length: `64` tokens
- Mixed-precision training: FP16

## Results and Comparison

Both approaches are evaluated using Accuracy and Macro F1-score, supported by confusion-matrix analysis to examine classification patterns across dialects.

| Metric | TF-IDF + Logistic Regression | MARBERT |
|---|---:|---:|
| Accuracy | 72% | 79% |
| Macro F1-score | 60% | 67% |

MARBERT outperformed the TF-IDF baseline by **7 percentage points in both reported metrics**. This indicates that contextual representations were more effective in the evaluated setup. Confusion-matrix analysis also provides insight into errors involving linguistically similar dialects.

These results should be interpreted in light of the dataset distribution and validation methodology; they do not necessarily establish how the models will perform on unseen data from other sources.

## Interactive Demo

The project also includes a Gradio interface that allows users to enter an Arabic sentence and view the model's predicted dialect along with a confidence score.

This interface provides a practical way to explore the classifier's predictions beyond the training and evaluation workflow.

## Technologies

- **Programming:** Python
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-learn, TF-IDF, Logistic Regression
- **Deep Learning and NLP:** MARBERT, Hugging Face Transformers
- **Interface:** Gradio
- **Development Environment:** Google Colab

## Project Files and Resources

- **Google Colab:** [Open the implementation notebook](https://colab.research.google.com/drive/17AFzrdWwsLayfIL_rEz0XM03upbiW8Lg?usp=sharing)
- **Notebook:** `Arabic_Dialect_Identification.ipynb`
- **Report:** `Arabic_Dialect_Identification_Report.pdf`

*The notebook and report should be included in the repository under the filenames listed above.*

## Future Directions

Potential improvements include:
- Testing on an independent dataset to assess generalization.
- Examining precision, recall, and F1-score for each dialect.
- Investigating confusion between closely related dialects.
- Improving evaluation for underrepresented categories.
- Documenting the model's inference requirements and reproducibility steps.

## Academic Context

**Course:** Applied Natural Language Processing  
**Institution:** Umm Al-Qura University
