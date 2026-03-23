# Character-Level RNN Name Classification

A PyTorch implementation of a character-level recurrent neural network (RNN) for classifying surnames by language of origin.

## Project Overview

This project builds and trains an RNN that learns to classify names based on their spelling across 18 different languages. The model reads names as sequences of characters and predicts the language of origin.

## Features

- Character-level RNN architecture with PyTorch
- Training on 18 language surname datasets
- Confusion matrix analysis for model evaluation
- Prediction interface for testing on new names

## Setup

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download `data.zip` from https://download.pytorch.org/tutorial/data.zip
   - Extract it to the project root directory (creates `data/names/` folder)

## Usage

Run the Jupyter notebook to train and evaluate the model:
```bash
jupyter notebook char_rnn_classification_tutorial.ipynb
```

The notebook includes:
- Data preprocessing
- Model architecture definition
- Training loop with loss tracking
- Evaluation with confusion matrix
- Prediction on custom names

## Model Details

- **Architecture**: 2-layer RNN with hidden size of 128
- **Input**: One-hot encoded character vectors
- **Output**: LogSoftmax probabilities over 18 language categories
- **Training**: 100,000 iterations with learning rate 0.005
- **Loss Function**: Negative Log Likelihood (NLLLoss)

## Results

The model achieves good performance across most languages, with particularly strong results on Greek and weaker performance on English (due to overlap with other languages).

## Requirements

- torch
- matplotlib

See `requirements.txt` for exact versions.

## References

Based on PyTorch tutorial by Sean Robertson:
https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
