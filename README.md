# Sentiment Analysis with ML Library

A complete sentiment analysis example using the ML Library.

## Features
- Text preprocessing (cleaning, tokenization, vectorization)
- Neural network training
- Model evaluation (accuracy, precision, recall, F1)
- Interactive prediction

## Usage

### Train the model
```bash
python src/train.py

### Interactive prediction
```bash
python src/predict.py

### Example Output
```text

Enter a sentence to analyze sentiment:
>> This movie is great! I love it!
    POSITIVE (confidence: 0.923)

>> Terrible film, waste of time
    NEGATIVE (confidence: 0.856)

### Project Structure
```text

sentiment_analysis/
├── data/                    # Dataset
├── models/                  # Saved models
├── src/
│   ├── preprocessing.py     # Text preprocessing
│   ├── train.py            # Training script
│   ├── predict.py          # Interactive prediction
│   └── utils.py            # Utilities
└── requirements.txt