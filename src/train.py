#!/usr/bin/env python3
"""
Train sentiment analysis model
"""
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor
from src.utils import create_model, evaluate_model, predict_sentiment
import machine_learning_module as ml
import time

def load_imdb_data(n_samples: int = 10000):
    """Load IMDB dataset (simulated)"""
    # In a real scenario, you'd load actual data
    # For demonstration, we'll generate synthetic data
    
    np.random.seed(42)
    
    # Positive words
    pos_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                 'love', 'happy', 'enjoy', 'best', 'awesome', 'perfect']
    
    # Negative words
    neg_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointing',
                 'hate', 'sad', 'boring', 'waste', 'poor', 'awful']
    
    texts = []
    labels = []
    
    for i in range(n_samples):
        # Randomly choose sentiment
        sentiment = np.random.choice([0, 1])
        
        # Build sentence
        if sentiment == 1:
            words = np.random.choice(pos_words, size=np.random.randint(5, 20))
            text = " ".join(words)
        else:
            words = np.random.choice(neg_words, size=np.random.randint(5, 20))
            text = " ".join(words)
        
        texts.append(text)
        labels.append(sentiment)
    
    return np.array(texts), np.array(labels)

def main():
    print("=" * 60)
    print("  SENTIMENT ANALYSIS TRAINING")
    print("=" * 60)
    print()
    
    # 1. Load data
    print("1. Loading data...")
    texts, labels = load_imdb_data(n_samples=5000)
    print(f"   Loaded {len(texts)} samples")
    print(f"   Class distribution: {np.sum(labels == 0)} negative, {np.sum(labels == 1)} positive\n")
    
    # 2. Split data
    print("2. Splitting data...")
    split_idx = int(0.8 * len(texts))
    X_train_texts, X_test_texts = texts[:split_idx], texts[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    print(f"   Train: {len(X_train_texts)}, Test: {len(X_test_texts)}\n")
    
    # 3. Preprocess
    print("3. Preprocessing texts...")
    preprocessor = TextPreprocessor(max_features=2000, max_len=50)
    X_train = preprocessor.fit_transform(X_train_texts)
    X_test = preprocessor.transform(X_test_texts)
    print(f"   Vocabulary size: {preprocessor.vocab_size}")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}\n")
    
    # 4. Create model
    print("4. Creating neural network...")
    model = create_model(
        input_size=preprocessor.max_len,
        hidden_units=[64, 32]
    )
    print(f"   Model summary:")
    model.summary()
    
    # 5. Train model
    print("\n5. Training model...")
    start_time = time.time()
    model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
    elapsed = time.time() - start_time
    print(f"\n   Training time: {elapsed:.2f} seconds\n")
    
    # 6. Evaluate
    print("6. Evaluating on test set...")
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}\n")
    
    # 7. Save model and preprocessor
    print("7. Saving model and preprocessor...")
    os.makedirs("models", exist_ok=True)
    
    model.save("models/sentiment_model.bin")
    print("   ✅ Model saved to models/sentiment_model.bin")
    
    # Save preprocessor state (custom serialization)
    import pickle
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump({
            'max_features': preprocessor.max_features,
            'max_len': preprocessor.max_len,
            'word_index': preprocessor.word_index,
            'vocab_size': preprocessor.vocab_size
        }, f)
    print("   ✅ Preprocessor saved to models/preprocessor.pkl\n")
    
    # 8. Test predictions
    print("8. Testing predictions:")
    test_phrases = [
        "This movie is great! I love it!",
        "Terrible film, waste of time",
        "Not bad, but not great either",
        "Absolutely fantastic, best movie ever!",
        "I hated it, so boring"
    ]
    
    for phrase in test_phrases:
        sentiment, proba = predict_sentiment(model, preprocessor, phrase)
        print(f"   '{phrase[:30]}...' -> {sentiment} (confidence: {proba:.3f})")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()