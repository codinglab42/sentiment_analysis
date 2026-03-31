#!/usr/bin/env python3
"""
Train sentiment analysis model
"""

import sys
import os
import numpy as np
import pickle as pkl
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor
from src.utils import create_model, evaluate_model, predict_sentiment, find_optimal_threshold
import machine_learning_module as ml

def load_imdb_data(n_samples: int = 5000):
    """Generate synthetic sentiment data with richer vocabulary"""
    np.random.seed(42)
    
    # Più parole positive
    pos_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'happy', 'enjoy', 'best', 'awesome', 'perfect', 'beautiful',
        'brilliant', 'outstanding', 'superb', 'terrific', 'magnificent',
        'delightful', 'pleased', 'satisfied', 'recommend', 'worth'
    ]
    
    # Più parole negative
    neg_words = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointing',
        'hate', 'sad', 'boring', 'waste', 'poor', 'mediocre',
        'dreadful', 'pathetic', 'ridiculous', 'useless', 'wasteful',
        'unpleasant', 'annoying', 'frustrating', 'dislike', 'avoid'
    ]
    
    # Frasi positive più lunghe e varie
    pos_templates = [
        "I really enjoyed this, it was {word}",
        "What a {word} movie!",
        "This is {word}, I {word2} it",
        "Absolutely {word}, highly recommended",
        "The best {word} film I've ever seen",
        "{word} performance, {word2} plot",
        "I'm {word} with this, {word2} job"
    ]
    
    # Frasi negative più lunghe e varie
    neg_templates = [
        "I hated this, it was {word}",
        "What a {word} waste of time",
        "This is {word}, I {word2} it",
        "Absolutely {word}, avoid at all costs",
        "The {word} movie I've ever seen",
        "{word} acting, {word2} direction",
        "I'm {word} with this, {word2} experience"
    ]
    
    texts = []
    labels = []
    
    for i in range(n_samples):
        sentiment = np.random.choice([0, 1])
        
        if sentiment == 1:
            template = np.random.choice(pos_templates)
            word1 = np.random.choice(pos_words)
            word2 = np.random.choice(pos_words)
            text = template.format(word=word1, word2=word2)
            if np.random.random() > 0.5:
                text = text.upper()
            texts.append(text)
            labels.append(1)
        else:
            template = np.random.choice(neg_templates)
            word1 = np.random.choice(neg_words)
            word2 = np.random.choice(neg_words)
            text = template.format(word=word1, word2=word2)
            if np.random.random() > 0.5:
                text = text.upper()
            texts.append(text)
            labels.append(0)
    
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
    preprocessor = TextPreprocessor(max_features=5000, max_len=100)
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
    print(f"   F1 Score:  {f1:.4f}")

    # Calcola la soglia ottimale
    print("\n   Finding optimal threshold...")
    optimal_threshold = find_optimal_threshold(model, X_test, y_test)
    print(f"   Optimal threshold: {optimal_threshold:.2f}")

    # Salva la soglia
    os.makedirs("models", exist_ok=True)
    with open("models/optimal_threshold.pkl", "wb") as f:
        pkl.dump(optimal_threshold, f)
    print(f"   ✅ Saved optimal threshold to models/optimal_threshold.pkl")

    # Ricalcola le metriche con la soglia ottimale
    y_pred_proba = model.predict(X_test).flatten()
    y_pred_opt = (y_pred_proba > optimal_threshold).astype(int)
    y_true = y_test.astype(int)

    tp = np.sum((y_pred_opt == 1) & (y_true == 1))
    fp = np.sum((y_pred_opt == 1) & (y_true == 0))
    fn = np.sum((y_pred_opt == 0) & (y_true == 1))

    precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_opt = 2 * precision_opt * recall_opt / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0
    accuracy_opt = np.mean(y_pred_opt == y_true)

    print(f"\n   With optimal threshold ({optimal_threshold:.2f}):")
    print(f"   Accuracy:  {accuracy_opt:.4f}")
    print(f"   Precision: {precision_opt:.4f}")
    print(f"   Recall:    {recall_opt:.4f}")
    print(f"   F1 Score:  {f1_opt:.4f}")
    
    # 7. Save model and preprocessor
    print("\n7. Saving model and preprocessor...")
    
    model.save("models/sentiment_model.bin")
    print("   ✅ Model saved to models/sentiment_model.bin")
    
    with open("models/preprocessor.pkl", "wb") as f:
        pkl.dump({
            'max_features': preprocessor.max_features,
            'max_len': preprocessor.max_len,
            'word_index': preprocessor.word_index,
            'vocab_size': preprocessor.vocab_size
        }, f)
    print("   ✅ Preprocessor saved to models/preprocessor.pkl\n")
    
    # 8. Test predictions with optimal threshold
    print("8. Testing predictions (with optimal threshold):")
    test_phrases = [
        "This movie is great! I love it!",
        "Terrible film, waste of time",
        "Not bad, but not great either",
        "Absolutely fantastic, best movie ever!",
        "I hated it, so boring"
    ]
    
    for phrase in test_phrases:
        X_test = preprocessor.transform([phrase])
        raw_proba = model.predict(X_test)[0]
        sentiment, confidence = predict_sentiment(model, preprocessor, phrase, threshold=optimal_threshold)
        print(f"   '{phrase[:30]}...' -> {sentiment} (conf: {confidence:.3f}) [raw: {raw_proba:.3f}]")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()