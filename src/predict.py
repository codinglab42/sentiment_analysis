#!/usr/bin/env python3
"""
Interactive sentiment prediction
"""
import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import machine_learning_module as ml
from src.preprocessing import TextPreprocessor
from src.utils import predict_sentiment

def load_model_and_preprocessor():
    """Load saved model and preprocessor"""
    model = ml.NeuralNetwork()
    model.load("models/sentiment_model.bin")
    
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor_data = pickle.load(f)
    
    preprocessor = TextPreprocessor(
        max_features=preprocessor_data['max_features'],
        max_len=preprocessor_data['max_len']
    )
    preprocessor.word_index = preprocessor_data['word_index']
    preprocessor.vocab_size = preprocessor_data['vocab_size']
    
    # Carica anche la soglia ottimale se salvata
    threshold = 0.5
    threshold_file = "models/optimal_threshold.pkl"
    if os.path.exists(threshold_file):
        with open(threshold_file, "rb") as f:
            threshold = pickle.load(f)
        print(f"✅ Using optimal threshold: {threshold:.2f}")
    
    return model, preprocessor, threshold

def main():
    print("=" * 60)
    print("  SENTIMENT ANALYSIS - INTERACTIVE MODE")
    print("=" * 60)
    print()
    
    # Load model
    print("Loading model...")
    try:
        model, preprocessor, threshold = load_model_and_preprocessor()
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run train.py first.")
        return
    
    print("Enter a sentence to analyze sentiment (type 'quit' to exit):")
    print("-" * 60)
    
    while True:
        text = input("\n>> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        try:
            sentiment, proba = predict_sentiment(model, preprocessor, text, threshold=threshold)
            emoji = "😊" if sentiment == "POSITIVE" else "😞"
            print(f"   {emoji} {sentiment} (confidence: {proba:.3f})")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main()