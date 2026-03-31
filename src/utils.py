"""
Utility functions for training and evaluation
"""
import numpy as np
import machine_learning_module as ml
from typing import Tuple, List

def create_model(input_size: int, hidden_units: List[int]) -> ml.NeuralNetwork:
    layer_sizes = [input_size] + hidden_units + [1]
    
    network = ml.NeuralNetwork(
        layer_sizes,
        activation="relu",
        output_activation="sigmoid",
        optimizer_type=ml.OptimizerType.ADAM,
        learning_rate=0.01
    )
    network.set_loss_function("binary_crossentropy")
    network.set_epochs(200)  # Aumenta a 100
    network.set_batch_size(64)
    network.set_validation_split(0.2)
    network.set_verbose(True)
    
    return network

def evaluate_model(model: ml.NeuralNetwork, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray) -> Tuple[float, float, float]:
    """Evaluate model and return metrics"""
    
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    y_true = y_test.astype(int)
    
    # Accuracy
    accuracy = np.mean(y_pred_class == y_true)
    
    # Precision, Recall, F1
    tp = np.sum((y_pred_class == 1) & (y_true == 1))
    fp = np.sum((y_pred_class == 1) & (y_true == 0))
    fn = np.sum((y_pred_class == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

def find_optimal_threshold(model, X_val, y_val):
    """Trova la soglia ottimale che massimizza l'F1 score"""
    y_pred_proba = model.predict(X_val).flatten()
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba > threshold).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"   Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold

def predict_sentiment(model, preprocessor, text, threshold=0.5):
    """Predict sentiment with custom threshold"""
    X = preprocessor.transform([text])
    proba = model.predict(X)[0]
    
    # Se la probabilità è troppo bassa, usa la soglia ottimale
    # Ma per ora usiamo threshold personalizzabile
    if proba > threshold:
        sentiment = "NEGATIVE"
        confidence = proba
    else:
        sentiment = "POSITIVE" 
        confidence = 1 - proba
    
    return sentiment, confidence