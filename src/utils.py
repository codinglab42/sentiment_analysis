"""
Utility functions for training and evaluation
"""
import numpy as np
import machine_learning_module as ml
from typing import Tuple, List

def create_model(input_size: int, hidden_units: List[int]) -> ml.NeuralNetwork:
    """Create a neural network for binary sentiment classification"""
    
    # Build layer sizes: input -> hidden layers -> output
    layer_sizes = [input_size] + hidden_units + [1]
    
    network = ml.NeuralNetwork(
        layer_sizes,
        activation="relu",
        output_activation="sigmoid",
        optimizer_type=ml.OptimizerType.ADAM,
        learning_rate=0.001
    )
    network.set_loss_function("binary_crossentropy")
    network.set_epochs(50)
    network.set_batch_size(32)
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

def predict_sentiment(model: ml.NeuralNetwork,
                      preprocessor,
                      text: str,
                      threshold: float = 0.5) -> Tuple[str, float]:
    """Predict sentiment for a single text"""
    
    # Preprocess
    X = preprocessor.transform([text])
    
    # Predict
    proba = model.predict(X)[0]
    
    # Determine sentiment
    sentiment = "POSITIVE" if proba > threshold else "NEGATIVE"
    
    return sentiment, proba