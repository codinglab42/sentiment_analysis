"""
Preprocessing module for text data
"""
import numpy as np
import re
from collections import Counter
from typing import List, Tuple, Dict

class TextPreprocessor:
    """Simple text preprocessing for sentiment analysis"""
    
    def __init__(self, max_features: int = 5000, max_len: int = 100):
        self.max_features = max_features
        self.max_len = max_len
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.vocab_size = 0
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            word_counts.update(words)
        
        # Keep most frequent words
        most_common = word_counts.most_common(self.max_features - 1)
        
        # Reserve index 0 for padding/unknown
        self.word_index = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
        self.index_word = {idx + 1: word for idx, (word, _) in enumerate(most_common)}
        self.vocab_size = len(self.word_index) + 1  # +1 for padding
        
        print(f"✅ Vocabulary size: {self.vocab_size}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            seq = [self.word_index.get(word, 0) for word in words[:self.max_len]]
            
            # Pad or truncate
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[:self.max_len]
            
            sequences.append(seq)
        
        return np.array(sequences)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vocabulary and transform texts"""
        self.fit(texts)
        return self.transform(texts)