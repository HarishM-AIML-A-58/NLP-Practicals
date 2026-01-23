"""
Text Normalization Implementation
Converts text to a standard, consistent format by:
- Converting to lowercase
- Removing URLs and email addresses
- Removing special characters and digits
- Removing extra whitespace
"""

import re
import nltk

def text_normalization(text):
    """
    Normalize text to lowercase, remove special characters and extra spaces
    
    Args:
        text (str): Input text to normalize
    
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep letters and basic punctuation)
    text = re.sub(r'[^a-zA-Z\s.,!?;:\'\"\-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def demonstrate():
    """Demonstrate text normalization"""
    
    sample_texts = [
        "HELLO WORLD! This is a TEST.",
        "Visit https://example.com or email test@example.com for more info.",
        "Text with   extra    spaces   and\n\nnewlines.",
        "Special chars: @#$%^&*()123456789",
        "Furthermore, as an encouragement to revisionist thinking, it manifestly is fair."
    ]
    
    print("="*80)
    print("TEXT NORMALIZATION DEMONSTRATION")
    print("="*80)
    
    for i, text in enumerate(sample_texts, 1):
        normalized = text_normalization(text)
        print(f"\n{i}. Original: {text}")
        print(f"   Normalized: {normalized}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demonstrate()
