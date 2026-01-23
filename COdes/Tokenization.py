"""
Tokenization Implementation
Breaks down text into smaller units (tokens) such as words or sentences.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize

# Download required data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def word_tokenization(text):
    """
    Tokenize text into words
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of word tokens
    """
    if not text:
        return []
    
    return word_tokenize(text)


def sentence_tokenization(text):
    """
    Tokenize text into sentences
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of sentences
    """
    if not text:
        return []
    
    return sent_tokenize(text)


def wordpunct_tokenization(text):
    """
    Tokenize text into words and punctuation
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of tokens including punctuation
    """
    if not text:
        return []
    
    return wordpunct_tokenize(text)


def demonstrate():
    """Demonstrate different tokenization methods"""
    
    sample_text = """Hello World! This is a demonstration of tokenization.
    Tokenization breaks text into smaller units. It's very useful for NLP tasks.
    For example: word_tokenize() splits text into words."""
    
    print("="*80)
    print("TOKENIZATION DEMONSTRATION")
    print("="*80)
    
    print("\nOriginal Text:")
    print(sample_text)
    
    # Word Tokenization
    word_tokens = word_tokenization(sample_text)
    print("\n1. WORD TOKENIZATION:")
    print(f"   Tokens ({len(word_tokens)}): {word_tokens}")
    
    # Sentence Tokenization
    sent_tokens = sentence_tokenization(sample_text)
    print("\n2. SENTENCE TOKENIZATION:")
    for i, sent in enumerate(sent_tokens, 1):
        print(f"   Sentence {i}: {sent}")
    
    # WordPunct Tokenization
    wordpunct_tokens = wordpunct_tokenization(sample_text)
    print("\n3. WORD-PUNCT TOKENIZATION:")
    print(f"   Tokens ({len(wordpunct_tokens)}): {wordpunct_tokens[:30]}...")
    
    # Example with different text
    text2 = "Dr. Smith went to Washington D.C. He met Prof. Johnson at 3:30 PM."
    print("\n4. HANDLING ABBREVIATIONS AND PUNCTUATION:")
    print(f"   Text: {text2}")
    print(f"   Word Tokens: {word_tokenize(text2)}")
    print(f"   Sentence Tokens: {sent_tokenize(text2)}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demonstrate()
