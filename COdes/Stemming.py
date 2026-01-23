"""
Stemming Implementation
Reduces words to their root/base form by removing suffixes.
Example: running, runs, ran -> run

Implements multiple stemming algorithms:
- Porter Stemmer (most common)
- Snowball Stemmer (improved Porter)
- Lancaster Stemmer (aggressive)
"""

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# Download required data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def porter_stemming(tokens):
    """
    Apply Porter Stemmer to tokens
    
    Args:
        tokens (list): List of word tokens
    
    Returns:
        list: Stemmed tokens
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def snowball_stemming(tokens, language='english'):
    """
    Apply Snowball Stemmer to tokens
    
    Args:
        tokens (list): List of word tokens
        language (str): Language for stemming
    
    Returns:
        list: Stemmed tokens
    """
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(word) for word in tokens]


def lancaster_stemming(tokens):
    """
    Apply Lancaster Stemmer to tokens (most aggressive)
    
    Args:
        tokens (list): List of word tokens
    
    Returns:
        list: Stemmed tokens
    """
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in tokens]


def stem_text(text, stemmer_type='porter'):
    """
    Stem text using specified stemmer
    
    Args:
        text (str): Input text
        stemmer_type (str): 'porter', 'snowball', or 'lancaster'
    
    Returns:
        tuple: (stemmed_tokens, original_tokens)
    """
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Apply stemmer
    if stemmer_type == 'porter':
        stemmed = porter_stemming(tokens)
    elif stemmer_type == 'snowball':
        stemmed = snowball_stemming(tokens)
    elif stemmer_type == 'lancaster':
        stemmed = lancaster_stemming(tokens)
    else:
        raise ValueError(f"Unknown stemmer type: {stemmer_type}")
    
    return stemmed, tokens


def compare_stemmers(words):
    """
    Compare different stemmers on a list of words
    
    Args:
        words (list): List of words to stem
    
    Returns:
        dict: Dictionary with stemming results
    """
    results = {
        'original': words,
        'porter': porter_stemming(words),
        'snowball': snowball_stemming(words),
        'lancaster': lancaster_stemming(words)
    }
    return results


def demonstrate():
    """Demonstrate stemming with different algorithms"""
    
    print("="*80)
    print("STEMMING DEMONSTRATION")
    print("="*80)
    
    # Example words showing stemming
    test_words = [
        'running', 'runs', 'ran', 'runner', 'easily', 'fairly',
        'fairness', 'studies', 'studying', 'study', 'connection',
        'connected', 'connecting', 'connections', 'organization',
        'organize', 'organizing', 'organizational'
    ]
    
    print("\nCOMPARISON OF STEMMING ALGORITHMS:")
    print("-"*80)
    print(f"{'Original':<20} {'Porter':<20} {'Snowball':<20} {'Lancaster':<20}")
    print("-"*80)
    
    results = compare_stemmers(test_words)
    for i, word in enumerate(test_words):
        print(f"{word:<20} {results['porter'][i]:<20} {results['snowball'][i]:<20} {results['lancaster'][i]:<20}")
    
    # Example with sentences
    print("\n" + "="*80)
    print("STEMMING ON SENTENCES:")
    print("="*80)
    
    sentences = [
        "The runners were running in the marathon yesterday.",
        "Studies have shown that studying improves learning outcomes.",
        "The organization organized an organizational meeting."
    ]
    
    for sent in sentences:
        print(f"\nOriginal: {sent}")
        
        porter, tokens = stem_text(sent, 'porter')
        print(f"Tokens: {tokens}")
        print(f"Porter Stemmed: {porter}")
        
        snowball, _ = stem_text(sent, 'snowball')
        print(f"Snowball Stemmed: {snowball}")
    
    # Show differences between stemmers
    print("\n" + "="*80)
    print("KEY DIFFERENCES:")
    print("="*80)
    print("Porter: Most commonly used, moderate stemming")
    print("Snowball: Improved Porter, slightly better accuracy")
    print("Lancaster: Most aggressive, may over-stem")
    print("="*80)


if __name__ == "__main__":
    demonstrate()
