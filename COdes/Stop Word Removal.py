"""
Stop Word Removal Implementation
Removes common words (stop words) that don't carry significant meaning.
Examples: the, is, at, which, on, etc.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def remove_stopwords(tokens, language='english', remove_punctuation=True):
    """
    Remove stop words from a list of tokens
    
    Args:
        tokens (list): List of word tokens
        language (str): Language for stop words (default: 'english')
        remove_punctuation (bool): Whether to remove punctuation
    
    Returns:
        list: Filtered tokens without stop words
    """
    if not tokens:
        return []
    
    stop_words = set(stopwords.words(language))
    
    filtered_tokens = []
    for word in tokens:
        # Check if word should be kept
        if word.lower() not in stop_words:
            if remove_punctuation:
                if word not in string.punctuation and len(word) > 1:
                    filtered_tokens.append(word)
            else:
                filtered_tokens.append(word)
    
    return filtered_tokens


def remove_stopwords_from_text(text, language='english'):
    """
    Remove stop words from text (tokenizes first)
    
    Args:
        text (str): Input text
        language (str): Language for stop words
    
    Returns:
        tuple: (filtered_tokens, original_tokens)
    """
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    filtered = remove_stopwords(tokens, language)
    
    return filtered, tokens


def get_stopwords_list(language='english'):
    """
    Get the list of stop words for a language
    
    Args:
        language (str): Language name
    
    Returns:
        set: Set of stop words
    """
    return set(stopwords.words(language))


def demonstrate():
    """Demonstrate stop word removal"""
    
    print("="*80)
    print("STOP WORD REMOVAL DEMONSTRATION")
    print("="*80)
    
    # Show English stop words
    english_stopwords = get_stopwords_list('english')
    print(f"\nTotal English stop words: {len(english_stopwords)}")
    print(f"Sample stop words: {list(english_stopwords)[:20]}")
    
    # Example 1: Simple sentence
    text1 = "The quick brown fox jumps over the lazy dog in the park."
    filtered1, original1 = remove_stopwords_from_text(text1)
    
    print("\n" + "-"*80)
    print("EXAMPLE 1:")
    print(f"Original text: {text1}")
    print(f"Original tokens ({len(original1)}): {original1}")
    print(f"Filtered tokens ({len(filtered1)}): {filtered1}")
    print(f"Removed: {set(original1) - set(filtered1)}")
    
    # Example 2: Complex sentence
    text2 = """Furthermore, as an encouragement to revisionist thinking, 
    it manifestly is fair to admit that any fraternity has a constitutional right."""
    filtered2, original2 = remove_stopwords_from_text(text2)
    
    print("\n" + "-"*80)
    print("EXAMPLE 2:")
    print(f"Original text: {text2}")
    print(f"Original tokens ({len(original2)}): {original2}")
    print(f"Filtered tokens ({len(filtered2)}): {filtered2}")
    
    # Example 3: Showing impact
    text3 = "This is a test. The test is important for the evaluation."
    filtered3, original3 = remove_stopwords_from_text(text3)
    
    print("\n" + "-"*80)
    print("EXAMPLE 3 - Impact Analysis:")
    print(f"Original: {text3}")
    print(f"Original tokens: {original3}")
    print(f"After removal: {filtered3}")
    print(f"Reduction: {len(original3)} -> {len(filtered3)} tokens ({100 - len(filtered3)/len(original3)*100:.1f}% removed)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demonstrate()
