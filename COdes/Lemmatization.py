"""
Lemmatization Implementation
Reduces words to their dictionary base form (lemma) using vocabulary and morphological analysis.
More accurate than stemming but slower.

Example: 
- better -> good
- running -> run
- was -> be
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download required data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(treebank_tag):
    """
    Convert TreeBank POS tags to WordNet POS tags
    
    Args:
        treebank_tag (str): TreeBank POS tag
    
    Returns:
        str: WordNet POS tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def lemmatize_tokens(tokens, use_pos=False):
    """
    Lemmatize a list of tokens
    
    Args:
        tokens (list): List of word tokens
        use_pos (bool): Whether to use POS tagging for better accuracy
    
    Returns:
        list: Lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    
    if use_pos:
        # Get POS tags
        pos_tags = nltk.pos_tag(tokens)
        # Lemmatize with POS tags
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
                     for word, pos in pos_tags]
    else:
        # Simple lemmatization (assumes noun)
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    return lemmatized


def lemmatize_text(text, use_pos=False):
    """
    Lemmatize text
    
    Args:
        text (str): Input text
        use_pos (bool): Whether to use POS tagging
    
    Returns:
        tuple: (lemmatized_tokens, original_tokens)
    """
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Lemmatize
    lemmatized = lemmatize_tokens(tokens, use_pos)
    
    return lemmatized, tokens


def compare_with_different_pos(word):
    """
    Show how lemmatization differs with POS tags
    
    Args:
        word (str): Word to lemmatize
    
    Returns:
        dict: Lemmatization results for different POS
    """
    lemmatizer = WordNetLemmatizer()
    
    return {
        'original': word,
        'noun': lemmatizer.lemmatize(word, wordnet.NOUN),
        'verb': lemmatizer.lemmatize(word, wordnet.VERB),
        'adjective': lemmatizer.lemmatize(word, wordnet.ADJ),
        'adverb': lemmatizer.lemmatize(word, wordnet.ADV)
    }


def demonstrate():
    """Demonstrate lemmatization"""
    
    print("="*80)
    print("LEMMATIZATION DEMONSTRATION")
    print("="*80)
    
    # Example 1: Compare lemmatization with different POS
    print("\n1. LEMMATIZATION WITH DIFFERENT POS TAGS:")
    print("-"*80)
    
    test_words = ['running', 'better', 'was', 'caring', 'studies', 'flies']
    
    print(f"{'Word':<15} {'Noun':<15} {'Verb':<15} {'Adj':<15} {'Adv':<15}")
    print("-"*80)
    
    for word in test_words:
        results = compare_with_different_pos(word)
        print(f"{results['original']:<15} {results['noun']:<15} {results['verb']:<15} "
              f"{results['adjective']:<15} {results['adverb']:<15}")
    
    # Example 2: Lemmatization vs Stemming
    print("\n2. LEMMATIZATION VS STEMMING:")
    print("-"*80)
    
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    
    comparison_words = ['studies', 'studying', 'better', 'good', 'was', 'caring', 'organization']
    
    print(f"{'Original':<20} {'Stemmed':<20} {'Lemmatized':<20}")
    print("-"*80)
    
    for word in comparison_words:
        stemmed = stemmer.stem(word)
        lemmatized = WordNetLemmatizer().lemmatize(word)
        print(f"{word:<20} {stemmed:<20} {lemmatized:<20}")
    
    # Example 3: Sentence lemmatization
    print("\n3. SENTENCE LEMMATIZATION:")
    print("="*80)
    
    sentences = [
        "The striped bats are hanging on their feet for best.",
        "Studies have shown that studying improves learning outcomes.",
        "He was running better than his competitors in the marathon."
    ]
    
    for sent in sentences:
        print(f"\nOriginal: {sent}")
        
        # Without POS
        lemmatized_simple, tokens = lemmatize_text(sent, use_pos=False)
        print(f"Tokens: {tokens}")
        print(f"Lemmatized (simple): {lemmatized_simple}")
        
        # With POS
        lemmatized_pos, _ = lemmatize_text(sent, use_pos=True)
        print(f"Lemmatized (with POS): {lemmatized_pos}")
    
    # Example 4: Benefits of lemmatization
    print("\n" + "="*80)
    print("KEY ADVANTAGES OF LEMMATIZATION:")
    print("="*80)
    print("1. Returns actual dictionary words (better -> good, not 'bett')")
    print("2. Context-aware with POS tags (caring: verb->care, noun->caring)")
    print("3. More accurate than stemming but computationally expensive")
    print("4. Preserves meaning better than stemming")
    print("="*80)


if __name__ == "__main__":
    demonstrate()
