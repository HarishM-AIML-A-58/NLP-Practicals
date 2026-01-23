import nltk
from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import string
import re
from collections import Counter
import pandas as pd

# Download required NLTK data
print("Downloading required NLTK resources...")
nltk.download('brown')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
print("Download complete!\n")

# ==========================================
# 1. LOAD BROWN CORPUS
# ==========================================
print("="*60)
print("LOADING BROWN CORPUS")
print("="*60)

# Load sample data from Brown Corpus
# You can choose different categories: news, editorial, reviews, religion, etc.
categories = ['news', 'editorial', 'reviews']
brown_sentences = brown.sents(categories=categories)

# Take first 100 sentences for demonstration
sample_sentences = brown_sentences[:100]

# Convert list of word lists to actual sentences
raw_text_samples = [' '.join(sent) for sent in sample_sentences[:5]]

print(f"Total sentences loaded: {len(sample_sentences)}")
print(f"\nSample raw sentences:")
for i, sent in enumerate(raw_text_samples[:3], 1):
    print(f"{i}. {sent}")

# ==========================================
# 2. TEXT NORMALIZATION
# ==========================================
print("\n" + "="*60)
print("TEXT NORMALIZATION")
print("="*60)

def normalize_text(text):
    """
    Normalize text by:
    - Converting to lowercase
    - Removing extra whitespace
    - Handling contractions (basic)
    - Removing special characters (optional)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optional: Expand common contractions
    contractions = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    return text

# Apply normalization
normalized_samples = [normalize_text(sent) for sent in raw_text_samples]

print("Original vs Normalized:")
for i in range(3):
    print(f"\nOriginal: {raw_text_samples[i]}")
    print(f"Normalized: {normalized_samples[i]}")

# ==========================================
# 3. TOKENIZATION
# ==========================================
print("\n" + "="*60)
print("TOKENIZATION")
print("="*60)

def tokenize_text(text):
    """
    Tokenize text into words using NLTK's word_tokenize
    """
    tokens = word_tokenize(text)
    return tokens

# Apply tokenization
tokenized_samples = [tokenize_text(sent) for sent in normalized_samples]

print("\nTokenization Examples:")
for i in range(3):
    print(f"\nSentence {i+1}: {normalized_samples[i]}")
    print(f"Tokens: {tokenized_samples[i]}")
    print(f"Token count: {len(tokenized_samples[i])}")

# ==========================================
# 4. STOP WORD REMOVAL
# ==========================================
print("\n" + "="*60)
print("STOP WORD REMOVAL")
print("="*60)

# Get English stop words
stop_words = set(stopwords.words('english'))

print(f"Total stop words in NLTK: {len(stop_words)}")
print(f"Sample stop words: {list(stop_words)[:20]}")

def remove_stopwords(tokens, remove_punctuation=True):
    """
    Remove stop words and optionally punctuation from tokens
    """
    if remove_punctuation:
        # Remove both stop words and punctuation
        filtered_tokens = [
            token for token in tokens 
            if token not in stop_words and token not in string.punctuation
        ]
    else:
        # Remove only stop words
        filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return filtered_tokens

# Apply stop word removal
filtered_samples = [remove_stopwords(tokens) for tokens in tokenized_samples]

print("\n\nBefore and After Stop Word Removal:")
for i in range(3):
    print(f"\nSentence {i+1}:")
    print(f"Before: {tokenized_samples[i]}")
    print(f"After:  {filtered_samples[i]}")
    print(f"Removed {len(tokenized_samples[i]) - len(filtered_samples[i])} tokens")

# ==========================================
# 5. STEMMING
# ==========================================
print("\n" + "="*60)
print("STEMMING")
print("="*60)

# Initialize stemmers
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer('english')

def stem_tokens(tokens, stemmer_type='porter'):
    """
    Apply stemming to tokens
    stemmer_type: 'porter' or 'snowball'
    """
    if stemmer_type == 'porter':
        stemmer = porter_stemmer
    else:
        stemmer = snowball_stemmer
    
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed

# Apply stemming with both stemmers
porter_stemmed = [stem_tokens(tokens, 'porter') for tokens in filtered_samples]
snowball_stemmed = [stem_tokens(tokens, 'snowball') for tokens in filtered_samples]

print("\nPorter vs Snowball Stemmer Comparison:")
for i in range(2):
    print(f"\nSentence {i+1}:")
    print(f"Original:  {filtered_samples[i]}")
    print(f"Porter:    {porter_stemmed[i]}")
    print(f"Snowball:  {snowball_stemmed[i]}")

# Demonstrate stemming with common word variations
print("\n\nStemming Word Variations:")
word_variations = ['running', 'runs', 'ran', 'runner', 'happily', 'happiness', 
                   'happy', 'studies', 'studying', 'studied']
print(f"{'Word':<15} {'Porter Stem':<15} {'Snowball Stem':<15}")
print("-" * 45)
for word in word_variations:
    porter = porter_stemmer.stem(word)
    snowball = snowball_stemmer.stem(word)
    print(f"{word:<15} {porter:<15} {snowball:<15}")

# ==========================================
# 6. LEMMATIZATION
# ==========================================
print("\n" + "="*60)
print("LEMMATIZATION")
print("="*60)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """
    Convert treebank POS tag to WordNet POS tag
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

def lemmatize_tokens(tokens, use_pos=True):
    """
    Lemmatize tokens with optional POS tagging for better accuracy
    """
    if use_pos:
        # Get POS tags
        pos_tags = pos_tag(tokens)
        # Lemmatize with POS information
        lemmatized = [
            lemmatizer.lemmatize(token, get_wordnet_pos(pos)) 
            for token, pos in pos_tags
        ]
    else:
        # Lemmatize without POS (defaults to noun)
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized

# Apply lemmatization
lemmatized_without_pos = [lemmatize_tokens(tokens, use_pos=False) for tokens in filtered_samples]
lemmatized_with_pos = [lemmatize_tokens(tokens, use_pos=True) for tokens in filtered_samples]

print("\nLemmatization Comparison (with vs without POS):")
for i in range(2):
    print(f"\nSentence {i+1}:")
    print(f"Original:      {filtered_samples[i]}")
    print(f"Without POS:   {lemmatized_without_pos[i]}")
    print(f"With POS:      {lemmatized_with_pos[i]}")

# Demonstrate lemmatization with word variations
print("\n\nLemmatization Examples:")
test_words = [
    ('running', 'VBG'), ('runs', 'VBZ'), ('ran', 'VBD'),
    ('better', 'JJR'), ('best', 'JJS'),
    ('am', 'VBP'), ('is', 'VBZ'), ('are', 'VBP'),
    ('mice', 'NNS'), ('geese', 'NNS')
]

print(f"{'Word':<15} {'POS':<10} {'Lemma (no POS)':<20} {'Lemma (with POS)':<20}")
print("-" * 65)
for word, pos in test_words:
    lemma_no_pos = lemmatizer.lemmatize(word)
    lemma_with_pos = lemmatizer.lemmatize(word, get_wordnet_pos(pos))
    print(f"{word:<15} {pos:<10} {lemma_no_pos:<20} {lemma_with_pos:<20}")

# ==========================================
# 7. COMPLETE PIPELINE COMPARISON
# ==========================================
print("\n" + "="*60)
print("COMPLETE PREPROCESSING PIPELINE")
print("="*60)

def preprocess_pipeline(text, method='lemmatize'):
    """
    Complete preprocessing pipeline
    method: 'stem' or 'lemmatize'
    """
    # Step 1: Normalize
    text = normalize_text(text)
    
    # Step 2: Tokenize
    tokens = tokenize_text(text)
    
    # Step 3: Remove stop words and punctuation
    tokens = remove_stopwords(tokens, remove_punctuation=True)
    
    # Step 4: Stem or Lemmatize
    if method == 'stem':
        tokens = stem_tokens(tokens, 'porter')
    else:
        tokens = lemmatize_tokens(tokens, use_pos=True)
    
    return tokens

# Apply complete pipeline
print("\nProcessing sample sentences through complete pipeline:\n")
for i in range(3):
    print(f"Sentence {i+1}:")
    print(f"Original:     {raw_text_samples[i]}")
    print(f"Stemmed:      {preprocess_pipeline(raw_text_samples[i], 'stem')}")
    print(f"Lemmatized:   {preprocess_pipeline(raw_text_samples[i], 'lemmatize')}")
    print()

# ==========================================
# 8. STATISTICS AND ANALYSIS
# ==========================================
print("\n" + "="*60)
print("PREPROCESSING STATISTICS")
print("="*60)

# Process larger sample for statistics
sample_for_stats = sample_sentences[:50]
raw_texts = [' '.join(sent) for sent in sample_for_stats]

# Apply different preprocessing levels
original_tokens = [word_tokenize(normalize_text(text)) for text in raw_texts]
after_stopwords = [remove_stopwords(tokens) for tokens in original_tokens]
after_stemming = [stem_tokens(tokens, 'porter') for tokens in after_stopwords]
after_lemmatization = [lemmatize_tokens(tokens, use_pos=True) for tokens in after_stopwords]

# Calculate statistics
stats = {
    'Original tokens': sum(len(tokens) for tokens in original_tokens),
    'After stop word removal': sum(len(tokens) for tokens in after_stopwords),
    'After stemming': sum(len(tokens) for tokens in after_stemming),
    'After lemmatization': sum(len(tokens) for tokens in after_lemmatization),
}

print("\nToken Count Comparison (50 sentences):")
print("-" * 40)
for step, count in stats.items():
    print(f"{step:<30} {count:>6} tokens")

# Vocabulary size comparison
vocab_original = len(set([token for tokens in original_tokens for token in tokens]))
vocab_after_stop = len(set([token for tokens in after_stopwords for token in tokens]))
vocab_stemmed = len(set([token for tokens in after_stemming for token in tokens]))
vocab_lemmatized = len(set([token for tokens in after_lemmatization for token in tokens]))

print("\nVocabulary Size Comparison:")
print("-" * 40)
print(f"{'Original vocabulary:':<30} {vocab_original:>6} unique tokens")
print(f"{'After stop words:':<30} {vocab_after_stop:>6} unique tokens")
print(f"{'After stemming:':<30} {vocab_stemmed:>6} unique tokens")
print(f"{'After lemmatization:':<30} {vocab_lemmatized:>6} unique tokens")

# Most common words after preprocessing
print("\n\nTop 20 Most Common Words After Lemmatization:")
all_lemmatized = [token for tokens in after_lemmatization for token in tokens]
word_freq = Counter(all_lemmatized)
print("-" * 40)
for word, count in word_freq.most_common(20):
    print(f"{word:<20} {count:>4}")

# ==========================================
# 9. SAVE PROCESSED DATA
# ==========================================
print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)

# Create a DataFrame for easy analysis
df_data = []
for i, text in enumerate(raw_texts[:10]):
    df_data.append({
        'id': i+1,
        'original': text,
        'normalized': normalize_text(text),
        'tokens': original_tokens[i],
        'after_stopwords': after_stopwords[i],
        'stemmed': after_stemming[i],
        'lemmatized': after_lemmatization[i],
        'token_count_original': len(original_tokens[i]),
        'token_count_processed': len(after_lemmatization[i])
    })

df = pd.DataFrame(df_data)

# Display sample
print("\nSample Processed Data (first 3 rows):")
print(df[['id', 'original', 'lemmatized']].head(3).to_string())

# Save to CSV
output_file = 'brown_corpus_preprocessed.csv'
df.to_csv(output_file, index=False)
print(f"\nProcessed data saved to: {output_file}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)