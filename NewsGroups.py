import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import os

# Download required resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ==========================================
# LOAD 20 NEWSGROUPS DATA
# ==========================================
print("Loading 20 NewsGroups dataset...")

# List all available newsgroups
newsgroups_path = "/workspaces/NLP-Practicals/20 NewsGroups"
newsgroups_files = [f for f in os.listdir(newsgroups_path) if f.endswith('.txt')]
print(f"Found {len(newsgroups_files)} newsgroup categories\n")

# Load first newsgroup file
first_file = os.path.join(newsgroups_path, newsgroups_files[0])
with open(first_file, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Get first 500 characters for demonstration
sample_text = text[:500]
print(f"Using: {newsgroups_files[0]}")
print(f"Original Text:\n{sample_text}...\n")

# ==========================================
# 1. TEXT NORMALIZATION
# ==========================================
print("1. NORMALIZATION")
normalized_text = sample_text.lower()  # Lowercase
normalized_text = normalized_text.replace("n't", " not")  # Handle contractions
normalized_text = normalized_text.replace("'re", " are")
normalized_text = normalized_text.replace("'s", " is")
print(f"Result: {normalized_text[:200]}...\n")

# ==========================================
# 2. TOKENIZATION
# ==========================================
print("2. TOKENIZATION")
tokens = word_tokenize(normalized_text)
print(f"Tokens: {tokens[:20]}")
print(f"Total tokens: {len(tokens)}\n")

# ==========================================
# 3. STOP WORD REMOVAL
# ==========================================
print("3. STOP WORD REMOVAL")
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
print(f"After removing stop words: {filtered_tokens[:20]}")
print(f"Removed {len(tokens) - len(filtered_tokens)} tokens\n")

# ==========================================
# 4. STEMMING
# ==========================================
print("4. STEMMING")
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print(f"Stemmed: {stemmed_tokens[:20]}\n")

# ==========================================
# 5. LEMMATIZATION
# ==========================================
print("5. LEMMATIZATION")
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    """Convert POS tag to WordNet format"""
    from nltk.corpus import wordnet as wn
    if tag.startswith('J'): return wn.ADJ
    elif tag.startswith('V'): return wn.VERB
    elif tag.startswith('N'): return wn.NOUN
    elif tag.startswith('R'): return wn.ADV
    else: return wn.NOUN

# Get POS tags and lemmatize
pos_tags = pos_tag(filtered_tokens)
lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
                     for word, pos in pos_tags]
print(f"Lemmatized: {lemmatized_tokens[:20]}\n")

# ==========================================
# COMPARISON
# ==========================================
print("="*60)
print("COMPARISON")
print("="*60)
print(f"Original tokens:        {len(tokens)}")
print(f"After stop words:       {len(filtered_tokens)}")
print(f"Unique after stemming:  {len(set(stemmed_tokens))}")
print(f"Unique after lemmatize: {len(set(lemmatized_tokens))}")

# Show specific examples
print("\n" + "="*60)
print("EXAMPLE TRANSFORMATIONS")
print("="*60)
examples = ['running', 'better', 'studies', 'are', 'mice']
print(f"{'Word':<15} {'Stemmed':<15} {'Lemmatized':<15}")
print("-"*45)
for word in examples:
    stemmed = stemmer.stem(word)
    # Get POS for lemmatization (assume verb for demo)
    lemma = lemmatizer.lemmatize(word, wordnet.VERB)
    print(f"{word:<15} {stemmed:<15} {lemma:<15}")

# ==========================================
# PROCESS ALL NEWSGROUPS
# ==========================================
print("\n" + "="*60)
print("PROCESSING ALL NEWSGROUPS")
print("="*60)

results = {}
for filename in sorted(newsgroups_files):
    filepath = os.path.join(newsgroups_path, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()[:500]  # First 500 chars
        
        # Tokenize and filter
        normalized = content.lower()
        tokens = word_tokenize(normalized)
        filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
        
        results[filename] = {
            'total_tokens': len(tokens),
            'filtered_tokens': len(filtered),
            'unique_tokens': len(set(filtered))
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"\n{'Category':<35} {'Total':<10} {'Filtered':<10} {'Unique':<10}")
print("-"*55)
for category, stats in results.items():
    print(f"{category:<35} {stats['total_tokens']:<10} {stats['filtered_tokens']:<10} {stats['unique_tokens']:<10}")

print(f"\nTotal categories processed: {len(results)}")
