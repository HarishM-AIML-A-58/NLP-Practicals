import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import pandas as pd
import os

# Download required resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==========================================
# LOAD REUTERS DATASET
# ==========================================
print("Loading Reuters dataset...")

# Load ModApte train data
reuters_path = "/workspaces/NLP-Practicals/Reuters"
train_file = os.path.join(reuters_path, "ModApte_train.csv")

# Read CSV
df = pd.read_csv(train_file)
print(f"Total documents: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# Get first document
first_doc = df.iloc[0]
text = str(first_doc['text'])
print(f"Document ID: {first_doc.get('new_id', 'N/A')}")
print(f"Topics: {first_doc.get('topics', 'N/A')}")
print(f"Original Text:\n{text[:500]}...\n")

# ==========================================
# 1. TEXT NORMALIZATION
# ==========================================
print("1. NORMALIZATION")
normalized_text = text.lower()  # Lowercase
normalized_text = normalized_text.replace("n't", " not")  # Handle contractions
normalized_text = normalized_text.replace("'re", " are")
normalized_text = normalized_text.replace("'s", " is")
normalized_text = normalized_text.replace("'ve", " have")
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
    lemma = lemmatizer.lemmatize(word, wordnet.VERB)
    print(f"{word:<15} {stemmed:<15} {lemma:<15}")

# ==========================================
# PROCESS MULTIPLE DOCUMENTS
# ==========================================
print("\n" + "="*60)
print("PROCESSING MULTIPLE DOCUMENTS")
print("="*60)

# Process first 10 documents
results = []
for idx in range(min(10, len(df))):
    doc = df.iloc[idx]
    text = str(doc['text'])
    
    # Tokenize and filter
    normalized = text.lower()
    tokens = word_tokenize(normalized)
    filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
    
    results.append({
        'doc_id': doc.get('new_id', idx),
        'topics': doc.get('topics', 'N/A'),
        'total_tokens': len(tokens),
        'filtered_tokens': len(filtered),
        'unique_tokens': len(set(filtered))
    })

print(f"\n{'Doc ID':<10} {'Topics':<20} {'Total':<10} {'Filtered':<10} {'Unique':<10}")
print("-"*70)
for r in results:
    topics = str(r['topics'])[:18]
    print(f"{r['doc_id']:<10} {topics:<20} {r['total_tokens']:<10} {r['filtered_tokens']:<10} {r['unique_tokens']:<10}")

# ==========================================
# DATASET STATISTICS
# ==========================================
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

# Available files
files = [f for f in os.listdir(reuters_path) if f.endswith('.csv')]
print(f"\nAvailable files:")
for f in sorted(files):
    file_path = os.path.join(reuters_path, f)
    df_temp = pd.read_csv(file_path)
    print(f"  {f:<30} {len(df_temp):>8} documents")

print(f"\nTotal files: {len(files)}")
