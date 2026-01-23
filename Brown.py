import nltk
from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Download required resources
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ==========================================
# LOAD DATA
# ==========================================
print("Loading Brown Corpus...")
sentences = brown.sents(categories=['news'])[:10]  # First 10 sentences
text = ' '.join([' '.join(sent) for sent in sentences])
print(f"Original Text:\n{text[:200]}...\n")

# ==========================================
# 1. TEXT NORMALIZATION
# ==========================================
print("1. NORMALIZATION")
normalized_text = text.lower()  # Lowercase
normalized_text = normalized_text.replace("n't", " not")  # Handle contractions
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
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

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
