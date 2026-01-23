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
nltk.download('omw-1.4', quiet=True)

# ==========================================
# LOAD UD ENGLISH-EWT DATASET
# ==========================================
print("Loading UD English-EWT Treebank...")

treebank_path = "/workspaces/NLP-Practicals/UD_English-EWT-master"
train_file = os.path.join(treebank_path, "en_ewt-ud-train.conllu")

def parse_conllu(file_path, max_sentences=10):
    """Parse CoNLL-U format file and extract sentences"""
    sentences = []
    current_sentence = []
    sentence_text = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Extract original sentence text from comment
            if line.startswith('# text = '):
                sentence_text = line[9:]
            
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                if current_sentence and sentence_text:
                    sentences.append({
                        'text': sentence_text,
                        'tokens': current_sentence
                    })
                    current_sentence = []
                    sentence_text = None
                    
                    if len(sentences) >= max_sentences:
                        break
                continue
            
            # Parse token line: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            parts = line.split('\t')
            if len(parts) >= 10 and '-' not in parts[0] and '.' not in parts[0]:
                current_sentence.append({
                    'id': parts[0],
                    'form': parts[1],
                    'lemma': parts[2],
                    'upos': parts[3],
                    'xpos': parts[4]
                })
    
    return sentences

# Load sentences
sentences = parse_conllu(train_file, max_sentences=10)
print(f"Loaded {len(sentences)} sentences\n")

# Get first sentence
first_sent = sentences[0]
text = first_sent['text']
print(f"Original Text:\n{text}\n")
print("Annotated Tokens:")
for token in first_sent['tokens'][:10]:
    print(f"  {token['form']:<15} Lemma: {token['lemma']:<15} POS: {token['upos']}")
print()

# ==========================================
# 1. TEXT NORMALIZATION
# ==========================================
print("1. NORMALIZATION")
normalized_text = text.lower()  # Lowercase
normalized_text = normalized_text.replace("n't", " not")  # Handle contractions
normalized_text = normalized_text.replace("'re", " are")
normalized_text = normalized_text.replace("'s", " is")
print(f"Result: {normalized_text}\n")

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

# Compare with gold-standard lemmas from treebank
print("\n" + "="*60)
print("COMPARISON WITH GOLD-STANDARD LEMMAS")
print("="*60)
print(f"{'Word':<15} {'Our Lemma':<15} {'Gold Lemma':<15} {'Match'}")
print("-"*60)
for token in first_sent['tokens'][:10]:
    word = token['form'].lower()
    gold_lemma = token['lemma'].lower()
    
    # Get our lemmatization
    pos = token['xpos']
    wordnet_pos = get_wordnet_pos(pos)
    our_lemma = lemmatizer.lemmatize(word, wordnet_pos)
    
    match = "✓" if our_lemma == gold_lemma else "✗"
    print(f"{word:<15} {our_lemma:<15} {gold_lemma:<15} {match}")

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
# PROCESS MULTIPLE SENTENCES
# ==========================================
print("\n" + "="*60)
print("PROCESSING MULTIPLE SENTENCES")
print("="*60)

results = []
for idx, sent in enumerate(sentences):
    text = sent['text']
    
    # Tokenize and filter
    normalized = text.lower()
    tokens = word_tokenize(normalized)
    filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
    
    results.append({
        'sent_id': idx + 1,
        'total_tokens': len(tokens),
        'filtered_tokens': len(filtered),
        'unique_tokens': len(set(filtered))
    })

print(f"\n{'Sent ID':<10} {'Total':<10} {'Filtered':<10} {'Unique':<10}")
print("-"*40)
for r in results:
    print(f"{r['sent_id']:<10} {r['total_tokens']:<10} {r['filtered_tokens']:<10} {r['unique_tokens']:<10}")

# ==========================================
# DATASET STATISTICS
# ==========================================
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

# Available files
files = ['en_ewt-ud-train.conllu', 'en_ewt-ud-dev.conllu', 'en_ewt-ud-test.conllu']
print(f"\nAvailable files:")
for f in files:
    file_path = os.path.join(treebank_path, f)
    if os.path.exists(file_path):
        # Count lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = sum(1 for line in file)
        print(f"  {f:<30} {lines:>8} lines")

print(f"\nNote: CoNLL-U format includes tokens, lemmas, POS tags, and dependency relations")
