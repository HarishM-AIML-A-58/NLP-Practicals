"""
Text Preprocessing Pipeline for Universal Dependencies - English TreeBank
Implements: Text Normalization, Tokenization, Stop Word Removal, Stemming, Lemmatization
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class UDTreeBankPreprocessor:
    """Preprocessing pipeline for Universal Dependencies English TreeBank"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.porter_stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
        
    def parse_conllu_file(self, file_path):
        """
        Parse CoNLL-U format file and extract sentences
        """
        sentences = []
        current_sentence = {
            'text': '',
            'tokens': [],
            'lemmas': [],
            'pos_tags': [],
            'metadata': {}
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines (sentence boundary)
                if not line:
                    if current_sentence['text']:
                        sentences.append(current_sentence)
                        current_sentence = {
                            'text': '',
                            'tokens': [],
                            'lemmas': [],
                            'pos_tags': [],
                            'metadata': {}
                        }
                    continue
                
                # Parse metadata comments
                if line.startswith('#'):
                    if line.startswith('# text ='):
                        current_sentence['text'] = line.replace('# text =', '').strip()
                    elif line.startswith('# sent_id ='):
                        current_sentence['metadata']['sent_id'] = line.replace('# sent_id =', '').strip()
                    continue
                
                # Parse token lines
                if not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 10 and '-' not in parts[0] and '.' not in parts[0]:
                        # Extract token, lemma, and POS tag
                        current_sentence['tokens'].append(parts[1])
                        current_sentence['lemmas'].append(parts[2])
                        current_sentence['pos_tags'].append(parts[3])
        
        # Add last sentence if exists
        if current_sentence['text']:
            sentences.append(current_sentence)
        
        return sentences
    
    def text_normalization(self, text):
        """
        Normalize text: lowercase, remove special characters, extra spaces
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
        text = re.sub(r'[^a-zA-Z\s.,!?;:\'\"-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenization(self, text):
        """
        Tokenize text into words
        """
        if not text:
            return []
        
        # Word tokenization
        tokens = word_tokenize(text)
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stop words from tokenized text
        """
        if not tokens:
            return []
        
        # Remove stopwords and single character tokens
        filtered_tokens = [word for word in tokens 
                          if word.lower() not in self.stop_words 
                          and len(word) > 1
                          and word not in string.punctuation]
        
        return filtered_tokens
    
    def stem_text(self, tokens, stemmer_type='porter'):
        """
        Apply stemming to tokens
        stemmer_type: 'porter' or 'snowball'
        """
        if not tokens:
            return []
        
        stemmer = self.porter_stemmer if stemmer_type == 'porter' else self.snowball_stemmer
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        
        return stemmed_tokens
    
    def lemmatize_text(self, tokens):
        """
        Apply lemmatization to tokens
        """
        if not tokens:
            return []
        
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return lemmatized_tokens
    
    def preprocess_sentence(self, sentence_dict):
        """
        Complete preprocessing pipeline for a sentence
        """
        text = sentence_dict['text']
        
        # Step 1: Text Normalization
        normalized_text = self.text_normalization(text)
        
        # Step 2: Tokenization
        tokens = self.tokenization(normalized_text)
        
        # Step 3: Remove Stop Words
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Step 4: Stemming
        porter_stemmed = self.stem_text(filtered_tokens, 'porter')
        snowball_stemmed = self.stem_text(filtered_tokens, 'snowball')
        
        # Step 5: Lemmatization
        lemmatized = self.lemmatize_text(filtered_tokens)
        
        return {
            'original': text,
            'normalized': normalized_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'porter_stemmed': porter_stemmed,
            'snowball_stemmed': snowball_stemmed,
            'lemmatized': lemmatized,
            'ud_lemmas': sentence_dict.get('lemmas', []),  # Original UD lemmas
            'ud_tokens': sentence_dict.get('tokens', []),  # Original UD tokens
            'pos_tags': sentence_dict.get('pos_tags', [])
        }


def process_ud_treebank(file_path='UD_English-EWT-master/en_ewt-ud-train.conllu', 
                        output_path='UD_English-EWT-master/ud_preprocessed.txt',
                        sample_size=None):
    """
    Process Universal Dependencies TreeBank file
    """
    print("Loading UD English TreeBank dataset...")
    preprocessor = UDTreeBankPreprocessor()
    
    # Parse CoNLL-U file
    sentences = preprocessor.parse_conllu_file(file_path)
    
    if sample_size:
        sentences = sentences[:sample_size]
    
    print(f"Processing {len(sentences)} sentences...")
    
    # Process each sentence
    processed_sentences = []
    for sent in sentences:
        result = preprocessor.preprocess_sentence(sent)
        processed_sentences.append(result)
    
    # Save results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, result in enumerate(processed_sentences, 1):
            f.write(f"{'='*80}\n")
            f.write(f"SENTENCE {i}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Original: {result['original']}\n")
            f.write(f"Normalized: {result['normalized']}\n")
            f.write(f"Tokens: {' '.join(result['tokens'])}\n")
            f.write(f"Filtered: {' '.join(result['filtered_tokens'])}\n")
            f.write(f"Porter Stemmed: {' '.join(result['porter_stemmed'])}\n")
            f.write(f"Snowball Stemmed: {' '.join(result['snowball_stemmed'])}\n")
            f.write(f"Lemmatized: {' '.join(result['lemmatized'])}\n")
            f.write(f"UD Original Tokens: {' '.join(result['ud_tokens'])}\n")
            f.write(f"UD Original Lemmas: {' '.join(result['ud_lemmas'])}\n")
            f.write("\n")
    
    print(f"Preprocessed data saved to {output_path}")
    
    return processed_sentences


def demonstrate_preprocessing():
    """
    Demonstrate preprocessing on sample sentence
    """
    preprocessor = UDTreeBankPreprocessor()
    
    # Sample sentence
    sample_sentence = {
        'text': "Al-Zaman: American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim, near the Syrian border.",
        'tokens': ['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah', 'al', '-', 'Ani'],
        'lemmas': ['Al', '-', 'Zaman', ':', 'American', 'force', 'kill', 'Shaikh', 'Abdullah', 'al', '-', 'Ani'],
        'pos_tags': ['PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'ADJ', 'NOUN', 'VERB', 'PROPN', 'PROPN', 'PROPN', 'PUNCT', 'PROPN']
    }
    
    print("="*80)
    print("UD ENGLISH TREEBANK PREPROCESSING DEMONSTRATION")
    print("="*80)
    
    results = preprocessor.preprocess_sentence(sample_sentence)
    
    print("\n1. ORIGINAL TEXT:")
    print(results['original'])
    
    print("\n2. NORMALIZED TEXT:")
    print(results['normalized'])
    
    print("\n3. TOKENIZATION:")
    print(results['tokens'])
    
    print("\n4. STOP WORD REMOVAL:")
    print(results['filtered_tokens'])
    
    print("\n5. PORTER STEMMING:")
    print(results['porter_stemmed'])
    
    print("\n6. SNOWBALL STEMMING:")
    print(results['snowball_stemmed'])
    
    print("\n7. LEMMATIZATION (NLTK):")
    print(results['lemmatized'])
    
    print("\n8. UD ORIGINAL LEMMAS:")
    print(results['ud_lemmas'])
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Demonstrate preprocessing
    demonstrate_preprocessing()
    
    # Process UD TreeBank (sample of 50 sentences for demo)
    print("\nProcessing UD English TreeBank dataset...")
    process_ud_treebank(sample_size=50)
    
    print("\nPreprocessing complete!")
