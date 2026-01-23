"""
Text Preprocessing Pipeline for Brown Corpus Dataset
Implements: Text Normalization, Tokenization, Stop Word Removal, Stemming, Lemmatization
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
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


class BrownCorpusPreprocessor:
    """Preprocessing pipeline for Brown Corpus"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.porter_stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
        
    def text_normalization(self, text):
        """
        Normalize text: lowercase, remove special characters, extra spaces
        """
        if pd.isna(text):
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
        if not text or pd.isna(text):
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
    
    def preprocess_complete(self, text, include_stemming=True, include_lemmatization=True):
        """
        Complete preprocessing pipeline
        """
        # Step 1: Text Normalization
        normalized_text = self.text_normalization(text)
        
        # Step 2: Tokenization
        tokens = self.tokenization(normalized_text)
        
        # Step 3: Remove Stop Words
        filtered_tokens = self.remove_stopwords(tokens)
        
        results = {
            'original': text,
            'normalized': normalized_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens
        }
        
        # Step 4: Stemming (optional)
        if include_stemming:
            porter_stemmed = self.stem_text(filtered_tokens, 'porter')
            snowball_stemmed = self.stem_text(filtered_tokens, 'snowball')
            results['porter_stemmed'] = porter_stemmed
            results['snowball_stemmed'] = snowball_stemmed
        
        # Step 5: Lemmatization (optional)
        if include_lemmatization:
            lemmatized = self.lemmatize_text(filtered_tokens)
            results['lemmatized'] = lemmatized
        
        return results


def process_brown_corpus(csv_path='Brown Corpus/brown.csv', output_path='Brown Corpus/brown_preprocessed.csv', sample_size=None):
    """
    Process Brown Corpus CSV file
    """
    print("Loading Brown Corpus dataset...")
    df = pd.read_csv(csv_path)
    
    if sample_size:
        df = df.head(sample_size)
    
    print(f"Processing {len(df)} records...")
    
    preprocessor = BrownCorpusPreprocessor()
    
    # Apply preprocessing to raw_text column
    results = df['raw_text'].apply(lambda x: preprocessor.preprocess_complete(x))
    
    # Extract results into separate columns
    df['normalized_text'] = results.apply(lambda x: x['normalized'])
    df['tokens'] = results.apply(lambda x: ' '.join(x['tokens']))
    df['filtered_tokens'] = results.apply(lambda x: ' '.join(x['filtered_tokens']))
    df['porter_stemmed'] = results.apply(lambda x: ' '.join(x['porter_stemmed']))
    df['snowball_stemmed'] = results.apply(lambda x: ' '.join(x['snowball_stemmed']))
    df['lemmatized'] = results.apply(lambda x: ' '.join(x['lemmatized']))
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    return df


def demonstrate_preprocessing():
    """
    Demonstrate preprocessing on sample text
    """
    preprocessor = BrownCorpusPreprocessor()
    
    # Sample text from Brown Corpus
    sample_text = """Furthermore, as an encouragement to revisionist thinking, 
    it manifestly is fair to admit that any fraternity has a constitutional 
    right to refuse to accept persons it dislikes."""
    
    print("="*80)
    print("BROWN CORPUS PREPROCESSING DEMONSTRATION")
    print("="*80)
    
    results = preprocessor.preprocess_complete(sample_text)
    
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
    
    print("\n7. LEMMATIZATION:")
    print(results['lemmatized'])
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Demonstrate preprocessing
    demonstrate_preprocessing()
    
    # Process Brown Corpus (sample of 100 records for demo)
    print("\nProcessing Brown Corpus dataset...")
    process_brown_corpus(sample_size=100)
    
    print("\nPreprocessing complete!")
