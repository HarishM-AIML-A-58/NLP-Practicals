"""
Usage Examples and Testing Script
Run this script to test all preprocessing implementations
"""

def test_installations():
    """Test if all required packages are installed"""
    print("="*80)
    print("TESTING INSTALLATIONS")
    print("="*80)
    
    try:
        import nltk
        print("✓ NLTK installed")
    except ImportError:
        print("✗ NLTK not installed. Run: pip install nltk")
        return False
    
    try:
        import pandas
        print("✓ Pandas installed")
    except ImportError:
        print("✗ Pandas not installed. Run: pip install pandas")
        return False
    
    # Check NLTK data
    print("\nChecking NLTK data...")
    try:
        nltk.data.find('tokenizers/punkt')
        print("✓ punkt tokenizer found")
    except LookupError:
        print("! punkt not found. Downloading...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
        print("✓ stopwords found")
    except LookupError:
        print("! stopwords not found. Downloading...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
        print("✓ wordnet found")
    except LookupError:
        print("! wordnet not found. Downloading...")
        nltk.download('wordnet')
    
    return True


def run_all_tests():
    """Run all preprocessing demonstrations"""
    print("\n" + "="*80)
    print("RUNNING ALL PREPROCESSING TESTS")
    print("="*80)
    
    # Test 1: Text Normalization
    print("\n1. TESTING TEXT NORMALIZATION")
    print("-"*80)
    try:
        exec(open('Text Normalization.py').read())
        print("✓ Text Normalization works")
    except Exception as e:
        print(f"✗ Text Normalization error: {e}")
    
    # Test 2: Tokenization
    print("\n2. TESTING TOKENIZATION")
    print("-"*80)
    try:
        exec(open('Tokenization.py').read())
        print("✓ Tokenization works")
    except Exception as e:
        print(f"✗ Tokenization error: {e}")
    
    # Test 3: Stop Word Removal
    print("\n3. TESTING STOP WORD REMOVAL")
    print("-"*80)
    try:
        exec(open('Stop Word Removal.py').read())
        print("✓ Stop Word Removal works")
    except Exception as e:
        print(f"✗ Stop Word Removal error: {e}")
    
    # Test 4: Stemming
    print("\n4. TESTING STEMMING")
    print("-"*80)
    try:
        exec(open('Stemming.py').read())
        print("✓ Stemming works")
    except Exception as e:
        print(f"✗ Stemming error: {e}")
    
    # Test 5: Lemmatization
    print("\n5. TESTING LEMMATIZATION")
    print("-"*80)
    try:
        exec(open('Lemmatization.py').read())
        print("✓ Lemmatization works")
    except Exception as e:
        print(f"✗ Lemmatization error: {e}")


def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    
    print("""
STEP 1: Install Dependencies
    pip install nltk pandas

STEP 2: Test Individual Techniques
    python "Text Normalization.py"
    python Tokenization.py
    python "Stop Word Removal.py"
    python Stemming.py
    python Lemmatization.py

STEP 3: Run Demonstrations
    python preprocessing_all_datasets.py --demo

STEP 4: Process Datasets
    # All datasets
    python preprocessing_all_datasets.py --dataset all
    
    # Individual datasets
    python preprocessing_all_datasets.py --dataset brown --sample 100
    python preprocessing_all_datasets.py --dataset ud --sample 50
    python preprocessing_all_datasets.py --dataset newsgroups
    python preprocessing_all_datasets.py --dataset reuters --sample 100

STEP 5: Use in Your Code
    from preprocessing_brown_corpus import BrownCorpusPreprocessor
    
    preprocessor = BrownCorpusPreprocessor()
    text = "Your text here..."
    results = preprocessor.preprocess_complete(text)
    
    print(results['normalized'])
    print(results['filtered_tokens'])
    print(results['lemmatized'])

For detailed documentation, see PREPROCESSING_README.md
    """)
    
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("NLP PRACTICALS - TEXT PREPROCESSING")
    print("Exercise 1: Category 1 - Text Cleaning and Preprocessing")
    print("="*80)
    
    # Test installations
    if test_installations():
        print("\n✓ All dependencies installed successfully!")
        
        # Show usage
        show_usage_instructions()
        
        print("\nRun this script with --test flag to test all implementations:")
        print("    python usage_examples.py --test")
    else:
        print("\n✗ Please install missing dependencies first")
        print("Run: pip install nltk pandas")
