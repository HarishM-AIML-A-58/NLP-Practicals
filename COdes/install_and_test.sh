#!/bin/bash

# Installation and Testing Script for NLP Practicals
# Text Preprocessing Implementation

echo "================================================================================"
echo "NLP PRACTICALS - TEXT PREPROCESSING INSTALLATION & TESTING"
echo "Exercise 1: Category 1 - Text Cleaning and Preprocessing"
echo "================================================================================"

# Step 1: Check Python installation
echo ""
echo "Step 1: Checking Python installation..."
if command -v python3 &> /dev/null
then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python is installed: $PYTHON_VERSION"
else
    echo "✗ Python is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
echo "Installing nltk and pandas..."
python3 -m pip install --user nltk pandas

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Step 3: Download NLTK data
echo ""
echo "Step 3: Downloading NLTK data..."
python3 << EOF
import nltk
print("Downloading punkt tokenizer...")
nltk.download('punkt', quiet=True)
print("Downloading stopwords...")
nltk.download('stopwords', quiet=True)
print("Downloading wordnet...")
nltk.download('wordnet', quiet=True)
print("Downloading omw-1.4...")
nltk.download('omw-1.4', quiet=True)
print("Downloading averaged_perceptron_tagger...")
nltk.download('averaged_perceptron_tagger', quiet=True)
print("✓ All NLTK data downloaded")
EOF

# Step 4: Test individual techniques
echo ""
echo "================================================================================"
echo "Step 4: Testing Individual Preprocessing Techniques"
echo "================================================================================"

echo ""
echo "Testing Text Normalization..."
python3 "Text Normalization.py"
echo ""

echo "Testing Tokenization..."
python3 Tokenization.py
echo ""

echo "Testing Stop Word Removal..."
python3 "Stop Word Removal.py"
echo ""

echo "Testing Stemming..."
python3 Stemming.py
echo ""

echo "Testing Lemmatization..."
python3 Lemmatization.py
echo ""

# Step 5: Run unified demonstration
echo ""
echo "================================================================================"
echo "Step 5: Running Unified Preprocessing Demonstration"
echo "================================================================================"
python3 preprocessing_all_datasets.py --demo

# Summary
echo ""
echo "================================================================================"
echo "INSTALLATION AND TESTING COMPLETE!"
echo "================================================================================"
echo ""
echo "✓ All preprocessing techniques tested successfully"
echo ""
echo "Next Steps:"
echo "  1. View documentation: PREPROCESSING_README.md"
echo "  2. View summary: IMPLEMENTATION_SUMMARY.md"
echo "  3. Process datasets:"
echo "     python3 preprocessing_all_datasets.py --dataset all"
echo "     python3 preprocessing_all_datasets.py --dataset brown --sample 100"
echo "     python3 preprocessing_all_datasets.py --dataset reuters --sample 100"
echo ""
echo "================================================================================"
