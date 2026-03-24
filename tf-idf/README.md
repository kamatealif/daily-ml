# TF-IDF and Text Vectorization Examples

This folder contains simple Python examples for common NLP text encoding and search ideas. Each script is written in a beginner-friendly style and can be run directly with Python.

## Files

- `bag_of_words.py`  
  Shows how to build a basic bag-of-words vocabulary and count vector for each document.

- `one_hot_encoding.py`  
  Demonstrates one-hot encoding and multi-hot sentence encoding.

- `one_hot_search.py`  
  Uses one-hot style vectors with cosine similarity for simple search and recommendation.

- `ngram_encoding.py`  
  Builds n-gram vocabularies and sentence vectors.

- `ngram_search.py`  
  Uses n-gram vectors to rank the most relevant sentences for a query.

- `tf_search.py`  
  Computes normalized term frequency for each line and searches by word frequency.

- `tfidf_vectorizer.py`  
  Demonstrates TF-IDF vectorization by building a vocabulary, calculating IDF scores, and generating TF-IDF vectors for documents.

## Run the examples

Use Python to run any file:

```bash
python bag_of_words.py
python one_hot_encoding.py
python one_hot_search.py
python ngram_encoding.py
python ngram_search.py
python tf_search.py
python tfidf_vectorizer.py
```

## What the TF-IDF example shows

The `tfidf_vectorizer.py` script:

- tokenizes each document
- builds a shared vocabulary
- computes document frequency and smoothed IDF scores
- creates a TF-IDF vector for each document
- prints the top weighted terms in each document

This makes it easier to see how TF-IDF gives higher importance to words that are frequent in one document but less common across the full collection.
