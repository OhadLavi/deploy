import gensim
import numpy as np
from gensim.models import KeyedVectors
import os
import re

def is_hebrew_word(word):
    """Check if a word contains only Hebrew characters."""
    return bool(re.match(r'^[\u0590-\u05FF]+$', word))

def reduce_model(input_model_path, output_model_path, target_dims=50, min_count=10):
    """
    Reduce the size of a Word2Vec model by:
    1. Converting to KeyedVectors format
    2. Reducing vector dimensions
    3. Pruning non-Hebrew and infrequent words
    4. Using memory-efficient storage
    """
    print(f"Loading model from {input_model_path}...")
    model = gensim.models.Word2Vec.load(input_model_path)
    
    # Get vocabulary size and current dimensions
    vocab_size = len(model.wv.index_to_key)
    current_dims = model.wv.vector_size
    print(f"Original model: {vocab_size} words, {current_dims} dimensions")
    
    # Filter to keep only Hebrew words
    hebrew_words = [word for word in model.wv.index_to_key if is_hebrew_word(word)]
    print(f"Hebrew words: {len(hebrew_words)} out of {vocab_size}")
    
    # Filter by frequency
    word_freqs = {word: model.wv.get_vecattr(word, 'count') 
                 for word in hebrew_words 
                 if model.wv.get_vecattr(word, 'count') >= min_count}
    print(f"Words after frequency filtering: {len(word_freqs)}")
    
    # Get vectors for Hebrew words only
    hebrew_vectors = np.array([model.wv[word] for word in word_freqs.keys()])
    
    # Apply PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=target_dims)
    reduced_vectors = pca.fit_transform(hebrew_vectors)
    
    # Create new KeyedVectors with reduced dimensions
    new_vectors = KeyedVectors(target_dims)
    new_vectors.add_vectors(list(word_freqs.keys()), reduced_vectors)
    
    # Save in word2vec binary format for better compression
    print(f"Saving reduced model to {output_model_path}...")
    new_vectors.save_word2vec_format(output_model_path + '.bin', binary=True)
    
    # Print size comparison
    new_vocab_size = len(new_vectors.index_to_key)
    print(f"\nModel size comparison:")
    print(f"Original: {vocab_size} words, {current_dims} dimensions")
    print(f"Reduced:  {new_vocab_size} words, {target_dims} dimensions")
    
    # Print file size comparison
    orig_size = sum(os.path.getsize(f) for f in [input_model_path, 
                                                input_model_path + '.wv.vectors.npy',
                                                input_model_path + '.syn1neg.npy'])
    new_size = os.path.getsize(output_model_path + '.bin')
    print(f"\nFile size comparison:")
    print(f"Original: {orig_size / 1024 / 1024:.1f} MB")
    print(f"Reduced:  {new_size / 1024 / 1024:.1f} MB")
    
    # Test some similarities to verify quality
    test_words = ['בית', 'ספר', 'שולחן', 'כיסא', 'מחשב']
    print("\nTesting similarities:")
    for w1 in test_words[:2]:
        for w2 in test_words[2:4]:
            if w1 in new_vectors and w2 in new_vectors:
                old_sim = model.wv.similarity(w1, w2)
                new_sim = new_vectors.similarity(w1, w2)
                print(f"{w1} - {w2}:")
                print(f"  Original similarity: {old_sim:.3f}")
                print(f"  Reduced similarity:  {new_sim:.3f}")
    
    return new_vectors

if __name__ == "__main__":
    # Paths
    input_model = "model/model.mdl"
    output_model = "model/model_small"  # .bin will be added automatically
    
    # Reduce model size
    reduced_model = reduce_model(
        input_model_path=input_model,
        output_model_path=output_model,
        target_dims=50,   # Reduce to 50 dimensions
        min_count=10      # Remove words that appear less than 10 times
    ) 