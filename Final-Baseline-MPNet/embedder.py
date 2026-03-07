from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ConnectionsEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initializes the sentence transformer model.
        The default is 'all-mpnet-base-v2' which provides high accuracy for sentence embeddings.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

    def get_embeddings(self, words: list[str]) -> np.ndarray:
        """
        Returns normalized embeddings for a list of words.
        Shape: (len(words), embedding_dim)
        Normalization ensures that cosine similarity can be computed via dot product.
        """
        # encode handles batching and tokenization
        embeddings = self.model.encode(words, show_progress_bar=False, normalize_embeddings=True)
        return embeddings

    def get_lexical_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a lexical similarity matrix using character n-grams.
        This captures syntactic relationships like "words that start with C", 
        palindromes, or prefixes/suffixes.
        """
        # Character n-grams (prefix, suffix, and internal)
        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        X = vectorizer.fit_transform(words)
        return cosine_similarity(X)

if __name__ == "__main__":
    # Test the embedder
    embedder = ConnectionsEmbedder()
    words = ["APPLE", "BANANA", "CHAIR", "TABLE"]
    embeddings = embedder.get_embeddings(words)
    print(f"Embedded {len(words)} words. Shape: {embeddings.shape}")
    
    # Cosine similarity matrix (since they are normalized, dot product = cosine similarity)
    similarity = np.dot(embeddings, embeddings.T)
    print("Similarity Matrix:")
    print(similarity)
