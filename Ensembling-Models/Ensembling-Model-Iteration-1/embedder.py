from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet as wn
import requests
import requests_cache
import concurrent.futures

requests_cache.install_cache('wikidata_cache', expire_after=86400)

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

    def get_wordnet_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a Knowledge Graph similarity matrix using WordNet.
        Uses Wu-Palmer similarity to find the maximum similarity between any synsets
        of the two words. Captures explicit synonyms and hypernyms.
        """
        n = len(words)
        matrix = np.zeros((n, n))
        
        # Precompute synsets for each word
        synsets_list = []
        for w in words:
            # Connections words are often uppercase, lower them for WordNet
            syns = wn.synsets(w.lower())
            # If empty, try lemmatizing or just keep empty
            synsets_list.append(syns)
            
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                    
                syns_i = synsets_list[i]
                syns_j = synsets_list[j]
                
                max_sim = 0.0
                if syns_i and syns_j:
                    for syn_i in syns_i:
                        for syn_j in syns_j:
                            # Wu-Palmer similarity
                            sim = syn_i.wup_similarity(syn_j)
                            if sim is not None and sim > max_sim:
                                max_sim = sim
                                
                matrix[i][j] = max_sim
                matrix[j][i] = max_sim
                
        return matrix

    def _get_wikidata_properties(self, word: str) -> set:
        url = "https://www.wikidata.org/w/api.php"
        headers = {"User-Agent": "ConnectionsSolver/1.0"}
        
        # 1. Search
        search_params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": word,
            "limit": 1
        }
        
        try:
            search_res = requests.get(url, params=search_params, headers=headers).json()
            if not search_res.get('search'):
                return set()
            entity_id = search_res['search'][0]['id']
            
            # 2. Get properties
            prop_params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": entity_id,
                "props": "claims",
                "languages": "en"
            }
            prop_res = requests.get(url, params=prop_params, headers=headers).json()
            claims = prop_res.get('entities', {}).get(entity_id, {}).get('claims', {})
            
            properties = set()
            for prop_id, claim_list in claims.items():
                for claim in claim_list:
                    try:
                        datavalue = claim['mainsnak']['datavalue']['value']
                        if isinstance(datavalue, dict) and datavalue.get('entity-type') == 'item':
                            properties.add(datavalue['id'])
                    except KeyError:
                        pass
            return properties
        except Exception:
            return set()

    def get_wikidata_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a Knowledge Graph similarity matrix using WikiData.
        Queries the WikiData API for properties and computes Jaccard similarity.
        """
        n = len(words)
        matrix = np.zeros((n, n))
        
        # Precompute property sets concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            props_list = list(executor.map(self._get_wikidata_properties, words))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                    
                set_i = props_list[i]
                set_j = props_list[j]
                
                intersection = set_i.intersection(set_j)
                union = set_i.union(set_j)
                
                sim = 0.0
                if union:
                    sim = len(intersection) / len(union)
                    
                # Scale up a bit as Jaccard values for large sets are often small but significant
                matrix[i][j] = sim * 5.0  # arbitrary scale, will be center-normalized anyway
                matrix[j][i] = sim * 5.0
                
        return matrix

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
