import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet as wn
try:
    from nltk.corpus import cmudict
    pronouncing_dict = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    from nltk.corpus import cmudict
    pronouncing_dict = cmudict.dict()

import requests
import requests_cache
import concurrent.futures
import gensim.downloader as gensim_api

requests_cache.install_cache('wikidata_cache', expire_after=86400)

class ConnectionsEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2",
                 numberbatch_path: str = "numberbatch-en-19.08.txt.gz"):
        """
        Initialises the sentence transformer and auxiliary word vector models.
        Loads GloVe-100 (via gensim cache) and ConceptNet Numberbatch-300 from
        the local file `numberbatch_path` (word2vec text format, plain English words).
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")
        print("Loading GloVe word vectors...")
        self.glove = gensim_api.load('glove-wiki-gigaword-100')
        print("GloVe loaded.")
        print(f"Loading ConceptNet Numberbatch from {numberbatch_path}...")
        from gensim.models import KeyedVectors
        self.numberbatch = KeyedVectors.load_word2vec_format(
            numberbatch_path, binary=False, encoding='utf-8'
        )
        print(f"Numberbatch loaded ({len(self.numberbatch)} words, dim=300).")

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
    def _get_datamuse_associations(self, word: str) -> set:
        url = "https://api.datamuse.com/words"
        # rel_trg queries words that are statistically associated with the trigger word in the same sentence/context
        params = {"rel_trg": word, "max": 100}
        try:
            res = requests.get(url, params=params).json()
            return set(item['word'].upper() for item in res)
        except Exception:
            return set()

    def get_datamuse_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a similarity matrix using Datamuse statistical associations.
        """
        n = len(words)
        matrix = np.zeros((n, n))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            assoc_list = list(executor.map(self._get_datamuse_associations, words))
            
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                    
                set_i = assoc_list[i]
                set_j = assoc_list[j]
                
                # Check direct association or overlapping associations
                direct_match = 1.0 if (words[j] in set_i or words[i] in set_j) else 0.0
                
                intersection = set_i.intersection(set_j)
                union = set_i.union(set_j)
                
                jaccard = 0.0
                if union:
                    jaccard = len(intersection) / len(union)
                    
                # Weight direct match heavily, and add jaccard for indirect association
                sim = max(direct_match * 0.8, jaccard * 5.0) 
                
                matrix[i][j] = sim
                matrix[j][i] = sim
                
        return matrix
        
    def _get_conceptnet_associations(self, word: str) -> set:
        import sqlite3
        import os
        db_path = "conceptnet_en.sqlite"
        if not os.path.exists(db_path):
            return set()
            
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            # Fetch all words that have an edge with this word
            c.execute("SELECT word2 FROM edges WHERE word1 = ?", (word,))
            connected1 = {row[0] for row in c.fetchall()}
            c.execute("SELECT word1 FROM edges WHERE word2 = ?", (word,))
            connected2 = {row[0] for row in c.fetchall()}
            conn.close()
            return connected1.union(connected2)
        except Exception:
            return set()

    def get_conceptnet_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a Knowledge Graph similarity matrix using local SQLite ConceptNet cache.
        Captures common-sense relationships (IsA, PartOf, UsedFor).
        """
        n = len(words)
        matrix = np.zeros((n, n))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            assoc_list = list(executor.map(self._get_conceptnet_associations, words))
            
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                    
                set_i = assoc_list[i]
                set_j = assoc_list[j]
                
                # Check direct association or overlapping associations
                direct_match = 1.0 if (words[j] in set_i or words[i] in set_j) else 0.0
                
                intersection = set_i.intersection(set_j)
                union = set_i.union(set_j)
                
                jaccard = 0.0
                if union:
                    jaccard = len(intersection) / len(union)
                    
                sim = max(direct_match * 0.8, jaccard * 5.0) 
                
                matrix[i][j] = sim
                matrix[j][i] = sim
                
        return matrix
        
    def get_numberbatch_embedding(self, words: list[str]) -> np.ndarray:
        """
        Returns ConceptNet Numberbatch-300 embeddings for each word.
        The English-only file uses plain lowercase tokens (no URI prefix).

        Lookup order:
          1. word_lower           (direct match)
          2. mean of sub-tokens   (for multi-word phrases, split on whitespace)
          3. zero vector          (if no match found)

        Returns L2-normalised embeddings. Shape: (N, 300)
        """
        dim = 300
        vecs = []
        for w in words:
            wl = w.lower()
            if wl in self.numberbatch:
                v = self.numberbatch[wl].astype(np.float32)
            else:
                # Multi-word phrase: average sub-token vectors
                tokens = wl.split()
                token_vecs = [
                    self.numberbatch[t].astype(np.float32)
                    for t in tokens if t in self.numberbatch
                ]
                v = np.mean(token_vecs, axis=0) if token_vecs else np.zeros(dim, dtype=np.float32)
            vecs.append(v)

        mat = np.array(vecs, dtype=np.float32)   # (N, 300)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def get_concatenated_mpnet_numberbatch_embedding(self, words: list[str]) -> np.ndarray:
        """
        Concatenates MPNet-768 and Numberbatch-300 embeddings, then re-normalizes
        to unit norm so that the dot product of two concatenated vectors equals
        their cosine similarity in the joint 1068-dim space.

        This captures both:
          - Distributional semantics (MPNet, trained on 1B sentence pairs)
          - Common-sense / graph-structured knowledge (Numberbatch)

        Shape: (N, 1068)
        """
        mpnet_embs = self.get_embeddings(words)              # (N, 768), already L2-normalized
        nb_embs    = self.get_numberbatch_embedding(words)   # (N, 300), L2-normalized
        combined   = np.concatenate([mpnet_embs, nb_embs], axis=1)  # (N, 1068)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return combined / norms

    def get_knowledge_graph_ppr_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a Knowledge Graph topological similarity matrix using Personalized PageRank.
        Builds a combined graph from WordNet, ConceptNet, and Datamuse, then measures
        the structural proximity between the 16 target words through all multi-hop connections.
        """
        n = len(words)
        matrix = np.zeros((n, n))
        
        G = nx.Graph()
        
        # Add target words to graph
        for w in words:
            G.add_node(w)
            
        def fetch_neighbors(w):
            neighbors = set()
            # 1. WordNet neighbors
            try:
                syns = wn.synsets(w.lower())
                for syn in syns:
                    for related in syn.hypernyms() + syn.hyponyms():
                        for lemma in related.lemmas():
                            neighbors.add(lemma.name().upper().replace('_', ' '))
            except Exception:
                pass
                        
            # 2. Datamuse
            try:
                dm = self._get_datamuse_associations(w)
                neighbors.update(dm)
            except Exception:
                pass
            
            # 3. ConceptNet
            try:
                cn = self._get_conceptnet_associations(w.lower())
                for c in cn:
                    neighbors.add(c.upper())
            except Exception:
                pass
                
            return w, neighbors

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(fetch_neighbors, words))
            
        for w, neighbors in results:
            for neighbor in neighbors:
                if neighbor != w:
                    G.add_edge(w, neighbor)
                    
        # Apply Personalized PageRank for each target node
        for i, source_word in enumerate(words):
            if source_word not in G:
                continue
            
            personalization = {node: 0.0 for node in G.nodes()}
            personalization[source_word] = 1.0
            
            try:
                # Calculate PPR
                ppr = nx.pagerank(G, alpha=0.85, personalization=personalization, weight=None)
                
                for j, target_word in enumerate(words):
                    if i == j:
                        matrix[i][j] = 1.0
                    else:
                        matrix[i][j] = ppr.get(target_word, 0.0)
            except Exception:
                pass
                
        # Symmetrize to handle directed tendencies conceptually
        matrix = (matrix + matrix.T) / 2
        
        # Scale values up slightly for the ensemble weighting later
        # since PPR values are typically small (sum to 1.0)
        matrix = matrix * len(G.nodes())
        
        return matrix


    def _get_wikipedia_categories(self, word: str) -> set:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": word.title(),
            "prop": "categories",
            "cllimit": "max"
        }
        headers = {"User-Agent": "ConnectionsSolver/1.0"}
        try:
            r = requests.get(url, params=params, headers=headers).json()
            pages = r.get("query", {}).get("pages", {})
            cats = set()
            for k, v in pages.items():
                if "categories" in v:
                    for c in v["categories"]:
                        cats.add(c["title"].replace("Category:", ""))
            return cats
        except Exception:
            return set()

    def get_wikipedia_categories_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a Knowledge Graph similarity matrix using Wikipedia Categories.
        Captures specific sets (e.g., 'Members of the Beatles', 'Constellations').
        """
        n = len(words)
        matrix = np.zeros((n, n))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            cats_list = list(executor.map(self._get_wikipedia_categories, words))
            
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                    
                set_i = cats_list[i]
                set_j = cats_list[j]
                
                # Filter out generic categories with too many words like "All stub articles"
                generic_keywords = {"articles", "terms", "disambiguation", "wikipedia", "cs1", "all "}
                def is_specific(c):
                    return not any(gk in c.lower() for gk in generic_keywords)
                    
                clean_i = {c for c in set_i if is_specific(c)}
                clean_j = {c for c in set_j if is_specific(c)}
                
                intersection = clean_i.intersection(clean_j)
                union = clean_i.union(clean_j)
                
                sim = 0.0
                if union:
                    sim = len(intersection) / len(union)
                    
                matrix[i][j] = sim * 5.0 # Scale up for center-normalization later
                matrix[j][i] = sim * 5.0
                
        return matrix
        
    # --- SYMBOLIC REASONING FEATURE EXTRACTORS ---
    
    def _get_datamuse_phrases(self, word: str) -> tuple[set, set]:
        """Returns sets of common left (bgb) and right (bga) bounding words"""
        left_words, right_words = set(), set()
        url = "https://api.datamuse.com/words"
        try:
            # Words that frequently come before
            res_bgb = requests.get(url, params={"rel_bgb": word, "max": 20}).json()
            # Words that frequently come after
            res_bga = requests.get(url, params={"rel_bga": word, "max": 20}).json()
            
            for item in res_bgb:
                left_words.add(item['word'].upper())
            for item in res_bga:
                right_words.add(item['word'].upper())
        except Exception:
            pass
        return left_words, right_words
        
    def get_phrase_completions(self, words: list[str]) -> tuple[list[set], list[set]]:
        """Returns list of left boundary sets and right boundary sets for each word"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(self._get_datamuse_phrases, words))
        lefts = [r[0] for r in results]
        rights = [r[1] for r in results]
        return lefts, rights
        
    def get_explicit_categories(self, words: list[str]) -> tuple[list[set], list[set]]:
        """Returns exact WikiData properties and direct WordNet ancestors per word"""
        # Wikidata
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            wiki_props = list(executor.map(self._get_wikidata_properties, words))
            
        # WordNet Exact Ancestors
        wn_ancestors = []
        for w in words:
            ancestors = set()
            syns = wn.synsets(w.lower())
            for syn in syns:
                # Add the synset itself
                ancestors.add(syn.name())
                # Add all immediate hypernym paths
                for path in syn.hypernym_paths():
                    for ancestor in path:
                        ancestors.add(ancestor.name())
            wn_ancestors.append(ancestors)
            
        return wiki_props, wn_ancestors
        
    def get_phonetics(self, words: list[str]) -> list[list[list[str]]]:
        """Returns all possible phonetic transcriptions for each word from CMU dict"""
        phonetics = []
        for w in words:
            w_lower = w.lower()
            if w_lower in pronouncing_dict:
                phonetics.append(pronouncing_dict[w_lower])
            else:
                phonetics.append([])
        return phonetics
        
    def get_glove_similarity(self, words: list[str]) -> np.ndarray:
        """Returns a similarity matrix using GloVe word vectors"""
        n = len(words)
        vecs = []
        for w in words:
            w_lower = w.lower()
            if w_lower in self.glove:
                vecs.append(self.glove[w_lower])
            else:
                # Try splitting multi-word phrases
                tokens = w_lower.split()
                token_vecs = [self.glove[t] for t in tokens if t in self.glove]
                if token_vecs:
                    vecs.append(np.mean(token_vecs, axis=0))
                else:
                    vecs.append(np.zeros(self.glove.vector_size))
        vecs = np.array(vecs)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        matrix = np.dot(vecs, vecs.T)
        return matrix


    def get_multisense_sim_matrix(self, words: list[str]) -> np.ndarray:
        """
        Multi-sense embedding similarity matrix (Tier 1.1).

        For each word, generates one embedding per WordNet synset using the
        synset's definition: "WORD: <definition>".  For each word pair (i, j)
        we pick the MAXIMUM cosine similarity over all sense combinations.

        This separates polysemous words (e.g. BAT = animal or sports equipment)
        that are mixed together in a plain token embedding.

        Returns L2-normalised pairwise maximum-sense similarity. Shape: (N, N).
        """
        n = len(words)
        # Build list-of-lists: sense_embeddings[i] = (D_i, 768) array
        sense_embeddings = []
        for w in words:
            syns = wn.synsets(w.lower())
            if syns:
                sense_texts = [f"{w}: {syn.definition()}" for syn in syns[:6]]  # cap at 6 senses
            else:
                sense_texts = [w]  # fallback: bare word
            embs = self.model.encode(sense_texts, show_progress_bar=False,
                                     normalize_embeddings=True)  # (D, 768)
            sense_embeddings.append(embs)

        matrix = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                # sim matrix between senses of word i and senses of word j
                sim_block = sense_embeddings[i] @ sense_embeddings[j].T  # (Di, Dj)
                best = float(sim_block.max())
                matrix[i][j] = best
                matrix[j][i] = best
        return matrix

    def get_pairwise_context_sim_matrix(self, words: list[str]) -> np.ndarray:
        """
        Pairwise relational context similarity matrix (Tier 2.2).

        For each ordered pair (A, B) embeds the prompt:
            "<A> and <B> are both"
        These relational embeddings capture shared categorical membership.
        The 16×16 matrix entry (i, j) is cosine similarity between the
        relational embedding of pair (i, j) and pair (j, i), averaged.

        Returns L2-normalised similarity. Shape: (N, N).
        """
        n = len(words)
        # Build N*N prompts for all ordered pairs; diagonal uses solo word
        prompts_ij = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    prompts_ij.append(words[i])
                else:
                    prompts_ij.append(f"{words[i]} and {words[j]} are both")
        all_embs = self.model.encode(prompts_ij, show_progress_bar=False,
                                     normalize_embeddings=True)  # (N^2, 768)
        emb_mat = all_embs.reshape(n, n, -1)  # (N, N, 768)

        matrix = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                # similarity between relational embedding ij and ji
                sim_ij_ji = float(np.dot(emb_mat[i, j], emb_mat[j, i]))
                # also average with cosine between ij and ji context vectors
                # relative to centroid of all relational embeddings of i
                row_i = emb_mat[i]  # (N, 768): all "wi and wk are both" for k
                row_j = emb_mat[j]  # (N, 768)
                # how similar is row_i's embedding towards j to row_j's embedding towards i?
                cross = float(np.dot(row_i[j], row_j[i]))
                sim = (sim_ij_ji + cross) / 2.0
                matrix[i][j] = sim
                matrix[j][i] = sim
        return matrix

    def get_template_context_sim_matrix(self, words: list[str]) -> np.ndarray:
        """
        Template-based context embedding similarity matrix (Tier 2.2 supplement).

        For each word W, embeds three templates:
            "<W> is a type of"
            "<W> belongs to the category of"
            "Examples of <W> include"
        and averages the embeddings (then re-normalises).  The 16×16 matrix is
        the cosine similarity between these averaged context vectors.

        Returns L2-normalised similarity. Shape: (N, N).
        """
        n = len(words)
        templates = [
            "{w} is a type of",
            "{w} belongs to the category of",
            "Examples of {w} include",
        ]
        context_vecs = []
        for w in words:
            texts = [t.format(w=w) for t in templates]
            embs = self.model.encode(texts, show_progress_bar=False,
                                     normalize_embeddings=True)  # (3, 768)
            avg = embs.mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            context_vecs.append(avg)
        vecs = np.array(context_vecs, dtype=np.float32)  # (N, 768)
        return (vecs @ vecs.T).astype(np.float32)


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
