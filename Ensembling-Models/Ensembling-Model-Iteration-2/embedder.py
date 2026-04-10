import os
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
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

class WikidataForbiddenError(Exception):
    """Raised when Wikidata API returns 403 Forbidden."""
    pass

class ConnectionsEmbedder:
    FINETUNED_PATH = "models/mpnet_finetuned"

    def __init__(self, model_name: str = "all-mpnet-base-v2",
                 numberbatch_path: str = "numberbatch-en-19.08.txt.gz"):
        """
        Initialises the sentence transformer and auxiliary word vector models.
        Loads GloVe-100 (via gensim cache) and ConceptNet Numberbatch-300 from
        the local file `numberbatch_path` (word2vec text format, plain English words).

        If models/mpnet_finetuned/ exists (produced by finetune.py), it is loaded
        automatically in place of the off-the-shelf model_name checkpoint.
        """
        if os.path.isdir(self.FINETUNED_PATH):
            load_path = self.FINETUNED_PATH
            print(f"Fine-tuned model detected — loading from: {load_path}")
        else:
            load_path = model_name
            print(f"Loading embedding model: {load_path}...")
        self.model = SentenceTransformer(load_path)
        print("Model loaded.")
        
        # Internal word/category cache to avoid redundant API hits in a 16-word grid
        self._grid_words = None
        self._wikidata_cache = {}
        self._wikipedia_cache = {}
        self._datamuse_cache = {}
        
        print("Loading GloVe word vectors...")
        self.glove = gensim_api.load('glove-wiki-gigaword-100')
        print("GloVe loaded.")
        try:
            print(f"Loading ConceptNet Numberbatch from {numberbatch_path}...")
            from gensim.models import KeyedVectors
            self.numberbatch = KeyedVectors.load_word2vec_format(
                numberbatch_path, binary=False, encoding='utf-8'
            )
            print(f"Numberbatch loaded ({len(self.numberbatch)} words, dim=300).")
        except Exception as e:
            print(f"Warning: Could not load Numberbatch ({e}). Skipping...")
            self.numberbatch = {}

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
        """
        Version 3: Hierarchy Traversal and Keyword Extraction.
        Refined to capture deeper connections (Arthropods -> Taxon) and 
        latent concepts (Sports -> Description Keywords).
        """
        api_url = "https://www.wikidata.org/w/api.php"
        headers = {"User-Agent": "ConnectionsSolver/3.0 (Deepmind Antigravity)"}
        
        # 1. Search for top 5 candidates
        search_params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": word,
            "limit": 5
        }
        
        try:
            resp = requests.get(api_url, params=search_params, headers=headers)
            if resp.status_code == 403:
                raise WikidataForbiddenError("Wikidata API 403 Forbidden. Stopping to avoid block.")
            
            search_res = resp.json()
            if not search_res.get('search'):
                return set()
            
            entity_ids = [item['id'] for item in search_res['search']]
            
            # 2. Bulk fetch data (claims, descriptions, aliases)
            prop_params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(entity_ids),
                "props": "claims|descriptions|aliases",
                "languages": "en"
            }
            resp = requests.get(api_url, params=prop_params, headers=headers)
            if resp.status_code == 403:
                raise WikidataForbiddenError("Wikidata API 403 Forbidden. Stopping to avoid block.")
                
            prop_res = resp.json()
            all_entities = prop_res.get('entities', {})
            
            properties = set()
            target_ids = set() 
            
            # Key properties (including Taxon, Material, Usage, Part of)
            PRIORITY_PROPS = {"P31", "P279", "P171", "P361", "P106", "P101", "P910", "P186", "P366"}

            # Pass 1: Collect Item IDs for labels + Keywords from descriptions/aliases
            for eid, ent_data in all_entities.items():
                # Add aliases
                for alias in ent_data.get('aliases', {}).get('en', []):
                    properties.add(alias['value'].lower())
                
                # Add keywords from description
                desc = ent_data.get('descriptions', {}).get('en', {}).get('value', '').lower()
                if desc:
                    # Simple keyword extraction
                    words = desc.replace(",", " ").replace(".", " ").split()
                    for w in words:
                        if len(w) > 3: properties.add(w)

                claims = ent_data.get('claims', {})
                for pid, claim_list in claims.items():
                    if pid not in PRIORITY_PROPS: continue
                    for claim in claim_list:
                        try:
                            val = claim['mainsnak']['datavalue']['value']
                            if isinstance(val, dict) and val.get('entity-type') == 'item':
                                target_ids.add(val['id'])
                        except KeyError: continue

            # 3. Step 2 Hierarchy (Follow P171/P279 one step deeper)
            if target_ids:
                h_params = {
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(list(target_ids)[:50]),
                    "props": "claims|labels",
                    "languages": "en"
                }
                resp = requests.get(api_url, params=h_params, headers=headers)
                if resp.status_code == 403:
                    raise WikidataForbiddenError("Wikidata API 403 Forbidden. Stopping to avoid block.")
                h_res = resp.json()
                label_map = {}
                for tid, t_ent in h_res.get('entities', {}).items():
                    label = t_ent.get('labels', {}).get('en', {}).get('value')
                    if label: label_map[tid] = label.lower()
                    
                    h_claims = t_ent.get('claims', {})
                    for p_id in ["P171", "P279"]:
                        for c in h_claims.get(p_id, []):
                            try:
                                h_val = c['mainsnak']['datavalue']['value']['id']
                                target_ids.add(h_val) 
                                if p_id == "P171": properties.add(h_val)
                            except (KeyError, TypeError): continue

                final_ids = [tid for tid in target_ids if tid not in label_map]
                if final_ids:
                    l_params = {"action": "wbgetentities", "format": "json", "ids": "|".join(final_ids[:50]), "props": "labels", "languages": "en"}
                    resp = requests.get(api_url, params=l_params, headers=headers)
                    if resp.status_code == 403:
                        raise WikidataForbiddenError("Wikidata API 403 Forbidden. Stopping to avoid block.")
                    l_res = resp.json()
                    for tid, t_ent in l_res.get('entities', {}).items():
                        label = t_ent.get('labels', {}).get('en', {}).get('value')
                        if label: label_map[tid] = label.lower()

            for tid in target_ids:
                properties.add(tid)
                if tid in label_map: properties.add(label_map[tid])
            
            return properties
        except WikidataForbiddenError as e:
            # Let this bubble up so the caller can stop the run
            raise e
        except Exception:
            return set()

    def _ensure_grid_cache(self, words: list[str]):
        """Populates the internal 1:N caches for the current 16-word grid."""
        if self._grid_words == words:
            return # Already cached
        
        self._grid_words = words
        self._wikidata_cache = {}
        self._wikipedia_cache = {}
        self._datamuse_cache = {}
        
        # 1. Wikidata (Sequential/Polite)
        for w in words:
            self._wikidata_cache[w] = self._get_wikidata_properties(w)
        
        # 2. Wikipedia (Parallel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            cats = list(executor.map(self._get_wikipedia_categories, words))
        for w, c in zip(words, cats):
            self._wikipedia_cache[w] = c
            
        # 3. Datamuse (Parallel)
        def _get_assoc(word):
            try:
                res = requests.get(f"https://api.datamuse.com/words?rel_trg={word}&max=40").json()
                return {r['word'].lower() for r in res}
            except: return set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            assocs = list(executor.map(_get_assoc, words))
        for w, a in zip(words, assocs):
            self._datamuse_cache[w] = a

    def get_grid_property_uniqueness_matrix(self, words: list[str]) -> np.ndarray:
        """
        Calculates a 'Property Uniqueness' matrix (Tier 4 Blueprint).
        """
        n = len(words)
        self._ensure_grid_cache(words)
        
        properties_per_word = [self._wikidata_cache[w] for w in words]
        
        # Count global occurrences in the grid
        global_property_counts = {}
        for props in properties_per_word:
            for p in props:
                global_property_counts[p] = global_property_counts.get(p, 0) + 1
        
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                shared = properties_per_word[i].intersection(properties_per_word[j])
                if not shared:
                    continue
                
                # Check uniqueness of shared properties
                max_uniqueness_score = 0.0
                for p in shared:
                    count = global_property_counts[p]
                    if count == 4:
                        # PERFECT MATCH: This property defines exactly 4 words in the grid
                        max_uniqueness_score = max(max_uniqueness_score, 1.0)
                    elif count == 3:
                        max_uniqueness_score = max(max_uniqueness_score, 0.4)
                    elif count > 4 and count <= 6:
                        # TRAP DETECTED: Property is too common, reduce score (ignore or negative)
                        max_uniqueness_score = max(max_uniqueness_score, 0.1)
                    else:
                        # Too generic (count > 6 or count < 3)
                        max_uniqueness_score = max(max_uniqueness_score, 0.05)
                
                matrix[i][j] = max_uniqueness_score
                matrix[j][i] = max_uniqueness_score
        
        return matrix

    def get_wikidata_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a Knowledge Graph similarity matrix using WikiData v3.
        """
        n = len(words)
        self._ensure_grid_cache(words)
        props_list = [self._wikidata_cache[w] for w in words]
        
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                set_i, set_j = props_list[i], props_list[j]
                intersection = set_i.intersection(set_j)
                union = set_i.union(set_j)
                sim = len(intersection) / len(union) if union else 0.0
                matrix[i][j] = matrix[j][i] = sim * 5.0
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
        self._ensure_grid_cache(words)
        cats_list = [self._wikipedia_cache[w] for w in words]
        
        matrix = np.zeros((n, n))
            
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
        
    def get_grid_category_uniqueness_matrix(self, words: list[str]) -> np.ndarray:
        """
        Grid-Aware Wikipedia Category Uniqueness (Tier 4 Blueprint).
        Finds categories that belong to EXACTLY 4 words in the grid.
        """
        n = len(words)
        self._ensure_grid_cache(words)
        cats_list = [self._wikipedia_cache[w] for w in words]
        
        # Count global occurrences of categories in the grid
        global_cat_counts = {}
        for cats in cats_list:
            for c in cats:
                global_cat_counts[c] = global_cat_counts.get(c, 0) + 1
        
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                shared = cats_list[i].intersection(cats_list[j])
                if not shared: continue
                
                score = 0.05
                for c in shared:
                    count = global_cat_counts[c]
                    if count == 4:
                        score = max(score, 1.0)
                    elif count == 3:
                        score = max(score, 0.4)
                matrix[i, j] = matrix[j, i] = score
        return matrix

    def get_grid_datamuse_uniqueness_matrix(self, words: list[str]) -> np.ndarray:
        """
        Grid-Aware Datamuse association uniqueness.
        """
        n = len(words)
        self._ensure_grid_cache(words)
        assoc_list = [self._datamuse_cache[w] for w in words]
            
        global_assoc_counts = {}
        for assocs in assoc_list:
            for a in assocs:
                global_assoc_counts[a] = global_assoc_counts.get(a, 0) + 1
        
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                shared = assoc_list[i].intersection(assoc_list[j])
                if not shared: continue
                
                score = 0.05
                for a in shared:
                    count = global_assoc_counts[a]
                    if count == 4: score = max(score, 1.0)
                    elif count == 3: score = max(score, 0.3)
                matrix[i, j] = matrix[j, i] = score
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
        # Reduced max_workers=1 to be polite to the API and avoid 403 blocks
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
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

    def get_phonetic_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a phonetic similarity matrix based on CMU phoneme edit distance.

        For each word pair, computes the normalised edit distance between their
        phoneme sequences and converts it to a similarity score. Words with
        multiple pronunciations use the best-matching pair across all combinations.
        Words absent from the CMU dict receive 0.0 similarity against all others.
        """
        phonetics = self.get_phonetics(words)
        n = len(words)
        matrix = np.zeros((n, n))

        def _edit_dist(seq1, seq2):
            m, k = len(seq1), len(seq2)
            if m == 0 or k == 0:
                return max(m, k)
            dp = [[0] * (k + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(k + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, k + 1):
                    cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                    dp[i][j] = min(dp[i-1][j] + 1,
                                   dp[i][j-1] + 1,
                                   dp[i-1][j-1] + cost)
            return dp[m][k]

        def _best_sim(trans_i, trans_j):
            if not trans_i or not trans_j:
                return 0.0
            best = 0.0
            for seq_i in trans_i:
                for seq_j in trans_j:
                    dist = _edit_dist(seq_i, seq_j)
                    max_len = max(len(seq_i), len(seq_j))
                    sim = 1.0 - dist / max_len if max_len > 0 else 1.0
                    if sim > best:
                        best = sim
            return best

        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = _best_sim(phonetics[i], phonetics[j])
                matrix[i][j] = sim
                matrix[j][i] = sim

        return matrix

    def get_morphological_similarity(self, words: list[str]) -> np.ndarray:
        """
        Returns a morphological similarity matrix based on shared lemma roots
        and affix (prefix/suffix) overlap.

        For each word pair:
          - Lemma match : 1.0 if both words reduce to the same lemma
          - Affix Jaccard: Jaccard similarity of prefix/suffix sets (length 3-5)
          - Final score  : max(lemma_match, affix_jaccard)
        """
        lemmatizer = WordNetLemmatizer()
        n = len(words)
        matrix = np.zeros((n, n))

        def _lemmatize(word):
            w = word.lower()
            # Take the shortest form across noun/verb/adjective — most reduced root
            candidates = [
                lemmatizer.lemmatize(w, 'n'),
                lemmatizer.lemmatize(w, 'v'),
                lemmatizer.lemmatize(w, 'a'),
            ]
            return min(candidates, key=len)

        def _affixes(word):
            affixes = set()
            for k in range(3, 6):
                if len(word) >= k:
                    affixes.add('PRE_' + word[:k])
                    affixes.add('SUF_' + word[-k:])
            return affixes

        lemmas     = [_lemmatize(w) for w in words]
        affix_sets = [_affixes(l)   for l in lemmas]

        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                lemma_score = 1.0 if lemmas[i] == lemmas[j] else 0.0
                a, b = affix_sets[i], affix_sets[j]
                jaccard = len(a & b) / len(a | b) if (a or b) else 0.0
                score = max(lemma_score, jaccard)
                matrix[i][j] = score
                matrix[j][i] = score

        return matrix

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
