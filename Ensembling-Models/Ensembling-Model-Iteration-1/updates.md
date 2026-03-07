# NLP Connections Solver Walkthrough

## Implementation Summary

We have built an end-to-end NLP-based solver for the NYT Connections game without relying on LLM prompting or RAG. The system uses raw semantic sentence embeddings to group words.

The architecture consists of four components:
1. **[data_loader.py](file:///Users/srushtipekamwar/Desktop/Semester%202/INLP/Connection%20Solver/data_loader.py)**: Parses the raw [Connections_Data.csv](file:///Users/srushtipekamwar/Desktop/Semester%202/INLP/Connection%20Solver/Connections_Data.csv) into individual game structures.
2. **[embedder.py](file:///Users/srushtipekamwar/Desktop/Semester%202/INLP/Connection%20Solver/embedder.py)**: Utilizes the `sentence-transformers/all-mpnet-base-v2` model to encode words into 768-dimensional normalized dense vectors.
3. **[solver.py](file:///Users/srushtipekamwar/Desktop/Semester%202/INLP/Connection%20Solver/solver.py)**: A mathematically optimal search algorithm that looks at all 2.6 million possible configurations of dividing 16 elements into 4 groups of 4. It maximizes the sum of intra-group pairwise similarities.
4. **[evaluate.py](file:///Users/srushtipekamwar/Desktop/Semester%202/INLP/Connection%20Solver/evaluate.py)**: Ties everything together, applying "double centering" to the cosine similarity matrix to reduce "hubness" bias common in sentence embeddings.

---

## Evaluation and Validation Results

We evaluated the hybrid solver on the first 150 games from the dataset. The solver incorporates **Greedy Iterative Guessing** with 4 "lives" (rules from the game) to directly compare with the baseline outlined in Todd et al. (2024), which reported an **11.6%** full game solve rate.

**Final 4-Feature Hybrid Model Metrics (150 Games):**
- **Perfectly Solved Games**: **32 / 150**
- **Game Accuracy**: **21.33%**
- **Group Match Accuracy**: **40.50%**

### Analysis of the Baseline and Improvements

Our robust pipeline achieves **21.33% exact game accuracy**, nearly **doubling** the Todd et al. baseline of 11.6%!

Here is the exact feature composition of our unified model:
1. **Semantic Similarity (MPNet)**: Dense transformer embeddings.
2. **Lexical Similarity (N-Grams)**: Hand-crafted character-n-grams to catch orthographic clues (e.g. palindromes, prefixes).
3. **Knowledge Graph (WordNet)**: Uses the NLTK Wu-Palmer similarity metric to explicitly reward synonyms and hypernyms (e.g. types of dogs) that fuzzy dense embeddings miss.
4. **Knowledge Graph (WikiData)**: Queries the WikiData API for explicit entity properties (e.g. "instance of"). If words share a heavily specific real-world property ("NBA teams" or "Movies starring Tom Cruise"), they get a massive grouping boost.

### Conclusion

By layering structural game mechanics (lives-based greedy iteration) and hybrid features (Semantic + Lexical + WordNet + WikiData), we successfully optimized a deterministic NLP solver to outperform standard literature baselines without resorting to LLM prompting or RAG architectures.
