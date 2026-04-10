# NYT-Connection Solver

An advanced, multi-stage machine learning pipeline designed to solve the New York Times "Connections" puzzle. The system leverages an ensemble of four distinct ML models (Random Forest, Gradient Boosting, LightGBM Ranker, and Logistic Regression Fusion) alongside a global partition optimizer to resolve complex linguistic, phonetic, and factual word groupings.

---

## 1. Project Overview

### The Connections Challenge
The NYT Connections puzzle presents 16 words that must be partitioned into 4 groups of 4. Unlike standard clustering tasks, Connections is adversarial:
- **Hard Decoys**: Words are deliberately included to suggest a category that only has 3 members (e.g., "Bass" could be a fish or an instrument).
- **Multiple Relationships**: A word like "Cloud" could relate to "Internet," "Rain," or "Confusion."
- **Combinatorial Explosion**: There are **1,820** possible ways to pick 4 words from 16. A complete 4-group partition must be found among billions of theoretical configurations.

### Our Approach: Multi-Model Ensembling
The solver does not rely on a single LLM or clustering algorithm. Instead, it breaks the problem into three phases:
1.  **Semantic Explosion**: Generating 17 different "points of view" (similarity matrices) for the 16 words.
2.  **Deep Feature Extraction**: Computing 153 features per group to describe cohesion and ambiguity.
3.  **Tiered Decision Pipeline**: Scoring groups via a pointwise/listwise ensemble and resolving the puzzle using global search.

---

## 2. Technical Methodology

### Phase A: Multi-Signal Embedding (17 Signals)
For every 16-word grid, the system builds 17 separate 16x16 similarity matrices. This "multi-view" approach ensures that if one signal is fooled by a decoy, another can override it.

- **Neural Semantic**:
    - **MPNet (Transformer)**: Deep contextual embeddings trained on 1B+ sentence pairs.
    - **GloVe**: Classical distributional vectors (Wiki-Gigaword-100).
    - **Numberbatch**: ConceptNet-powered embeddings focused on common-sense.
- **Knowledge Graphs**:
    - **WordNet**: Taxonomic similarity (Wu-Palmer) using synonym/hypernym hierarchies.
    - **Wikidata v3**: Shared property matching across 100M+ entities.
    - **Wikipedia Categories**: Shared membership in Wikipedia's category tree (e.g., "Beatles members").
    - **ConceptNet**: Edge-based structural proximity (IsA, PartOf, UsedFor).
- **Linguistic & Surface**:
    - **Phonetics**: CMU dictionary edit distance (detects rhymes/homophones).
    - **Lexical**: Character n-grams (prefix/suffix detection).
    - **Morphological**: Token-level structural similarity.
- **Contextual & Symbolic**:
    - **Datamuse Associations**: Statistical co-occurrence in sentences.
    - **Phrase Completions**: Detecting left/right bounding words (e.g., "_____ Cream").
    - **PPR (Personalized PageRank)**: Global graph topology score over a combined knowledge map.

### Phase B: Signal Centering
Raw similarity scores are often biased. The system applies **Double Centering**:
- Subtract row means and column means, then add the global mean.
- This ensures that a word that is "generically similar" to everything (like "Thing") does not unfairly boost group scores.

### Phase C: Deep Feature Engineering (153 Features)
For each of the 1,820 candidate groups, the system extracts 9 features from each of the 17 matrices:
1.  **mean_in**: Average internal cohesion.
2.  **min_in**: **The King of Features**. Measures the "weakest link." If one word doesn't fit, this score drops to zero.
3.  **var_in**: Consistency of the internal relationships.
4.  **max_out**: Similarity to the nearest outsider. Detects potential overlaps.
5.  **separation**: Internal mean minus external mean. Our primary decision metric.
6.  **second_max_out**: Security check against "one-away" decoys.
7.  **avg_top3_out**: General vulnerability to confusion.
8.  **group_ambiguity**: How "generic" the words in this group are relative to the grid.
9.  **triangle_score**: Structural transitivity (A~B, B~C => A~C).

---

## 3. The Ensemble Architecture

The solver uses a tiered hierarchy where each model covers the blind spots of the others.

### Layer 1: Pointwise Discovery (RF & GBM)
- **Random Forest**: 200 trees trained on feature subsets. Provides a stable, non-linear baseline.
- **Gradient Boosting (GBM)**: Trains sequentially. Each round focuses on the specific groups that fooled the previous round. It is exceptionally good at detecting subtle decoys that RF misses.

### Layer 2: Listwise Ranking (LightGBM Lambdarank)
- Rather than asking "Is this group correct?", LightGBM asks: "Given these 1,820 options, which one is the *best*?". 
- It optimizes **NDCG@1** (Normalized Discounted Cumulative Gain). This is the "Judge" that breaks ties when Layer 1 predicts multiple high-probability groups.

### Layer 3: Signal Coordination (LR Fusion)
- A simple Logistic Regression model that only sees the 17 separation scores.
- It learns the **learned reliability** of each signal (e.g., "MPNet is 3x more trustworthy than Lexical"). 
- Prevents noisy or broken signals (like the dead ConceptNet matrix) from corrupting the final decision.

### Layer 4: Global Strategy (solve_ml)
- If the model is uncertain, it abandons the "greedy" approach (picking the best group one-by-one).
- Instead, it searches for the **optimal 4-partition** that maximizes the *sum* of ensemble scores across all 4 groups simultaneously.

---

## 4. Performance & Results

Validated on a sealed **138-game test set** (never seen during training):
- **Perfect Games**: 31.1%
- **Group Accuracy**: 50.4%
- **One-Away Resilience**: The combination of GBM and LightGBM correctly resolves 72% of puzzles containing deliberate 3/4 overlap decoys.

---

## 5. Exhaustive Directory Structure

```text
NYT-Connection_Solver/
├── augmentation/                   # Data expansion modules
│   ├── Bable_net/                  # BabelNet synonym/hypernym logic
│   │   ├── puzzle_generator.py     # Augmented game generator
│   │   ├── dataset.jsonl           # Raw generated puzzles
│   │   └── dataset.csv             # Cleaned augmented samples
│   └── LLM_Based/                  # Gpt-4/Claude based generation
│       ├── llm.py                  # LLM integration script
│       └── connections_dataset.csv # LLM-generated puzzle backup
├── Baseline/                       # Benchmarks
│   ├── Baseline-0-Glove-KMeans/    # Simple zero-shot clustering
│   ├── Baseline-1-Glove-SBERT/     # SBERT embedding logic
│   └── Baseline-2-Final-Baseline-MPNet/ # Transformer baseline (no ensemble)
├── Ensembling-Models/              # Core ML Development
│   └── Ensembling-Model-Iteration-2/ # Current Gold Standard
│       ├── models/                 # Binary pkl files for the ensemble
│       │   ├── clf_rf.pkl          # Random Forest weights
│       │   ├── clf_gbm.pkl         # Gradient Boosting weights
│       │   └── clf_lgbm_ranker.pkl # LightGBM Ranker weights
│       ├── ablation/               # Iterative study results (v1 - v5)
│       │   ├── v1/                 # Initial RF/GBM setup
│       │   ├── v3/                 # Addition of Phonetics/Lexical
│       │   └── v5/                 # Ranker integration
│       ├── results/                # Evaluation logs
│       │   ├── with_aug.log        # Results with 8k games
│       │   └── without_aug.log     # Results with 915 games
│       ├── app.py                  # Streamlit Interface
│       ├── train.py                # Pipeline Orchestrator
│       ├── aug_train.py            # Augmented training entry
│       ├── evaluate.py             # Feature engine & simulator
│       ├── embedder.py             # The 17 similarity signals
│       ├── solver.py               # Global optimizer Class
│       ├── reasoner.py             # LLM explanation generator
│       ├── data_loader.py          # CSV Parser
│       ├── embedder.py             # Feature Extraction
│       └── requirements.txt        # Runtime dependencies
├── Evaluation-Metrics/             # Project success definitions
├── GNN_trying/                     # EXPERIMENTAL: Graph Neural Networks
│   ├── model.py                    # R-GCN Implementation
│   ├── graph.py                    # Subgraph construction
│   └── train.py                    # GNN training loop
├── Proposals/                      # Academic Research Papers
├── connections_dataset.csv         # Main Ground Truth Dataset (915 games)
├── project_report.html             # Visual deep-dive report
└── README.md                       # This documentation
```

---

## 6. Execution Guide

### 1. Training
The models are pre-trained in the `models/` directory, but you can retrain them from scratch.

**Option A: Standard Training (Chronological)**
Uses the original 915 puzzles from the NYT archives.
```bash
python "Ensembling-Models/Ensembling-Model-Iteration-2/train.py"
```

**Option B: Augmented Training (Highly Recommended)**
Uses 8,000+ synthetic puzzles generated via BabelNet and LLMs. This provides the models with far more exposure to "one-away" decoy patterns.
```bash
python "Ensembling-Models/Ensembling-Model-Iteration-2/aug_train.py"
```

---

## 2. Evaluation
Run benchmarks against the sealed test set or generate specific error metrics.

**Full Test Suite:**
```bash
python "Ensembling-Models/Ensembling-Model-Iteration-2/evaluate.py"
```

**Advanced Benchmarking:**
Includes "One-Away" success rates and per-tier accuracy (Yellow vs. Purple).
```bash
python "Ensembling-Models/Ensembling-Model-Iteration-2/evaluate_test_advanced.py"
```

---

## 3. Interactive Solver
The Streamlit application provides an "Engineer Insights" panel where you can see:
- Probability agreement between RF and GBM.
- The separation scores for all 17 matrices.
- LLM-generated reasoning for *why* a group was chosen.

**Launch Command:**
```bash
streamlit run "Ensembling-Models/Ensembling-Model-Iteration-2/app.py"
```

---

## 7. Mathematical Foundations

### Normalized Discounted Cumulative Gain (NDCG)
Used by the LightGBM Ranker to prioritize correct groups:
```text
DCG = Σ (rel_i / log2(rank_i + 1))
NDCG = DCG / IDCG
```
Where `rel_i` is 2 for correct groups, 1 for hard decoys, and 0 for others.

### Double Centering (DC)
Applied to all similarity matrices `S` to remove bias:
```text
DC(S) = S - row_means(S) - col_means(S) + total_mean(S)
```

### Ensemble Weighting
The final confidence score `C` for a group:
```text
C = (0.15 * p_rf) + (0.15 * p_gbm) + (0.50 * score_lgbm) + (0.20 * p_lr)
```

---

## 8. Troubleshooting

### Knowledge Graph API 403 Errors
The Wikidata and Wikipedia scrapers are rate-limited. If you see "Forbidden" errors:
1.  The `embedder.py` will automatically switch to using the `wikidata_cache` and `wikipedia_cache`.
2.  Ensure you have a working internet connection for the initial cache population.

### Missing Model Artifacts
If the models in `Ensembling-Models/Ensembling-Model-Iteration-2/models/` are missing, you must run `train.py` or `aug_train.py` before launching the `app.py`.

### ConceptNet SQLite
If ConceptNet features show 0.00 importance, the `conceptnet_en.sqlite` file is missing. The system will continue to work using the other 16 matrices.

---

## 9. Future Directions
- **GNN Integration**: Replacing the linear LR Fusion with an R-GCN (Relational Graph Convolutional Network) to learn higher-order word relationships.
- **LLM-Reasoning Expansion**: Using the `reasoner.py` to give natural language hints instead of just full solutions.
- **Dynamic Weighting**: Adjusting signal weights in real-time based on the linguistic complexity of the current 16-word grid.