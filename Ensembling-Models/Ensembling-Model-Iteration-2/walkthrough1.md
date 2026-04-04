# Connections Solver — Architecture Upgrade Walkthrough

## What Was Done

All **Tier 1** and **Tier 2** improvements from the research-grade architecture plan were implemented across 4 files, with the strict 640/137/138 train/val/test split fully preserved.

---

## Files Changed

| File | What Changed |
|------|-------------|
| [embedder.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/embedder.py) | +3 new methods |
| [evaluate.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate.py) | Full rewrite — 7 matrices, 9 features/matrix, candidate filtering, ranker blending |
| [train.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/train.py) | Full rewrite — adds LightGBM ranker training as Step 3/4 |
| [evaluate_test.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate_test.py) | Updated to load ranker + pass new matrix flags |

---

## Tier 1 Upgrades

### 1. Multi-sense Embeddings ([embedder.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/embedder.py) → [get_multisense_sim_matrix](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/embedder.py#567-602))

For each word, embeds one sentence per WordNet synset: `"WORD: <synset definition>"` (capped at 6 senses). For each pair (i, j) the **maximum** cosine similarity over all sense combinations is used.

**Why it helps:** Plain embeddings mix all senses of "BAT" (animal + sports). This separates them into distinct sense clusters, suppressing distractor confusion.

**New matrix:** `MultiSense_sim`

---

### 2. Learning-to-Rank — LightGBM Lambdarank ([train.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/train.py) → [train_lgbm_ranker](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/train.py#109-207))

Trains a `lightgbm` ranker with `objective=lambdarank`. Training data per game:
- Correct groups → relevance **2**
- 3-overlap hard negatives → relevance **1**
- Random negatives → relevance **0**

At inference, scores are blended: **35% RF + 35% GBM + 30% Ranker**

**Why it helps:** Optimises NDCG (ranking) not just log-loss (binary classification). Forces the model to order correct groups above distractors, not just classify them.

**New model saved:** `models/clf_lgbm_ranker.pkl`

---

### 3. Two-Stage Candidate Filtering ([evaluate.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate.py) → [generate_candidates](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate.py#215-278))

At the start of each game:
1. Average all matrices → single combined similarity
2. Convert to distance, run **agglomerative clustering** (average linkage)
3. Sweep dendrogram thresholds, collect all size-4 flat clusters
4. Score by in-group mean similarity, keep **top 80**
5. Fall back to all 1820 subsets if < 80 found

**Why it helps:** Eliminates 95%+ of noisy subsets before ML scoring. The model focuses where it matters, reducing false positive distractors.

---

## Tier 2 Upgrades

### 4. Enhanced Cross-Group Penalty Features ([evaluate.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate.py) → [extract_features](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate.py#135-211))

Per matrix, 3 new features added:

| Feature | Description |
|---------|-------------|
| `second_max_out` | 2nd highest similarity from any group member to any outsider |
| `avg_top3_out` | Mean of the 3 highest out-of-group similarities |
| `group_ambiguity` | Sum of each group member's mean similarity to ALL 12 outsiders |

**Why it helps:** `max_out` alone is easily tricked. These features characterise how broadly a group leaks into other groups.

---

### 5. Pairwise Context Embeddings ([embedder.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/embedder.py))

Two new methods:
- **[get_pairwise_context_sim_matrix](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/embedder.py#603-643)**: Embeds `"<A> and <B> are both"` for all 256 ordered pairs, creates 16×16 relational similarity matrix. → `PairContext_sim`
- **[get_template_context_sim_matrix](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/embedder.py#644-675)**: Embeds 3 templates per word (`"<W> is a type of"`, `"<W> belongs to category of"`, `"Examples of <W> include"`), averages them per word. → `TemplateCtx_sim`

**Why it helps:** Creates implicit relational context without any LLM API call. Captures categorical membership signals not in static embeddings.

---

### 6. Triangle Consistency Feature ([evaluate.py](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate.py) → [extract_features](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/evaluate.py#135-211))

For each of the 4 triplets in a size-4 group, checks if transitivity holds: if A~B **and** B~C then A~C must hold (threshold = mean in-group similarity). Score = fraction of consistent triplets.  → `triangle_score`

**Why it helps:** Punishes "fake clusters" where 3 words are tight but the 4th is a distractor that breaks the chain.

---

## Summary: Feature Pipeline

```
7 matrices × 9 features/matrix = 63 total features
```

| Matrix | Source |
|--------|--------|
| MPNet+Numberbatch_sim | ✅ original |
| MPNet_sim | ✅ original |
| WordNet_sim | ✅ original |
| GloVe_sim | ✅ original |
| MultiSense_sim | 🆕 Tier 1.1 |
| PairContext_sim | 🆕 Tier 2.2 |
| TemplateCtx_sim | 🆕 Tier 2.2 |

---

## Smoke Test Results

```
✓ Imports OK
  ALL_FEATURE_NAMES has 63 entries
✓ LightGBM importable
✓ Matrices built: ['MPNet+Numberbatch_sim', 'MPNet_sim', 'WordNet_sim',
                   'GloVe_sim', 'MultiSense_sim', 'PairContext_sim', 'TemplateCtx_sim']
  Feature count  : 63
  Feature vector length : 63
✓ Feature vector length matches feature names
✓ generate_candidates returned candidates (all size 4)
✓ simulate_game_ml works in all 3 modes (no ranker / with ranker / with candidates)
ALL SMOKE TESTS PASSED
```

---

## How to Run

**Full training + evaluation (train/val/test split):**
```bash
cd "/Users/srushtipekamwar/Desktop/Semester 2/Connections/xg boost try"
python train.py --csv Connections_Data.csv --out-dir models
```

Expected output:
- Steps 1–4 printed (features → RF → GBM → Ranker → Save)
- [models/clf_rf.pkl](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/models/clf_rf.pkl), [models/clf_gbm.pkl](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/models/clf_gbm.pkl), `models/clf_lgbm_ranker.pkl`
- [feature_weights.txt](file:///Users/srushtipekamwar/Desktop/Semester%202/Connections/xg%20boost%20try/feature_weights.txt) showing 63-feature importances
- `[VAL]` and `[TEST]` results with the **frozen** saved models

**Test-only evaluation (after training):**
```bash
python evaluate_test.py
```

> [!NOTE]
> Training is significantly slower now because of the 3 new embedding calls per game. For 640 training games, expect ~2–4 hours depending on hardware. On CPU, consider running overnight.
