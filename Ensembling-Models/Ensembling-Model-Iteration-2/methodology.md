# Proposed Methodology: NYT Connections NLP Solver

*Multi-Signal Similarity Ensemble with Contrastive Fine-Tuning, Extended Features, and Beam Search Decoding*

---

## Problem Statement

The NYT Connections puzzle presents 16 words on a board. The player must partition them into exactly 4 groups of 4, where each group shares a hidden theme. The themes are deliberately designed to be tricky — words are chosen to have plausible connections across multiple groups, forcing the solver to commit to one interpretation over another. For example, BASS could belong to a group of fish, musical instruments, or voice types. This ambiguity makes the puzzle a genuinely hard NLP problem, not merely a lookup task.

The challenge we are addressing is: given 16 words, can a system automatically discover the correct grouping using only linguistic and semantic knowledge, without access to the hidden theme labels?

---

## Why This Is Hard

Standard word similarity models fail here for a fundamental reason. If you embed all 16 words in a vector space and cluster them, you get groups based on broad semantic proximity. However, Connections puzzles are specifically designed to defeat this strategy. The puzzle setters deliberately include words that are semantically close but belong to different groups, and words that appear unrelated but share a very specific hidden connection. A solver that relies on a single type of similarity will consistently fall into these traps.

This is why we propose a multi-signal approach: different types of relatedness must be measured simultaneously, and the system must learn which signal is most informative for a given group of words. Our baseline system achieved **24% perfect game accuracy**, which, while meaningful, leaves substantial room for improvement. The updated methodology described here introduces four targeted enhancements informed by that failure analysis.

---

## Stage I — Data Acquisition and Preparation

The input to the system is 16 words. For training and evaluation we use a dataset of approximately 915 historical Connections games, each with ground-truth group labels and difficulty levels (the puzzle assigns each group a colour from easiest to hardest).

We split this dataset chronologically — 640 games for training, 137 for validation, and 138 for testing. The chronological split is deliberate. A random split would allow the model to exploit temporal patterns or repeated themes across splits, producing an over-optimistic evaluation. By preserving time order, we ensure the evaluation reflects performance on genuinely unseen future puzzles.

### NEW — Data Augmentation

A key weakness of the baseline is the small training corpus. With only 640 training games and 2,560 positive group examples, the classifier is at risk of overfitting. We propose **synonym substitution augmentation**: for each training puzzle, we generate synthetic variants by replacing individual words with near-synonyms drawn from WordNet synsets or words with high MPNet cosine similarity. The group structure is preserved while the specific vocabulary changes, effectively multiplying the number of distinct training examples without requiring new labelled puzzles.

For example, if the original puzzle contains GUITAR in an instrument group, a synthetic variant may replace GUITAR with BANJO, preserving the musical instrument category while forcing the classifier to generalise across surface forms.

---

## Stage II — Word Representation

Before computing similarity, each word must be represented numerically. We use three embedding models, each capturing a distinct type of linguistic knowledge, and combine two of them into a joint representation.

### IMPROVED — Fine-Tuned Semantic Embedding

In the baseline, we used off-the-shelf MPNet (`all-mpnet-base-v2`), a Sentence-BERT variant trained on over one billion sentence pairs that produces a 768-dimensional dense vector per word. In the improved system, we **contrastively fine-tune** this model on the 640 training puzzles. Using a triplet or contrastive loss, words belonging to the same ground-truth group are pulled together in embedding space while words belonging to different groups are pushed apart. This domain adaptation means the embedding space is directly optimised for the Connections task rather than for general semantic similarity.

We also evaluate **E5-Large** (1,024-dim) as a stronger base model, as larger sentence transformers consistently outperform MPNet on semantic similarity benchmarks.

### Distributional Embedding (Unchanged)

GloVe, trained on Wikipedia and the Gigaword corpus, learns embeddings by factorising a word co-occurrence matrix. Words that frequently appear near each other in text receive similar vectors. This 100-dimensional representation is shallower than MPNet but complementary — it captures topic-level associations that may not surface in contextual embeddings.

### Common-Sense Embedding (Unchanged)

ConceptNet Numberbatch is derived from the ConceptNet knowledge graph, which encodes structured relational facts such as "a guitar IsA musical instrument" or "playing UsedFor entertainment." Its 300-dimensional vectors encode common-sense world knowledge that does not appear reliably in co-occurrence statistics.

### IMPROVED — Joint Embedding

We concatenate the **fine-tuned** MPNet vector (768-dim) with the Numberbatch vector (300-dim) and L2-normalise the result to produce a **1,068-dimensional joint representation**. Because the base MPNet is now fine-tuned on Connections data, this joint vector captures both distributional semantics optimised for the task and structured common-sense knowledge simultaneously. The joint embedding is used as an additional similarity signal in Stage III and as the primary feature matrix in Stage IV.

---

## Stage III — Pairwise Similarity Computation

Using the word representations above, we compute **nine independent 16×16 similarity matrices** — one per signal type. Each entry S(i,j) in a matrix encodes the similarity between word i and word j according to that specific signal. All nine matrices are subsequently double-centred before fusion.

### Existing Seven Signals (Unchanged in Design)

**Semantic similarity** uses cosine similarity computed on MPNet embeddings. Now that the model is fine-tuned, this signal is directly optimised for the Connections grouping task rather than general-purpose semantic similarity.

**Lexical similarity** uses character n-gram cosine similarity (n = 2 to 4). This captures surface-form patterns such as shared prefixes, suffixes, or embedded subwords. It is critical for categories based on wordplay or spelling, which are invisible to any meaning-based model.

**WordNet similarity** uses the Wu-Palmer score over all synset pairs, taking the maximum across senses. WordNet encodes formal taxonomic relationships, making it the strongest signal for categories that correspond to a named class such as "types of fish" or "musical instruments." This was the highest-weighted signal in the baseline at 38%.

**WikiData similarity** computes Jaccard overlap between the entity property sets retrieved for each word from the WikiData knowledge graph. This captures real-world categorical identity — for example, two members of the same band or two Olympic sports.

**Datamuse similarity** queries the Datamuse API for statistically associated words and computes Jaccard overlap between association sets. This captures idiomatic and collocational relationships — for example, words that all collocate with a hidden bridging word.

**ConceptNet similarity** queries a local SQLite cache of ConceptNet for the neighbourhood of each word and computes Jaccard overlap. This captures functional relationships encoded in the common-sense graph.

**Wikipedia category similarity** fetches Wikipedia category lists for each word and computes Jaccard overlap after filtering generic maintenance categories. This is particularly effective for trivia-style and pop-culture categories.

### NEW — Phonetic Similarity

We add a phonetic similarity matrix computed from the CMU Pronouncing Dictionary, which is already loaded in the embedding module but was not previously used as a standalone signal. For each word pair, we compute the edit distance between their phoneme sequences and convert it to a similarity score. This signal directly targets categories based on sound — rhyming words, homophones, or words that sound like something else — which are entirely invisible to every existing signal.

### NEW — Morphological Similarity

We add a morphological similarity matrix based on shared lemma roots and affix overlap. Words that share a morphological stem or that can be derived from each other by a common prefix or suffix receive a high score. This targets categories such as "words that become new words when you add a prefix," "words that contain a hidden animal," or "words that are all past tenses of verbs," which no embedding model can reliably detect.

### Double-Centering Normalisation

After computing all nine matrices, we apply **double-centering** to each one independently. For each entry S(i,j), the normalised value is:

```
S̃(i,j) = S(i,j) − S_row_mean(i) − S_col_mean(j) + S_global_mean
```

This removes systematic bias from words that are universally similar to everything else, and places all nine matrices on a comparable scale before fusion. Without this step, matrices with larger absolute values would dominate the ensemble regardless of their informativeness.

---

## Stage IV — Downstream Pipelines

After Stage III we have nine normalised 16×16 similarity matrices. The architecture branches into two parallel pipelines.

### Path A — Inference Pipeline

**Weighted Matrix Fusion.** We take a linear combination of the nine double-centred matrices. In the baseline, weights were hand-tuned across seven signals. In the improved system, weights are learned via a small logistic regression trained on the validation set, allowing the combination to adapt to the expanded signal set. The fused matrix is a single 16×16 similarity matrix integrating all nine signals.

**IMPROVED — Beam Search Solver.** The baseline used a greedy solver: always guess the highest-confidence group first, remove those words, and repeat. This is optimal only if the first guess is always correct, which it is not. We replace it with **beam search of width K=5**. At each step, the solver maintains the five most promising candidate first groups and evaluates each one by looking ahead to the second guess. The first group that leads to the best combined two-step score is selected. This prevents the solver from committing to a slightly wrong first guess that destabilises all subsequent guesses, which was a primary failure mode in the baseline.

**IMPROVED — Uncertainty Filter.** Before each guess, we check whether the Random Forest and Gradient Boosting classifiers agree. If their predicted probabilities differ by more than a tunable threshold, the system skips that candidate group and tries the next most confident one. This conservative strategy sacrifices some recall in favour of higher precision on early guesses, preserving lives for the harder groups that arrive later in the game.

### Path B — Training and Evaluation Pipeline

Path B trains a machine learning ensemble to predict whether any candidate group of 4 words is a true Connections group. This is a supervised binary classification problem: a candidate group is labelled 1 if it exactly matches one of the four ground-truth groups, and 0 otherwise.

**IMPROVED — Extended Feature Extraction.** The baseline extracted 17 compact features per candidate group: five aggregate statistics (mean, minimum, variance, maximum out-group similarity, separation) computed across four matrices. This compression discarded important geometric information. The improved system extracts approximately **60 features**:

- All 6 raw pairwise similarity scores within the group (C(4,2) = 6 pairs), per matrix — capturing the full within-group geometry rather than just its summary statistics.
- The original five aggregate statistics per matrix, retained for interpretability and comparison.
- **Rank-based features**: for each word in the candidate group, its rank among all 16 words when sorted by similarity to the other three group members. If all four words are each other's nearest neighbours in similarity space, the group is almost certainly correct.
- **Cross-matrix consistency features**: the Pearson correlation between within-group pairwise scores across pairs of matrices. High consistency across multiple signals is a strong positive indicator; inconsistency suggests the group may be a spurious cluster in one signal space only.

**Training Sample Construction.** For each training puzzle we include all 4 correct groups as positive examples. We sample up to 8 hard negatives per puzzle — candidate groups sharing exactly 3 of 4 words with a correct group — and up to 8 easy random negatives. Hard negatives are critical because they represent the primary failure mode: the solver selecting an almost-correct group with one wrong word substituted in.

**Classifiers.** We train a Random Forest (200 trees, maximum depth 10, Gini criterion) and a Gradient Boosting classifier (200 trees, learning rate 0.1, subsample 0.8). At prediction time, the probabilities from both classifiers are averaged. The ensemble consistently outperforms either model in isolation by reducing variance through diversity of decision boundaries.

**IMPROVED — Beam Search at Evaluation Time.** Analogously to Path A, the greedy game simulation in the baseline is replaced by beam search of width K=5. The ensemble scores all candidate groups from the remaining words, maintains the five highest-confidence candidates, and selects the first group whose choice leads to the best two-step look-ahead outcome. A consistency check verifies that the predicted groups together form a valid partition of the remaining words before each guess is committed.

---

## Evaluation Design

We evaluate on two primary metrics. **Game accuracy** is the fraction of test puzzles where all 4 groups are correctly identified — the strictest and most meaningful measure of end-to-end performance. **Group accuracy** is the fraction of individual groups correctly identified across all test puzzles, providing a more granular measure that does not penalise near-misses as heavily.

We additionally report a **partial overlap breakdown**: how many guesses shared 3, 2, 1, or 0 words with a correct group. This breakdown is particularly diagnostic — a high proportion of 3/4 overlap failures indicates the model is finding the correct cluster but pulling in one wrong word (a feature quality problem), whereas 1/4 or 2/4 failures indicate the model is not finding the clusters at all (a fundamental signal problem).

We run four ablation conditions to isolate the contribution of each proposed improvement:

1. Baseline (7 signals, 17 features, greedy solver)
2. + Contrastive fine-tuning only
3. + Extended features only (9 signals, ~60 features)
4. Full improved system (fine-tuning + extended features + beam search)

Feature importance analysis from the trained ensemble is reported in all conditions. In the baseline, the single most predictive feature was the minimum pairwise similarity within the candidate group under the joint MPNet+Numberbatch embedding, confirming the intuition that a correct Connections group has no weak links. We expect the extended feature set to surface rank-based and cross-matrix consistency features as additional high-importance signals.

---

## Proposed Contributions

The central claim remains that multi-signal similarity fusion with double-centering is a principled and effective approach to the Connections puzzle, and that no single signal is sufficient. The updated methodology adds four concrete contributions beyond the baseline:

1. **Contrastive domain adaptation** — fine-tuning the semantic embedding model directly on Connections data produces a representation space that better separates genuinely related groups from distractors.

2. **Expanded signal coverage** — two new signals (phonetic and morphological similarity) address an entire class of wordplay-based categories that the seven-signal baseline cannot detect by design.

3. **Richer feature representation** — replacing 17 aggregate statistics with approximately 60 features including raw pairwise scores, rank features, and cross-matrix consistency gives the classifier sufficient information to distinguish correct groups from near-miss candidates.

4. **Beam search decoding** — replacing the greedy solver with look-ahead beam search reduces the commitment errors that were the primary failure mode in the baseline, particularly for ambiguous early guesses that destabilise the remainder of the game.

The ablation study directly tests the marginal contribution of each enhancement, enabling a clean quantitative comparison. The two-path design — rule-based inference (Path A) versus learned ensemble (Path B) — remains, allowing us to assess whether the learned improvements transfer to the inference-only setting via the updated similarity signals and weight learning.

---
