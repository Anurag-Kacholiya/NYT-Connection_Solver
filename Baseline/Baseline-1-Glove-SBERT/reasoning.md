### Why Baseline 1 Failed: Architectural and Methodological Flaws

The baseline architecture successfully clustered ~22.5% of the groups (90 out of 400 in the test set) but completely failed to solve a single full game. This partial success followed by catastrophic game-level failure highlights fundamental limitations in relying purely on dense semantic vectors and distance-based clustering for constraint satisfaction tasks.

#### 1. The "Red Herring" Vulnerability (Semantic Distractors)

Sentence transformers like `all-MiniLM-L6-v2` are explicitly trained to maximize the cosine similarity of words that appear in similar textual contexts. NYT Connections designers actively exploit this by embedding "red herrings"—words that are highly semantically related but belong to completely different groups. For example, if "HEAT" (NBA Team) and "SNOW" (Wet Weather) appear in the same puzzle, the embeddings will place them close together in the vector space. Agglomerative clustering blindly merges these nearest neighbors, falling directly into the human-designed traps.

#### 2. Blindness to Lexical and Lateral Wordplay

Standard embeddings map the *meaning* of a word, completely discarding its structural or phonetic properties. A significant portion of Connections categories rely on lateral thinking: palindromes (LEVEL, KAYAK), shared prefixes (PRE-, PRO-), or homophones. The sentence transformer has zero mathematical awareness of these string-level constraints, rendering it incapable of clustering them unless by pure random chance.

#### 3. Polysemy and Lack of Contextual Disambiguation

Words in these puzzles are highly polysemous (having multiple meanings). "SHIFT" can be a work schedule, a movement, or a keyboard key. Because the baseline encodes each word independently `embeddings = model.encode(words)`, the transformer defaults to the word's most statistically common meaning. Without a mechanism to look at the other 15 words and dynamically shift the embedding vector toward a secondary definition (e.g., realizing "SHIFT" is next to "TAB" and "RETURN"), the static representation breaks the grouping.

#### 4. The Greedy Nature of Agglomerative Clustering

Agglomerative clustering is a bottom-up, greedy algorithm. It finds the two vectors with the highest cosine similarity and locks them together permanently.  If it makes a single incorrect merge early in the process—such as pairing a red herring—it cannot backtrack or re-evaluate. Solving a 16-word puzzle requires global constraint satisfaction (ensuring all four groups of four are mathematically valid simultaneously), which a greedy distance-based algorithm physically cannot perform.

---

This baseline perfectly illustrates why zero-shot semantic similarity is insufficient for this task. It can pick off the one "obvious" semantic category (hence the 90 solved groups), but leaves the remaining words scrambled, guaranteeing a game-level solve rate of 0.

---

## outputs :


On 100 (random) puzzels :
```
===== FINAL EVALUATION METRICS =====

Evaluating Baseline 1 on 100 puzzles...
100%|██████████| 100/100 [00:28<00:00, 31.86it/s]
1) Grouping Accuracy :
   1.1) total no. of group solved: 90 (out of 400)
   1.2) no. of '3 words correct out od 4 in a group': 79
____
2) no. of games in which (2 groups solved completely and remaining 2 groups are '3 words correct out od 4 in a group'): 0
____
3) no. of games solved completely (out of 100 test games): 0
____

```

On all 914 puzzels
```
Evaluating Baseline 1 on 914 puzzles...
100%|██████████| 914/914 [00:28<00:00, 31.86it/s]
===== FINAL EVALUATION METRICS =====
1) Grouping Accuracy :
   1.1) total no. of group solved: 833 (out of 3656)
   1.2) no. of '3 words correct out od 4 in a group': 644
____
2) no. of games in which (2 groups solved completely and remaining 2 groups are '3 words correct out od 4 in a group'): 1
____
3) no. of games solved completely (out of 914 test games): 1
_____
```