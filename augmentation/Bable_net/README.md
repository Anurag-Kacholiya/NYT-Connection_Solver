# NYT Connections Puzzle Generator — v7
### A Synthetic Data Augmentation System for NLP Training

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Are We Building This?](#2-why-are-we-building-this)
3. [The Big Picture — How It All Fits Together](#3-the-big-picture--how-it-all-fits-together)
4. [Knowledge Sources — The Four Pillars](#4-knowledge-sources--the-four-pillars)
5. [The Puzzle Structure — 4 Levels Explained](#5-the-puzzle-structure--4-levels-explained)
6. [The Core Workflow — Step by Step](#6-the-core-workflow--step-by-step)
7. [Three Architectures — A, B, and C](#7-three-architectures--a-b-and-c)
8. [Quality Controls — Why Each One Exists](#8-quality-controls--why-each-one-exists)
9. [The L4 Hardcoded Stem Pool — The Key Fix in v7](#9-the-l4-hardcoded-stem-pool--the-key-fix-in-v7)
10. [Architecture Diagram (Text)](#10-architecture-diagram-text)
11. [Running the System](#11-running-the-system)
12. [Output Format (JSONL)](#12-output-format-jsonl)
13. [Design Decisions — The "Why" Behind Everything](#13-design-decisions--the-why-behind-everything)

---

## 1. What Is This Project?

This is a **synthetic puzzle generator** that creates training data shaped like the NYT Connections game.

In NYT Connections, you are shown a 4×4 grid of 16 words. Your job is to find 4 hidden groups of 4 words — each group sharing a secret theme. For example:

```
GRID:
  apple    mango    grape    cherry
  swift    rapid    fleet    brisk
  smoke    flame    ember    blaze
  foot     basket   base     snow

ANSWERS:
  🟡 YELLOW  →  FRUITS              (apple, mango, grape, cherry)
  🟢 GREEN   →  WORDS MEANING FAST  (swift, rapid, fleet, brisk)
  🔵 BLUE    →  ASSOCIATED WITH FIRE (smoke, flame, ember, blaze)
  🟣 PURPLE  →  ___ BALL            (foot, basket, base, snow)
```

This system **automatically generates thousands of such puzzles** so that an NLP model can be trained to understand and solve them.

---

## 2. Why Are We Building This?

Training a model to solve Connections puzzles requires a **large, diverse, high-quality dataset**. But real NYT puzzles are:

- **Limited in number** — only one published per day
- **Copyright protected** — cannot be used freely for training
- **Not labeled** — raw word grids don't come with the semantic annotations a model needs

So we build a pipeline that:

1. Analyses the patterns and rules of real Connections puzzles
2. Creates a **meta-prompt** capturing those rules
3. Uses that prompt to drive a **programmatic generation engine** (this code)
4. Outputs puzzles as structured **JSONL** records with full labels

The result is an **augmented dataset** of synthetic puzzles that mirror real ones closely enough to train on.

---

## 3. The Big Picture — How It All Fits Together

```
CURRENT CONNECTIONS DATASET (Original Data)
        |
        |  Input
        v
+---------------------------+
|   LLM — Analysis Phase    |
|  - Pattern Analysis       |   <-- Understand word relationships,
|  - Logic Extraction       |       difficulty, categories, game rules
+---------------------------+
        |
        |  Creates Prompt
        v
+---------------------------+
|      META-PROMPT          |   <-- Generative rules & templates
|  (Generative Rules        |       that encode how puzzles work
|   & Templates)            |
+---------------------------+
        |
        |  User inputs N (number of puzzles)
        v
+-------------------------------------------------------+
|              PYTHON SCRIPT                            |
|          (Automation & Orchestration)                 |
|                                                       |
|  +------------------+      +----------------------+  |
|  |  Request Manager |<---->|     LLM Client       |  |
|  +------------------+      +----------------------+  |
|           |                         |                 |
|           |                +--------v---------+       |
|           |                |   DATA PARSER    |       |
|           |                +------------------+       |
+----------|----------------------------|--------------+
           |  Sends Meta-Prompt         |  Output
           |  & N Requests              v
           v                  AUGMENTED CONNECTIONS
   LLM Generation Phase       DATASET (Synthetic Data)
   - Text Generation             (JSONL records)
   - Pattern Adherence
   - Variety Control
```

The Python script in this repo is **the Python Script box** above. It handles everything from anchor selection through puzzle assembly and JSONL output.

---

## 4. Knowledge Sources — The Four Pillars

The system uses **four freely available knowledge sources**, each chosen for a specific role:

### 4.1 BabelNet v9 (Online REST API)

**What it is:** A massive multilingual encyclopedic dictionary that links concepts across many languages and knowledge bases, including WordNet.

**Why we use it — Sense Disambiguation:**

The word "apple" can mean a fruit OR the technology company. The word "python" can mean a snake OR the programming language. If you just ask WordNet for hyponyms of "python," you might get programming terms mixed with reptiles.

BabelNet solves this by mapping any word to its **correct sense** before we query WordNet. We call `getSenses(anchor)`, filter results where `source=WN`, `language=EN`, and `bKeySense=true`, then extract the `wordNetOffset`. This gives us the exact synset ID (e.g., `edible_fruit.n.01`) for the sense we actually want.

**Cost:** Free tier — 50,000 queries per day (enough for hundreds of puzzles).

---

### 4.2 WordNet via NLTK (Offline Lexical Database)

**What it is:** A large English lexical database where words are grouped into sets of cognitive synonyms called "synsets," linked by semantic relationships.

**Why we use it — Semantic Structure:**

WordNet gives us three critical things:

- **Hyponym trees** — "what are all the types of X?" (e.g., all types of fruit → apple, mango, grape...) used for L1 groups
- **Synonym sets** — "what words mean the same as X?" used for L2 groups
- **Meronym relations** — "what are the parts of X?" (e.g., parts of a car → wheel, brake, hood...) used in Architecture C

WordNet is **offline**, so there are no API limits and queries are instant.

---

### 4.3 Datamuse API (Online REST API)

**What it is:** A word-finding API that supports queries like "words triggered by X", "words that mean like X", "words that often appear before/after X", and "words that often appear near X in text."

**Why we use it — Thematic Association & Compound Words:**

WordNet is great for strict taxonomy but poor at *thematic* or *cultural* associations. For example, WordNet won't tell you that "smoke," "ember," and "blaze" are all associated with "fire" — they aren't hyponyms or synonyms of fire. Datamuse's `triggered_by()` and `rel_bga=` (co-occurrence) queries capture exactly this kind of soft associative link, which is what L3 groups need.

Datamuse also powers L4 via `words_before()` and `words_after()` to find valid compound word stems.

**Cost:** Free — no API key required — ~100 requests/second.

---

### 4.4 Brown Corpus via NLTK (Offline Text Corpus)

**What it is:** A one-million-word corpus of American English text from the 1960s, included in NLTK. Used here purely as a **word frequency reference**.

**Why we use it — Ensuring Common Vocabulary:**

A puzzle must use words that any educated English speaker would know. Without frequency filtering, the system could pick valid but obscure hyponyms like "juneberry" (a type of fruit) or "grilse" (a type of fish) — technically correct but terrible puzzle words.

The Brown Corpus frequency table lets us enforce `freq >= 8` — meaning a word must appear at least 8 times in a million-word sample. This is a simple but highly effective proxy for "commonly known English word."

**Why Brown Corpus specifically?** It is small, fast to load offline, already bundled with NLTK, and its frequency ranks correlate well with general vocabulary familiarity. It is not perfect but it is free, reproducible, and sufficient.

---

## 5. The Puzzle Structure — 4 Levels Explained

Every puzzle has exactly 4 groups, each with exactly 4 words:

| Level | Colour | Difficulty | Type | Example |
|-------|--------|------------|------|---------|
| **L1** | 🟡 Yellow | Easiest | Concrete category (hyponyms) | apple, mango, grape, cherry → **FRUITS** |
| **L2** | 🟢 Green | Medium | Synonym group | swift, rapid, fleet, brisk → **WORDS MEANING FAST** |
| **L3** | 🔵 Blue | Medium-Hard | Associative/thematic | smoke, flame, ember, blaze → **ASSOCIATED WITH FIRE** |
| **L4** | 🟣 Purple | Hardest | Compound word pattern | foot, basket, base, snow → **___ BALL** |

The difficulty gradient is intentional:
- **L1** is straightforward — the connection is a named category
- **L2** requires knowing synonyms, not just categories  
- **L3** requires cultural/thematic knowledge, not dictionary lookup
- **L4** is the "trick" level — the words seem unrelated until you realise they all precede (or follow) a single word

---

## 6. The Core Workflow — Step by Step

Your understanding is **correct**. Here is the precise flow for each level (L1–L3):

```
STEP 1: ANCHOR SELECTION
------------------------
Pick a seed word from internal concept banks.
  L1 anchors: fruit, bird, metal, tool, vehicle, dance, gem ...
  L2 anchors: fast, sad, run, destroy, brave, clever ...
  L3 anchors: fire, ocean, war, castle, forest, moon ...

        |
        v

STEP 2: DISAMBIGUATION (BabelNet)
---------------------------------
Send anchor to BabelNet getSenses()
  --> Filter: source=WN, language=EN, bKeySense=true
  --> Extract wordNetOffset (e.g., offset=07705931 for fruit)
  --> NLTK synset_from_pos_and_offset() --> "edible_fruit.n.01"

This ensures "fruit" maps to the FOOD sense, not any other.

        |
        v

STEP 3: EXPANSION (WordNet or Datamuse)
----------------------------------------
Use the correct synset to find candidate words:

  L1: WordNet BFS hyponyms (depth <= 3) of edible_fruit.n.01
      --> [apple, mango, grape, cherry, peach, plum, lime ...]

  L2: WordNet all_synonyms + Datamuse rel_syn for anchor
      --> [swift, rapid, fleet, brisk, quick, speedy ...]

  L3: Datamuse triggered_by(anchor) OR rel_bga(anchor)
      --> [smoke, ember, blaze, ash, heat, spark ...]

  L4: Hardcoded stem pool + Datamuse words_before(base)
      --> [foot, basket, base, snow, cannon, soft ...]

        |
        v

STEP 4: LINGUISTIC FILTERING (Brown Corpus)
--------------------------------------------
For every candidate word, check:
  - Is it purely alphabetic? (no hyphens, apostrophes)
  - Is its length between 3 and 10 characters?
  - Does it appear >= 8 times in the Brown Corpus?
  - Is it already used in another group?

Reject words failing any check.

        |
        v

STEP 5: SEMANTIC VALIDATION
----------------------------
L1: is_hyponym_of() — walk the hypernym path up from the
    candidate word and confirm the category synset appears
    on the path. Rejects words that sound like fruits but
    aren't (e.g., "passion" would be rejected).

L2: Shared synset check — the candidate must share at least
    one WordNet synset with the anchor word. Pure near-synonyms
    that share no synset are rejected.

L3: No structural validation — Datamuse association is the
    validation. Just frequency and exclusivity checks apply.

        |
        v

STEP 6: ROOT DEDUPLICATION
---------------------------
Remove morphological near-duplicates within the same group.
Examples: gun/guns, run/running, fire/fires/firing
The _root() function strips common suffixes (-ing, -ed, -ers,
-tion, -ly, -s, etc.) and rejects words sharing a root.

        |
        v

STEP 7: EXCLUSIVITY CHECK (Jaccard Index)
------------------------------------------
For L1, L2, L3: For each candidate word, compute Jaccard
overlap of its WordNet synsets against the words already
chosen for ALL OTHER groups.

  Jaccard(word, group) = |synsets(word) ∩ synsets(group_words)|
                         ----------------------------------------
                         |synsets(word) ∪ synsets(group_words)|

If Jaccard >= 0.25 with any other group --> REJECT the word.

This prevents a word like "rapid" (meaning fast) from also
appearing near synonym groups for "run" or "move."

NOTE: L4 is EXEMPT from this check. L4 compound stems are
ALLOWED to overlap with other groups — that intentional
ambiguity IS the Purple level's difficulty and misdirection.

        |
        v

STEP 8: PICK 4 WORDS
---------------------
From the validated, deduplicated, exclusive pool, randomly
pick exactly 4 words and form the group.

Repeat Steps 1-8 for each of L1, L2, L3, L4.

        |
        v

STEP 9: ASSEMBLY & SCORING
---------------------------
Combine 4 groups (16 words total).
Check no word appears in more than one group.
Shuffle the 16 words into a random 4x4 grid.
Compute quality score (0-100):
  - Word frequency score (40%)
  - Level completeness (30%)
  - Polysemy/richness (30%)

Output as Puzzle object + JSONL record.
```

---

## 7. Three Architectures — A, B, and C

The system cycles through three architectures to maximise vocabulary diversity and prevent the trained model from learning surface-level shortcuts.

### Architecture A — Standard
```
  L1: WordNet hyponym BFS (common categories: fruits, birds, metals)
       --> FRUITS / BIRDS / TOOLS / WEAPONS / VEHICLES ...

  L2: WordNet synonyms + Datamuse rel_syn, strict shared-synset filter
       --> WORDS MEANING FAST / RUN / SAD / DESTROY ...

  L3: Datamuse triggered_by(concrete anchor)
       anchor bank: fire, ocean, war, forest, castle ...
       --> ASSOCIATED WITH FIRE / OCEAN / STORM ...

  L4: Hardcoded stems + Datamuse words_before(base)
       --> ___ BALL / ___ LINE / ___ ROOM / ___ YARD ...

  Pattern: ___ BASE (word appears BEFORE the base)
```

### Architecture B — Contrast-Based
```
  L1: WordNet hyponym BFS — DIFFERENT category set
       categories: dances, snakes, fabrics, hats, boats, gems, cheeses ...
       --> DANCES / CHEESES / FABRICS / DOG BREEDS ...

  L2: Datamuse means_like (broader than strict synonyms)
       Accepts more distant semantic relatives than Architecture A
       --> WORDS MEANING LARGE / ANGRY / TIRED ...

  L3: Datamuse means_like(abstract anchor)
       anchor bank: justice, courage, chaos, wisdom, pride ...
       --> WORDS RELATED TO JUSTICE / COURAGE / FEAR ...

  L4: Hardcoded stems + Datamuse words_after(prefix)
       --> OVER ___ / UNDER ___ / BACK ___ / OUT ___ ...

  Pattern: BASE ___ (word appears AFTER the base)
```

### Architecture C — Meronym-Based
```
  L1: WordNet part_meronyms / substance_meronyms
       whole concepts: body, car, house, plant, bicycle, book ...
       --> BODY PARTS / CAR PARTS / PARTS OF A HOUSE / PLANT PARTS ...

  L2: WordNet synonyms + Datamuse rel_syn (same as Architecture A)
       --> WORDS MEANING SHOUT / HIT / HIDE ...

  L3: Datamuse rel_bga (corpus co-occurrence in bigrams)
       anchor bank: same as A (fire, ocean, etc.)
       --> GOES WITH OCEAN / CASTLE / MARKET ...

  L4: Randomly pick BEFORE or AFTER pattern per puzzle
       --> ___ STONE  or  SUN ___ or FOOT ___ ...

  Pattern: Mixed (randomly chosen each puzzle)
```

### Why Three Architectures?

| Dimension | Architecture A | Architecture B | Architecture C |
|-----------|---------------|---------------|---------------|
| L1 type | Hyponym (types of X) | Hyponym (alt categories) | Meronym (parts of X) |
| L2 method | Strict synset overlap | Broad means_like | Strict synset overlap |
| L3 method | triggered_by (concrete) | means_like (abstract) | Co-occurrence |
| L4 pattern | ___ BASE (prefix) | BASE ___ (suffix) | Mixed |
| Overfitting risk | Low | Lower | Lowest |

Running them in a cycle **A → B → C → A → B → C → ...** ensures each puzzle in the dataset looks structurally different, preventing the NLP model from learning "pattern A always looks like this."

---

## 8. Quality Controls — Why Each One Exists

### 8.1 BabelNet Disambiguation
**Problem:** "Fruit" could mean food or the band. "Snake" could mean reptile or the card game.  
**Fix:** BabelNet pinpoints the exact WordNet synset before any expansion begins.

### 8.2 Hyponym Path Validation (is_hyponym_of)
**Problem:** WordNet's BFS sometimes returns words that are in the same neighbourhood but aren't true hyponyms of the category root.  
**Fix:** Walk the `hypernym_paths()` of each candidate and confirm the category synset is on the path. A word passes only if it is a genuine "type of" the category.

### 8.3 Root Deduplication (_root function)
**Problem:** The same group might get "gun" and "guns", or "run" and "running" — morphologically the same word, reducing variety.  
**Fix:** Strip common suffixes (-ing, -ings, -ed, -ers, -tion, -ly, -s, etc.) and reject candidates sharing a root with an already-chosen word.

### 8.4 Forbidden Word Filter
**Problem:** The category word itself could appear in its own group (e.g., "fruit" in a FRUITS group, or "dance" in a DANCES group).  
**Fix:** Pass `forbidden = {concept, concept+"s", singular_form}` to the WordFilter so the anchor and its morphological variants are always excluded.

### 8.5 Exclusivity Check (Jaccard Index) — L1 to L3
**Problem:** A word might semantically fit two different groups, confusing both the puzzle and the model.  
**Fix:** Compute Jaccard synset overlap between the candidate and every already-chosen word in all other groups. Reject if overlap ≥ 0.25.

```
Why 0.25?
Jaccard = 0.0  means completely different synsets (ideal)
Jaccard = 0.25 means ~25% synset overlap (borderline ambiguous)
Jaccard = 1.0  means identical synsets (exact duplicate)

0.25 is the tuned threshold: strict enough to prevent
real ambiguity, lenient enough to not reject too many candidates.
```

### 8.6 No Exclusivity on L4 (Intentional)
**Problem:** L4 stems like "foot," "fire," "back," "hand" naturally overlap with other semantic groups because they are common English roots.  
**Fix:** Exempt L4 from the Jaccard check entirely. This overlap IS the design — it creates the misdirection that makes Purple the hardest level. A solver might think "fire" belongs in the ASSOCIATED WITH FIRE group, not realising it is a compound stem for "___ SIDE" or "CAMP___."

### 8.7 Global Used-Word Tracking
**Problem:** When generating a dataset of many puzzles, the same word could appear in multiple puzzles.  
**Fix:** Maintain a `global_used` set across all puzzle generations. Any word already used in a previous puzzle is excluded from all future puzzles in the same dataset run.

### 8.8 Quality Score (0–100)
**Problem:** No signal for filtering out poorly generated puzzles from the training set.  
**Fix:** Compute a weighted score:
```
  Word Frequency Score   = avg(Brown freq of all 16 words) / 300  × 40
  Level Completeness     = (distinct levels present / 4)          × 30
  Polysemy Richness      = avg(synset count per word) / 8         × 30
  ─────────────────────────────────────────────────────────────────────
  Total Quality Score    = sum of above  (0 – 100)
```
Higher frequency → more common words → more learnable patterns.  
Higher polysemy → richer semantic content → harder, more interesting puzzles.

---

## 9. The L4 Hardcoded Stem Pool — The Key Fix in v7

### The Problem in v6

In earlier versions, L4 used only Datamuse `words_before(base)` to find compound stems. This failed because:

- The `used` set from L1+L2+L3 often contained common English words like "fire," "back," "hand"
- These same common words are also valid compound stems (e.g., "campfire," "backyard," "handshake")
- So Datamuse would return them → the filter would reject them → L4 would fail with < 4 valid stems

### The Fix in v7

L4 now uses **pre-verified hardcoded stem lists** as its primary source, with Datamuse only as a supplement:

```python
L4_STEM_POOLS = {
    "ball":  ["basket", "base", "foot", "snow", "cannon", "soft", "pin", "volley", ...],
    "fire":  ["camp", "cross", "gun", "wild", "open", "rapid", "cease", "miss", ...],
    "light": ["day", "flash", "moon", "sun", "star", "lime", "spot", "search", ...],
    ...
}
```

Each entry in the pool is hand-verified to form a real compound word with the base (e.g., "basket" + "ball" = "basketball"). The hardcoded stems are deliberately chosen to **avoid common L1–L3 words**, and the frequency threshold for L4 is lowered to `freq >= 3` (vs `freq >= 8` for other levels) because compound word components can be less frequent on their own.

The merge order is: **hardcoded first, then Datamuse** — giving priority to the more reliable source.

---

## 10. Architecture Diagram (Text)

```
INPUT: concept anchor word
        |
        v
+-----------------------------------------------+
| STAGE 1: DISAMBIGUATION & ANCHORING           |
|                                               |
|  ANCHOR BANKS                                 |
|  L1: fruit, bird, tool, metal, dance ...      |
|  L2: fast, sad, run, destroy, brave ...       |
|  L3: fire, ocean, war, castle, forest ...     |
|         |                                     |
|         v                                     |
|  BabelNet getSenses(anchor)                   |
|    --> filter source=WN, lang=EN, bKey=true   |
|    --> extract wordNetOffset                  |
|    --> NLTK synset_from_pos_and_offset()      |
|    --> correct WordNet synset name            |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STAGE 2: SEMANTIC & THEMATIC EXPANSION        |
|                                               |
|  WordNet (NLTK, Offline)                      |
|    L1: Hyponym BFS (depth<=3)                 |
|         --> Candidate Pool (30–100 words)     |
|    L2: Synonym lookup + is_hyponym check      |
|                  |                            |
|                  v                            |
|  Datamuse API (Free REST)                     |
|    L2 supplement: rel_syn                     |
|    L3: triggered_by() OR rel_bga()            |
|    L4: words_before() / words_after()         |
|         --> Candidate Pool                    |
|                  |                            |
|                  v                            |
|  +------------------------------------------+ |
|  | L4 STEM POOLS (Hardcoded, v7 Key Fix)    | |
|  |  "___ BALL": basket, base, foot, snow    | |
|  |  "OVER ___": come, look, time, load ...  | |
|  +------------------------------------------+ |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STAGE 3: FILTERING & VALIDATION               |
|                                               |
|  WordFilter (Brown Corpus)                    |
|    - alpha-only, length 3–10 chars            |
|    - Brown Corpus freq >= 8 (>= 3 for L4)    |
|    - not already in `used` set               |
|                  |                            |
|  Semantic Validation                          |
|    L1: is_hyponym_of() hypernym path walk     |
|    L2: shared WordNet synset membership       |
|    L3: frequency + exclusivity only           |
|                  |                            |
|  Root Deduplicator                            |
|    _root() strips -ing/-ed/-s/-tion etc.      |
|    Rejects same-root pairs in same group      |
|                  |                            |
|  Exclusivity Engine (L1–L3 only)              |
|    Jaccard synset overlap < 0.25 threshold    |
|    vs all words in all other groups           |
|    (L4 is EXEMPT — misdirection is by design) |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STAGE 4: ASSEMBLY                             |
|                                               |
|  Pick 4 validated words per level             |
|  Add words to `used` set                      |
|  Repeat for L1 → L2 → L3 → L4               |
|  Check: no duplicates across 16 words         |
|  Shuffle grid randomly                        |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STAGE 5: SCORING & CYCLING                    |
|                                               |
|  Quality Score (0–100):                       |
|    freq(40%) + level_completeness(30%)        |
|    + polysemy(30%)                            |
|                                               |
|  Architecture Cycling:                        |
|    Puzzle 1 → Arch A                          |
|    Puzzle 2 → Arch B                          |
|    Puzzle 3 → Arch C                          |
|    Puzzle 4 → Arch A  (repeat)                |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| OUTPUT: JSONL Record                          |
|  {                                            |
|    "architecture": "A",                       |
|    "grid": [16 shuffled words],               |
|    "quality_score": 78.5,                     |
|    "groups": [4 groups with metadata],        |
|    "word_labels": [1,1,1,1,2,2,2,2,...],     |
|    "all_words": [16 words in label order]     |
|  }                                            |
+-----------------------------------------------+
        |
        v
  AUGMENTED CONNECTIONS DATASET
  (Synthetic Training Data)
```

---

## 11. Running the System

### Install Dependencies

```bash
pip install nltk requests
python -c "import nltk; [nltk.download(x, quiet=True) for x in ['wordnet','omw-1.4','brown']]"
```

### Get a Free BabelNet API Key

Sign up at [babelnet.org](https://babelnet.org) — free tier gives 50,000 queries/day.

### Generate Puzzles

```bash
# Generate 5 puzzles cycling through all 3 architectures (A→B→C→A→B)
python puzzle_generator.py --key YOUR_BABELNET_KEY --puzzles 5 --output dataset.jsonl

# Generate 10 puzzles using only Architecture A
python puzzle_generator.py --key YOUR_KEY --puzzles 10 --arch A --output arch_a.jsonl

# Generate 10 puzzles using only Architecture B
python puzzle_generator.py --key YOUR_KEY --puzzles 10 --arch B --output arch_b.jsonl

# Generate 10 puzzles using only Architecture C
python puzzle_generator.py --key YOUR_KEY --puzzles 10 --arch C --output arch_c.jsonl
```

### Expected Console Output

```
============================================================
  Puzzle 1/5  |  Architecture A
  A — hyponym + synonym + triggered_by + prefix_compound (___ WORD)
============================================================

  [gen-A] Attempt 1/12…
    ✓ L1 [FRUITS]: ['mango', 'cherry', 'grape', 'plum']
    ✓ L2 [WORDS MEANING FAST]: ['swift', 'brisk', 'fleet', 'rapid']
    ✓ L3 [ASSOCIATED WITH FIRE]: ['smoke', 'ember', 'blaze', 'ash']
      [L4] ✓ base='ball' mode=before chosen=['foot', 'basket', 'base', 'snow']
    ✓ L4 [___ BALL]: ['foot', 'basket', 'base', 'snow']
    ★ Score: 74.2/100  arch=A
```

---

## 12. Output Format (JSONL)

Each line of the output file is a JSON object:

```json
{
  "architecture": "A",
  "grid": ["mango", "foot", "swift", "smoke", "cherry", "basket", "brisk", "ember",
           "grape", "base", "fleet", "blaze", "plum", "snow", "rapid", "ash"],
  "quality_score": 74.2,
  "groups": [
    {
      "level": 1,
      "color": "yellow",
      "category": "FRUITS",
      "pattern": "wn_hyponym",
      "wn_synset": "edible_fruit.n.01",
      "words": [
        {"word": "mango",  "polysemy": 2, "frequency": 12},
        {"word": "cherry", "polysemy": 4, "frequency": 89},
        {"word": "grape",  "polysemy": 3, "frequency": 54},
        {"word": "plum",   "polysemy": 4, "frequency": 67}
      ]
    },
    {
      "level": 2,
      "color": "green",
      "category": "WORDS MEANING FAST",
      "pattern": "wn_synonym",
      "wn_synset": "fast.a.01",
      "words": [ ... ]
    },
    ...
  ],
  "all_words":   ["mango", "cherry", "grape", "plum", "swift", "brisk", ...],
  "word_labels": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
}
```

The `word_labels` field aligns with `all_words` (1=yellow, 2=green, 3=blue, 4=purple) and is the primary supervision signal for model training.

---

## 13. Design Decisions — The "Why" Behind Everything

| Decision | Why |
|----------|-----|
| **BabelNet for disambiguation** | WordNet alone has no disambiguation — querying "python" gives reptile AND language results. BabelNet's bKeySense filter reliably returns the dominant human-intended sense. |
| **Brown Corpus for frequency** | It is offline, reproducible, already in NLTK, and its frequency rankings correlate well with general vocabulary familiarity. A word present ≥8 times in a million-word sample is almost certainly in common use. |
| **Jaccard threshold of 0.25** | Empirically tuned: below this, group separation is clean; above this, words start being genuinely ambiguous between groups. |
| **L4 exempt from Jaccard** | L4's compound stems are SUPPOSED to feel like they belong to other groups. "Foot" looks like it belongs in BODY PARTS. "Fire" looks like it belongs in ASSOCIATED WITH FIRE. This misdirection IS the Purple level's design signature. |
| **Hardcoded L4 stem pools** | Datamuse alone was unreliable for L4 because `words_before()` returns context-sensitive neighbours that often coincide with L1–L3 words already in `used`. Hardcoded pools are human-verified to form real compounds and are checked once offline. |
| **Three architectures cycling** | A single pipeline would produce similar-looking puzzles (always fruits/birds for L1, always "WORDS MEANING X" for L2). Cycling architectures forces the dataset to cover meronyms, abstract associations, different compound patterns, and alternate category sets — reducing systematic bias. |
| **min_freq=3 for L4 vs min_freq=8 elsewhere** | Compound word components like "cannon" (in cannonball) or "gum" (in gumball) are legitimate English words but not particularly frequent on their own. The stricter threshold would reject too many valid stems. |
| **Root deduplication** | Without it, a group might contain "run," "running," and "runner" — three tokens for the same concept. This would both reduce the puzzle's vocabulary richness and teach the model that morphological variants are distinct concepts. |
| **global_used across puzzles** | In a training set, repeated words across puzzles could let a model shortcut by memorising "mango always means fruit" rather than learning the underlying semantic reasoning. |

---

*Generated by NYT Connections Puzzle Generator v7 — BabelNet + WordNet + Datamuse + Brown Corpus*
