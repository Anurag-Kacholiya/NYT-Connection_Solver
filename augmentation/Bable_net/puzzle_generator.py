"""
NYT Connections Puzzle Generator — v7
======================================

ARCHITECTURE OVERVIEW (3 variants, cycled for dataset variety):

Architecture A — Hyponym + Synonym + Triggered + Prefix Compound
  concept → BabelNet (disambiguate) → WordNet synset
    ↓
  WordNet hyponym BFS → validated members → L1 group [FRUITS, BIRDS, ...]
    ↓
  WordNet all_synonyms + Datamuse rel_syn → shared-synset filter → L2 group [WORDS MEANING FAST]
    ↓
  Datamuse triggered_by(anchor) → frequency filter → L3 group [ASSOCIATED WITH FIRE]
    ↓
  Datamuse words_before(base) → freq filter → L4 group [___ BALL]
    ↓
  Puzzle (16 words, 4 groups, quality score)

Architecture B — Hyponym + Means-Like + Abstract + Suffix Compound
  concept → BabelNet → WordNet synset
    ↓
  WordNet hyponym BFS (different category pool) → L1 group [DANCES, GEMS, DOG BREEDS]
    ↓
  Datamuse means_like(anchor) → Brown freq filter → L2 group [WORDS MEANING LARGE]
    ↓
  Datamuse means_like(abstract_concept) → freq filter → L3 group [WORDS RELATED TO JUSTICE]
    ↓
  Datamuse words_after(prefix) → freq filter → L4 group [OVER ___]
    ↓
  Puzzle (broader semantic connections, different structural patterns)

Architecture C — Meronym + Synonym + Co-occurrence + Mixed Compound
  whole_concept → WordNet part_meronyms/substance_meronyms
    ↓
  Validated parts → L1 group [BODY PARTS, CAR PARTS, PARTS OF A HOUSE]
    ↓
  WordNet all_synonyms + Datamuse rel_syn → shared-synset → L2 group
    ↓
  Datamuse rel_bga= co-occurrence(anchor) → freq filter → L3 group [GOES WITH OCEAN]
    ↓
  Randomly choose words_before OR words_after → L4 group [___ STONE or BACK ___]
    ↓
  Puzzle (part-based L1 structure, corpus-co-occurrence L3)

KEY FIX v7:
  L4 was failing because `used` set (from L1+L2+L3) contained common stems
  like 'fire', 'back', 'hand' that also happen to be valid compound stems.
  Fix: L4 uses HARDCODED verified stem lists as primary source, Datamuse only
  as supplement. Hardcoded stems are pre-verified to not clash with most L1-L3 words.
  Also: L4 min_freq threshold lowered to 3, length extended to 3-12 chars.
"""

import argparse
import json
import random
import time
import requests
import re
import warnings
import contextlib
import io
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import nltk
with contextlib.redirect_stdout(io.StringIO()):
    nltk.download("wordnet",  quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("brown",   quiet=True)
from nltk.corpus import wordnet as wn, brown

# ─────────────────────────────────────────────────────────────
# Brown corpus frequency
# ─────────────────────────────────────────────────────────────
_FREQ: Optional[Counter] = None


def _freq_table() -> Counter:
    global _FREQ
    if _FREQ is None:
        print("  [init] Loading Brown corpus…")
        _FREQ = Counter(w.lower() for w in brown.words() if w.isalpha())
    return _FREQ


def word_freq(w: str) -> int:
    return _freq_table().get(w.lower(), 0)

# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────


@dataclass
class Word:
    text: str
    polysemy: int = 0
    frequency: int = 0


@dataclass
class Group:
    level: int
    name: str
    words: list = field(default_factory=list)
    pattern: str = ""
    wn_synset: str = ""


@dataclass
class Puzzle:
    groups: list = field(default_factory=list)
    grid: list = field(default_factory=list)
    score: float = 0.0
    arch: str = "A"


# ─────────────────────────────────────────────────────────────
# HARDCODED L4 STEM POOLS
# Each base → list of valid stems that form real compound words.
# These are pre-verified so L4 doesn't depend entirely on Datamuse
# returning stems that happen to not be in the `used` set.
# ─────────────────────────────────────────────────────────────
L4_STEM_POOLS = {
    # ___ BASE patterns
    "ball":   ["basket", "base", "foot", "snow", "cannon", "soft", "pin", "volley",
               "over", "hand", "net", "cricket", "stump", "gum", "odd"],
    "fire":   ["camp", "cross", "gun", "wild", "open", "rapid", "cease", "miss",
               "back", "spear", "sword", "anti"],
    "light":  ["day", "flash", "moon", "sun", "star", "lime", "spot", "search",
               "candle", "street", "tail", "head", "night", "green", "sky"],
    "house":  ["farm", "store", "ware", "court", "power", "jail", "round",
               "green", "club", "town", "work", "ale", "road", "tree"],
    "board":  ["card", "key", "skate", "surf", "snow", "over", "dart", "paste",
               "black", "cup", "flap", "score", "side", "check", "base"],
    "line":   ["base", "dead", "head", "side", "hair", "border", "life", "tag",
               "border", "clothes", "guide", "shore", "story", "under", "hot"],
    "yard":   ["back", "court", "vine", "grave", "barn", "door", "farm", "ship",
               "stock", "school", "church", "front"],
    "field":  ["corn", "mine", "air", "oil", "coal", "wheat", "out", "grass",
               "down", "play", "gold", "hay", "battle", "open", "ice"],
    "book":   ["hand", "note", "text", "pass", "case", "guide", "log", "cook",
               "copy", "guest", "pay", "rule", "score", "source", "work"],
    "stone":  ["lime", "sand", "key", "flag", "corner", "cobble", "grind",
               "hail", "mud", "sand", "brown", "head", "mill"],
    "room":   ["bed", "bath", "court", "show", "class", "ball", "store",
               "cloak", "dark", "dining", "dress", "gun", "living", "mud", "work"],
    "work":   ["frame", "net", "paper", "team", "ground", "field", "over",
               "house", "home", "class", "body", "social", "wood", "bridge"],
    "time":   ["day", "night", "over", "life", "half", "lunch", "any", "bed",
               "down", "full", "hard", "long", "old", "part", "peace"],
    "man":    ["fire", "door", "sales", "states", "gentle", "work", "play",
               "post", "horse", "marks", "gun", "chair", "service", "police"],
    "day":    ["birth", "every", "holi", "today", "week", "work", "pay",
               "mid", "birth", "week", "good", "sun"],
    "cake":   ["cup", "cheese", "short", "pan", "wafer"],
    "box":    ["sand", "tool", "bread", "music", "card", "post", "mail"],
    "fall":   ["down", "free", "water", "land", "over", "short", "night", "rain"],
    "side":   ["bed", "out", "in", "road", "lake", "hill", "sea", "ring", "court",
               "fire", "top", "water", "blind", "broad", "dark"],
    # BASE ___ patterns (prefix)
    "over":   ["come", "look", "time", "load", "turn", "all", "board", "cast",
               "coat", "due", "flow", "lap", "night", "pass", "rule", "see",
               "take", "throw", "work"],
    "under":  ["line", "cover", "go", "age", "cut", "dog", "done", "foot",
               "ground", "hand", "mine", "pass", "play", "score", "side",
               "take", "tone", "world"],
    "out":    ["run", "side", "door", "fit", "law", "let", "line", "post",
               "put", "reach", "right", "set", "skirts", "smart", "source",
               "stand", "weigh"],
    "back":   ["ground", "yard", "fire", "lash", "log", "pack", "seat", "slide",
               "stage", "stroke", "track", "ward", "bone", "door", "drop", "handed"],
    "down":   ["town", "stairs", "fall", "hill", "load", "right", "side",
               "size", "stairs", "stream", "swing", "town", "ward", "play"],
    "eye":    ["lid", "brow", "lash", "ball", "sight", "witness", "brow",
               "drop", "hole", "let", "piece"],
    "foot":   ["print", "note", "step", "ball", "hold", "wear", "bridge",
               "path", "rest", "stool", "work"],
    "hand":   ["shake", "bag", "made", "rail", "writing", "cuff", "hold",
               "maiden", "out", "picked", "stand", "work"],
    "head":   ["ache", "band", "line", "master", "quarters", "stone",
               "board", "count", "dress", "gear", "lamp", "land", "long",
               "phone", "set", "strong"],
    "sun":    ["burn", "flower", "glasses", "light", "rise", "set", "shine",
               "screen", "stroke"],
}

# Pattern metadata: (base, display_label, mode)
L4_PATTERNS = [
    ("ball",  "___ BALL",   "before"),
    ("fire",  "___ FIRE",   "before"),
    ("light", "___ LIGHT",  "before"),
    ("house", "___ HOUSE",  "before"),
    ("board", "___ BOARD",  "before"),
    ("line",  "___ LINE",   "before"),
    ("yard",  "___ YARD",   "before"),
    ("field", "___ FIELD",  "before"),
    ("book",  "___ BOOK",   "before"),
    ("stone", "___ STONE",  "before"),
    ("room",  "___ ROOM",   "before"),
    ("work",  "___ WORK",   "before"),
    ("time",  "___ TIME",   "before"),
    ("man",   "___ MAN",    "before"),
    ("day",   "___ DAY",    "before"),
    ("side",  "___ SIDE",   "before"),
    ("fall",  "___ FALL",   "before"),
    ("box",   "___ BOX",    "before"),
    ("over",  "OVER ___",   "after"),
    ("under", "UNDER ___",  "after"),
    ("out",   "OUT ___",    "after"),
    ("back",  "BACK ___",   "after"),
    ("down",  "DOWN ___",   "after"),
    ("eye",   "EYE ___",    "after"),
    ("foot",  "FOOT ___",   "after"),
    ("hand",  "HAND ___",   "after"),
    ("head",  "HEAD ___",   "after"),
    ("sun",   "SUN ___",    "after"),
]

# ─────────────────────────────────────────────────────────────
# BabelNet — synset disambiguation only
# ─────────────────────────────────────────────────────────────


class BabelNetDisambiguator:
    BASE = "https://babelnet.io/v9"

    def __init__(self, key: str):
        self.key = key
        self._cache = {}

    def _get(self, ep: str, params: dict):
        p = dict(params)
        p["key"] = self.key
        url = f"{self.BASE}/{ep}?" + "&".join(f"{k}={v}" for k, v in p.items())
        if url in self._cache:
            return self._cache[url]
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 429:
                print("  [BabelNet] rate limit — sleeping 5s")
                time.sleep(5)
                r = requests.get(url, timeout=12)
            r.raise_for_status()
            data = r.json()
            self._cache[url] = data
            time.sleep(0.12)
            return data
        except Exception as e:
            print(f"  [BabelNet] {ep}: {e}")
            return []

    def find_wn_synset(self, word: str, pos: str = "n",
                       preferred_domains: list = None) -> Optional[str]:
        wn_pos = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ}.get(pos, wn.NOUN)
        data = self._get(
            "getSenses", {"lemma": word, "searchLang": "EN", "targetLang": "EN"})
        candidates = []
        if isinstance(data, list):
            for s in data:
                p = s.get("properties", {})
                if p.get("source") != "WN" or p.get("language") != "EN":
                    continue
                digits = re.sub(r"[^0-9]", "", p.get("wordNetOffset", ""))
                if not digits:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        syn = wn.synset_from_pos_and_offset(
                            wn_pos, int(digits))
                    candidates.append((syn, p.get("bKeySense", False)))
                except Exception:
                    continue
        if not candidates:
            syns = wn.synsets(word, pos=wn_pos)
            if not syns:
                return None
            if preferred_domains:
                for s in syns:
                    if any(d in s.lexname() for d in preferred_domains):
                        return s.name()
            return syns[0].name()
        key_pool = [s for s, k in candidates if k] or [
            s for s, _ in candidates]
        if preferred_domains:
            for s in key_pool:
                if any(d in s.lexname() for d in preferred_domains):
                    return s.name()
        return key_pool[0].name()

    def probe(self):
        print("  [probe] BabelNet API check…")
        data = self._get(
            "getSenses", {"lemma": "apple", "searchLang": "EN", "targetLang": "EN"})
        ok = isinstance(data, list) and len(data) > 0
        syn = self.find_wn_synset(
            "apple", "n", ["noun.food"]) if ok else "FAILED"
        print(f"  [probe] {'✓' if ok else '✗'}  apple → {syn}\n")


# ─────────────────────────────────────────────────────────────
# WordNet Miner
# ─────────────────────────────────────────────────────────────
class WordNetMiner:
    CATEGORIES = {
        "fruit":       ("edible_fruit.n.01",       "noun.food",       "FRUITS"),
        "bird":        ("bird.n.01",               "noun.animal",     "BIRDS"),
        "fish":        ("fish.n.01",               "noun.animal",     "FISH"),
        "insect":      ("insect.n.01",             "noun.animal",     "INSECTS"),
        "mammal":      ("mammal.n.01",             "noun.animal",     "MAMMALS"),
        "vegetable":   ("vegetable.n.01",          "noun.food",       "VEGETABLES"),
        "tree":        ("tree.n.01",               "noun.plant",      "TREES"),
        "flower":      ("flower.n.01",             "noun.plant",      "FLOWERS"),
        "tool":        ("tool.n.01",               "noun.artifact",   "TOOLS"),
        "weapon":      ("weapon.n.01",             "noun.artifact",   "WEAPONS"),
        "vehicle":     ("wheeled_vehicle.n.01",    "noun.artifact",   "VEHICLES"),
        "furniture":   ("furniture.n.01",          "noun.artifact",   "FURNITURE"),
        "sport":       ("sport.n.01",              "noun.act",        "SPORTS"),
        "cheese":      ("cheese.n.01",             "noun.food",       "CHEESES"),
        "gem":         ("precious_stone.n.01",     "noun.substance",  "GEMS"),
        "metal":       ("metallic_element.n.01",   "noun.substance",  "METALS"),
        "currency":    ("currency.n.01",           "noun.possession", "CURRENCIES"),
        "dog":         ("dog.n.01",                "noun.animal",     "DOG BREEDS"),
        "planet":      ("planet.n.01",             "noun.object",     "PLANETS"),
        "instrument":  ("musical_instrument.n.01", "noun.artifact",   "INSTRUMENTS"),
        "dance":       ("dance.n.01",              "noun.act",        "DANCES"),
        "snake":       ("snake.n.01",              "noun.animal",     "SNAKES"),
        "fabric":      ("fabric.n.01",             "noun.artifact",   "FABRICS"),
        "hat":         ("hat.n.01",                "noun.artifact",   "HATS"),
        "boat":        ("boat.n.01",               "noun.artifact",   "BOATS"),
    }
    MERONYM_CATS = {
        "body":    ("body.n.01",    "BODY PARTS"),
        "car":     ("car.n.01",     "CAR PARTS"),
        "house":   ("house.n.01",   "PARTS OF A HOUSE"),
        "plant":   ("plant.n.02",   "PLANT PARTS"),
        "bicycle": ("bicycle.n.01", "BICYCLE PARTS"),
        "face":    ("face.n.01",    "PARTS OF THE FACE"),
        "book_n":  ("book.n.01",    "PARTS OF A BOOK"),
        "tree_n":  ("tree.n.01",    "PARTS OF A TREE"),
    }

    def hyponyms_of(self, synset_name: str, max_depth: int = 3, max_results: int = 100) -> list[str]:
        try:
            root = wn.synset(synset_name)
        except Exception:
            return []
        visited, queue, words = set(), [(root, 0)], []
        while queue and len(words) < max_results:
            node, d = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for lem in node.lemmas():
                tok = lem.name().replace("_", "")
                if tok.isalpha() and 3 <= len(tok) <= 10:
                    words.append(tok.lower())
            if d < max_depth:
                for h in node.hyponyms():
                    if h not in visited:
                        queue.append((h, d+1))
        return list(dict.fromkeys(words))

    def meronyms_of(self, synset_name: str) -> list[str]:
        try:
            root = wn.synset(synset_name)
        except Exception:
            return []
        words = []
        for rel in [root.part_meronyms(), root.substance_meronyms(), root.member_meronyms()]:
            for h in rel:
                for lem in h.lemmas():
                    tok = lem.name().replace("_", "")
                    if tok.isalpha() and 3 <= len(tok) <= 10:
                        words.append(tok.lower())
        return list(dict.fromkeys(words))

    def is_hyponym_of(self, word: str, synset_name: str) -> bool:
        try:
            cat = wn.synset(synset_name)
        except Exception:
            return True
        for syn in wn.synsets(word):
            for path in syn.hypernym_paths():
                if cat in path:
                    return True
        return False

    def all_synonyms(self, concept: str, pos) -> list[str]:
        out = set()
        for syn in wn.synsets(concept, pos=pos):
            for lem in syn.lemmas():
                w = lem.name().replace("_", "")
                if w.isalpha() and 3 <= len(w) <= 10 and w.lower() != concept.lower():
                    out.add(w.lower())
        return list(out)

    def polysemy(self, w: str) -> int: return len(wn.synsets(w))

    def synset_overlap(self, w1: str, w2: str) -> float:
        s1, s2 = set(wn.synsets(w1)), set(wn.synsets(w2))
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def max_overlap(self, word: str, group: list) -> float:
        if not group:
            return 0.0
        return max(self.synset_overlap(word, g) for g in group)


# ─────────────────────────────────────────────────────────────
# Datamuse
# ─────────────────────────────────────────────────────────────
class DatamuseMiner:
    BASE = "https://api.datamuse.com"
    _cache = {}

    def _get(self, params: dict) -> list:
        key = str(sorted(params.items()))
        if key in self._cache:
            return self._cache[key]
        try:
            r = requests.get(f"{self.BASE}/words", params=params, timeout=8)
            r.raise_for_status()
            self._cache[key] = r.json()
            return self._cache[key]
        except Exception as e:
            print(f"  [Datamuse] {e}")
            return []

    def _clean(self, res: list, exc: str = "") -> list[str]:
        out = []
        for item in res:
            w = item.get("word", "")
            if w and " " not in w and w.isalpha() and 3 <= len(w) <= 10 and w.lower() != exc.lower():
                out.append(w.lower())
        return out

    def triggered_by(self, w: str, n: int = 60) -> list[str]:
        return self._clean(self._get({"rel_trg": w, "max": n}), w)

    def means_like(self, w: str, n: int = 60) -> list[str]:
        return self._clean(self._get({"ml": w, "max": n}), w)

    def synonyms(self, w: str, n: int = 40) -> list[str]:
        return self._clean(self._get({"rel_syn": w, "max": n}), w)

    def words_before(self, suffix: str, n: int = 80) -> list[str]:
        return self._clean(self._get({"rc": suffix, "max": n}), suffix)

    def words_after(self, prefix: str, n: int = 80) -> list[str]:
        return self._clean(self._get({"lc": prefix, "max": n}), prefix)

    def cooccur(self, w: str, n: int = 60) -> list[str]:
        return self._clean(self._get({"rel_bga": w, "max": n}), w)


# ─────────────────────────────────────────────────────────────
# Word Filter + dedup helpers
# ─────────────────────────────────────────────────────────────
class WordFilter:
    def __init__(self, min_freq: int = 8):
        self.min_freq = min_freq

    def ok(self, w: str, used: set = None, forbidden: set = None) -> bool:
        wl = w.strip().lower()
        if not (wl.isalpha() and 3 <= len(wl) <= 10):
            return False
        if word_freq(wl) < self.min_freq:
            return False
        if used and wl in used:
            return False
        if forbidden and wl in forbidden:
            return False
        return True

    def apply(self, words, used=None, forbidden=None) -> list[str]:
        seen, out = set(), []
        for w in words:
            wl = w.lower()
            if wl not in seen and self.ok(wl, used, forbidden):
                seen.add(wl)
                out.append(wl)
        return out


def _root(w: str) -> str:
    for suf in ("ings", "ing", "tion", "ers", "ies", "es", "ed", "er", "ly", "s"):
        if w.endswith(suf) and len(w)-len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _dedup(words: list) -> list:
    seen, out = set(), []
    for w in words:
        r = _root(w.lower())
        if r not in seen:
            seen.add(r)
            out.append(w)
    return out


# ─────────────────────────────────────────────────────────────
# L2 concept bank (anchor → synonyms)
# ─────────────────────────────────────────────────────────────
L2_CONCEPTS = [
    ("run",       wn.VERB, "run.v.01",       "WORDS MEANING RUN"),
    ("steal",     wn.VERB, "steal.v.01",     "WORDS MEANING STEAL"),
    ("destroy",   wn.VERB, "destroy.v.01",   "WORDS MEANING DESTROY"),
    ("deceive",   wn.VERB, "deceive.v.01",   "WORDS MEANING DECEIVE"),
    ("walk",      wn.VERB, "walk.v.01",      "WORDS MEANING WALK"),
    ("increase",  wn.VERB, "increase.v.01",  "WORDS MEANING INCREASE"),
    ("shout",     wn.VERB, "shout.v.01",     "WORDS MEANING SHOUT"),
    ("fix",       wn.VERB, "repair.v.01",    "WORDS MEANING FIX"),
    ("speak",     wn.VERB, "talk.v.02",      "WORDS MEANING SPEAK"),
    ("help",      wn.VERB, "help.v.01",      "WORDS MEANING HELP"),
    ("laugh",     wn.VERB, "laugh.v.01",     "WORDS MEANING LAUGH"),
    ("hide",      wn.VERB, "hide.v.01",      "WORDS MEANING HIDE"),
    ("hit",       wn.VERB, "hit.v.01",       "WORDS MEANING HIT"),
    ("large",     wn.ADJ,  "large.a.01",     "WORDS MEANING LARGE"),
    ("fast",      wn.ADJ,  "fast.a.01",      "WORDS MEANING FAST"),
    ("angry",     wn.ADJ,  "angry.a.01",     "WORDS MEANING ANGRY"),
    ("sad",       wn.ADJ,  "sad.a.01",       "WORDS MEANING SAD"),
    ("happy",     wn.ADJ,  "happy.a.01",     "WORDS MEANING HAPPY"),
    ("brave",     wn.ADJ,  "brave.a.01",     "WORDS MEANING BRAVE"),
    ("clever",    wn.ADJ,  "clever.a.01",    "WORDS MEANING CLEVER"),
    ("strange",   wn.ADJ,  "strange.a.01",   "WORDS MEANING STRANGE"),
    ("small",     wn.ADJ,  "small.a.01",     "WORDS MEANING SMALL"),
    ("tired",     wn.ADJ,  "tired.a.01",     "WORDS MEANING TIRED"),
    ("beautiful", wn.ADJ,  "beautiful.a.01", "WORDS MEANING BEAUTIFUL"),
    ("old",       wn.ADJ,  "old.a.01",       "WORDS MEANING OLD"),
    ("dark",      wn.ADJ,  "dark.a.01",      "WORDS MEANING DARK"),
    ("rich",      wn.ADJ,  "rich.a.01",      "WORDS MEANING RICH"),
    ("weak",      wn.ADJ,  "weak.a.01",      "WORDS MEANING WEAK"),
]

L3_ANCHORS = [
    "fire", "water", "ice", "gold", "silver", "ocean", "desert",
    "forest", "space", "music", "war", "peace", "light", "wind",
    "heat", "snow", "night", "blood", "steel", "stone", "storm",
    "dream", "time", "rain", "sun", "moon", "earth", "cave",
    "river", "mountain", "castle", "pirate", "circus", "magic",
    "prison", "garden", "hospital", "kitchen", "wedding", "market",
]

L3_ABSTRACT = [
    "justice", "liberty", "chaos", "memory", "courage", "wealth",
    "danger", "beauty", "wisdom", "truth", "power", "glory",
    "honor", "shame", "pride", "faith", "hope", "fear", "love",
    "anger", "grief", "joy", "peace", "war", "time", "fate",
]


# ─────────────────────────────────────────────────────────────
# Core level-building helpers (shared across architectures)
# ─────────────────────────────────────────────────────────────
def _w(text, wn_m): return Word(text, wn_m.polysemy(text), word_freq(text))


def _excl_ok(word: str, others: list, wn_m: WordNetMiner, thr: float = 0.25) -> bool:
    for g in others:
        if wn_m.max_overlap(word, g) >= thr:
            return False
    return True


def _build_l1_hyponym(concept, cat_data, bn, wn_m, flt, used, others):
    syn_default, dom, display = cat_data
    synset = bn.find_wn_synset(concept, "n", [dom]) or syn_default
    raw = wn_m.hyponyms_of(synset)
    forbidden = {concept, concept+"s",
                 concept[:-1] if concept.endswith("s") else concept+"s"}
    validated = [w for w in flt.apply(
        raw, used, forbidden) if wn_m.is_hyponym_of(w, synset)]
    clean = _dedup([w for w in validated if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(1, display, [_w(c, wn_m) for c in clean[:4]], "wn_hyponym", synset)


def _build_l2_synonym(anchor, pos, syn_hint, label, dm, wn_m, flt, used, others):
    wn_syns = wn_m.all_synonyms(anchor, pos)
    dm_syns = dm.synonyms(anchor) + dm.means_like(anchor)
    pool = list(dict.fromkeys(wn_syns + dm_syns))
    anchor_synsets = set(wn.synsets(anchor, pos=pos))
    validated = [w for w in pool if set(wn.synsets(w)) & anchor_synsets]
    forbidden = {anchor, anchor+"s", anchor+"ing", anchor+"ed", anchor+"er"}
    cands = flt.apply(validated, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(2, label, [_w(c, wn_m) for c in clean[:4]], "wn_synonym", syn_hint)


def _build_l2_meanslike(anchor, pos, syn_hint, label, dm, wn_m, flt, used, others):
    pool = dm.means_like(anchor)
    forbidden = {anchor, anchor+"s", anchor+"ing", anchor+"ed"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(2, label, [_w(c, wn_m) for c in clean[:4]], "datamuse_means_like", syn_hint)


def _build_l3_triggered(anchor, dm, wn_m, flt, used, others):
    pool = dm.triggered_by(anchor)
    forbidden = {anchor, anchor+"s", anchor+"ed", anchor+"ing"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(3, f"ASSOCIATED WITH {anchor.upper()}",
                 [_w(c, wn_m) for c in clean[:4]], "datamuse_triggered")


def _build_l3_cooccur(anchor, dm, wn_m, flt, used, others):
    pool = dm.cooccur(anchor) or dm.triggered_by(anchor)
    forbidden = {anchor, anchor+"s"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(3, f"GOES WITH {anchor.upper()}",
                 [_w(c, wn_m) for c in clean[:4]], "datamuse_cooccur")


def _build_l3_abstract(anchor, dm, wn_m, flt, used, others):
    pool = dm.means_like(anchor)
    forbidden = {anchor, anchor+"s"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(3, f"WORDS RELATED TO {anchor.upper()}",
                 [_w(c, wn_m) for c in clean[:4]], "datamuse_abstract_ml")


def _build_l4(mode_filter, dm, wn_m, used, min_non_conflict=4):
    """
    L4 builder — uses HARDCODED stem pools as primary source,
    Datamuse as supplement. Only checks `used` (no exclusivity).
    mode_filter: "before", "after", or "any"
    """
    patterns = [(b, d, m) for b, d, m in L4_PATTERNS
                if mode_filter == "any" or m == mode_filter]
    random.shuffle(patterns)

    for base, display, mode in patterns:
        # Primary: hardcoded stems (verified real compounds)
        hardcoded = L4_STEM_POOLS.get(base, [])

        # Supplement: Datamuse live results
        try:
            live = (dm.words_before(base) if mode == "before"
                    else dm.words_after(base))
        except Exception:
            live = []

        # Merge: hardcoded first (more reliable), then live
        all_stems = list(dict.fromkeys(hardcoded + live))

        # Filter: alpha, 3-12 chars, freq >= 3, NOT in used, not the base itself
        # Deliberately lenient: L4 is misdirection by design
        cands = []
        for s in all_stems:
            sl = s.lower()
            if (sl.isalpha() and 3 <= len(sl) <= 12
                    and word_freq(sl) >= 3
                    and sl not in used
                    and sl != base):
                cands.append(sl)

        cands = _dedup(list(dict.fromkeys(cands)))

        if len(cands) < 4:
            print(
                f"      [L4] '{base}' only {len(cands)} stems after filter, skipping")
            continue

        random.shuffle(cands)
        chosen = cands[:4]
        print(f"      [L4] ✓ base='{base}' mode={mode} chosen={chosen}")
        return Group(4, display, [_w(c, wn_m) for c in chosen],
                     f"compound_{mode}")
    return None


# ─────────────────────────────────────────────────────────────
# Architecture A — Standard
# ─────────────────────────────────────────────────────────────
class ArchitectureA:
    """
    PIPELINE:
    concept → BabelNet getSenses → WordNet offset → NLTK synset
        ↓
    WordNet hyponym BFS (depth 3) → is_hyponym_of validation
        ↓ [L1: FRUITS / BIRDS / METALS / ...]
    WordNet all_synonyms + Datamuse rel_syn → shared-synset filter
        ↓ [L2: WORDS MEANING FAST / RUN / SAD / ...]
    Datamuse triggered_by(anchor) → Brown freq filter
        ↓ [L3: ASSOCIATED WITH FIRE / OCEAN / WAR / ...]
    Hardcoded stems + Datamuse words_before(base) → freq filter
        ↓ [L4: ___ BALL / ___ LINE / ___ ROOM / ...]
    Assemble → shuffle → Puzzle
    """
    NAME = "A — hyponym + synonym + triggered_by + prefix_compound (___ WORD)"

    def __init__(self, bn, wn_m, dm, flt):
        self.bn = bn
        self.wn = wn_m
        self.dm = dm
        self.flt = flt

    def level1(self, used, others):
        cats = list(self.wn.CATEGORIES.items())
        random.shuffle(cats)
        for concept, data in cats:
            g = _build_l1_hyponym(concept, data, self.bn,
                                  self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level2(self, used, others):
        concepts = random.sample(L2_CONCEPTS, len(L2_CONCEPTS))
        for anchor, pos, syn, label in concepts:
            g = _build_l2_synonym(anchor, pos, syn, label,
                                  self.dm, self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level3(self, used, others):
        anchors = random.sample(L3_ANCHORS, len(L3_ANCHORS))
        for anchor in anchors:
            g = _build_l3_triggered(
                anchor, self.dm, self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level4(self, used):
        return _build_l4("before", self.dm, self.wn, used)


# ─────────────────────────────────────────────────────────────
# Architecture B — Contrast (different categories + means_like + suffix)
# ─────────────────────────────────────────────────────────────
class ArchitectureB:
    """
    PIPELINE:
    concept → BabelNet getSenses → WordNet offset → NLTK synset
        ↓
    WordNet hyponym BFS — DIFFERENT category pool (dances/gems/dogs/hats/boats)
        ↓ [L1: DANCES / GEMS / DOG BREEDS / HATS / CHEESES / ...]
    Datamuse means_like(anchor) → Brown freq filter (BROADER than exact synonyms)
        ↓ [L2: WORDS MEANING LARGE / ANGRY / ...]
    Datamuse means_like(abstract) → Brown freq filter
        ↓ [L3: WORDS RELATED TO JUSTICE / COURAGE / FEAR / ...]
    Hardcoded stems + Datamuse words_after(prefix) → freq filter
        ↓ [L4: OVER ___ / UNDER ___ / BACK ___ / OUT ___ / ...]
    Assemble → shuffle → Puzzle
    """
    NAME = "B — hyponym(alt cats) + means_like + abstract_ml + suffix_compound (WORD ___)"

    B_CATS = ["dance", "snake", "fabric", "hat", "boat", "planet",
              "cheese", "dog", "gem", "flower", "insect", "fish"]

    def __init__(self, bn, wn_m, dm, flt):
        self.bn = bn
        self.wn = wn_m
        self.dm = dm
        self.flt = flt

    def level1(self, used, others):
        # Try B-specific categories first
        b_cats = [(k, self.wn.CATEGORIES[k])
                  for k in self.B_CATS if k in self.wn.CATEGORIES]
        random.shuffle(b_cats)
        for concept, data in b_cats:
            g = _build_l1_hyponym(concept, data, self.bn,
                                  self.wn, self.flt, used, others)
            if g:
                return g
        # Fallback to any category
        all_cats = list(self.wn.CATEGORIES.items())
        random.shuffle(all_cats)
        for concept, data in all_cats:
            g = _build_l1_hyponym(concept, data, self.bn,
                                  self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level2(self, used, others):
        concepts = random.sample(L2_CONCEPTS, len(L2_CONCEPTS))
        for anchor, pos, syn, label in concepts:
            g = _build_l2_meanslike(
                anchor, pos, syn, label, self.dm, self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level3(self, used, others):
        anchors = random.sample(L3_ABSTRACT, len(L3_ABSTRACT))
        for anchor in anchors:
            g = _build_l3_abstract(
                anchor, self.dm, self.wn, self.flt, used, others)
            if g:
                return g
        # Fallback to triggered
        anchors2 = random.sample(L3_ANCHORS, len(L3_ANCHORS))
        for anchor in anchors2:
            g = _build_l3_triggered(
                anchor, self.dm, self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level4(self, used):
        return _build_l4("after", self.dm, self.wn, used)


# ─────────────────────────────────────────────────────────────
# Architecture C — Meronym (parts-of-whole + cooccurrence)
# ─────────────────────────────────────────────────────────────
class ArchitectureC:
    """
    PIPELINE:
    whole_concept → WordNet part_meronyms / substance_meronyms
        ↓ [L1: BODY PARTS / CAR PARTS / PARTS OF A HOUSE / ...]
    WordNet all_synonyms + Datamuse rel_syn → shared-synset filter
        ↓ [L2: WORDS MEANING SHOUT / HIDE / HIT / ...]
    Datamuse rel_bga= corpus co-occurrence(anchor)
        ↓ [L3: GOES WITH OCEAN / CASTLE / MARKET / ...]
    Hardcoded stems + Datamuse (randomly before OR after) → freq filter
        ↓ [L4: ___ STONE or SUN ___ or FOOT ___ — random per puzzle]
    Assemble → shuffle → Puzzle
    """
    NAME = "C — meronym(parts) + synonym + cooccur + mixed_compound (random)"

    def __init__(self, bn, wn_m, dm, flt):
        self.bn = bn
        self.wn = wn_m
        self.dm = dm
        self.flt = flt

    def level1(self, used, others):
        cats = list(self.wn.MERONYM_CATS.items())
        random.shuffle(cats)
        for concept, (syn, display) in cats:
            raw = self.wn.meronyms_of(syn)
            forbidden = {concept, concept+"s"}
            cands = self.flt.apply(raw, used, forbidden)
            clean = _dedup([w for w in cands if _excl_ok(w, others, self.wn)])
            if len(clean) < 4:
                continue
            random.shuffle(clean)
            return Group(1, display, [_w(c, self.wn) for c in clean[:4]], "wn_meronym", syn)
        # Fallback to hyponym
        all_cats = list(self.wn.CATEGORIES.items())
        random.shuffle(all_cats)
        for concept, data in all_cats:
            g = _build_l1_hyponym(concept, data, self.bn,
                                  self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level2(self, used, others):
        concepts = random.sample(L2_CONCEPTS, len(L2_CONCEPTS))
        for anchor, pos, syn, label in concepts:
            g = _build_l2_synonym(anchor, pos, syn, label,
                                  self.dm, self.wn, self.flt, used, others)
            if g:
                return g
        return None

    def level3(self, used, others):
        anchors = random.sample(L3_ANCHORS, len(L3_ANCHORS))
        for anchor in anchors:
            g = _build_l3_cooccur(anchor, self.dm, self.wn,
                                  self.flt, used, others)
            if g:
                return g
        return None

    def level4(self, used):
        return _build_l4("any", self.dm, self.wn, used)


# ─────────────────────────────────────────────────────────────
# Assembler + Main Generator
# ─────────────────────────────────────────────────────────────
class Assembler:
    def build(self, groups: list, arch: str) -> Puzzle:
        words = [w.text for g in groups for w in g.words]
        dupes = {w for w in words if words.count(w) > 1}
        if dupes:
            raise ValueError(f"Duplicate words: {dupes}")
        random.shuffle(words)
        ws = [w for g in groups for w in g.words]
        score = round(
            min(sum(w.frequency for w in ws) / max(len(ws), 1) / 300.0, 1.0) * 40 +
            len({g.level for g in groups}) / 4.0 * 30 +
            min(sum(w.polysemy for w in ws) / max(len(ws), 1) / 8.0, 1.0) * 30, 1)
        return Puzzle(groups=groups, grid=words, score=score, arch=arch)


class NYTConnectionsGenerator:

    def __init__(self, key: str):
        print("  [init] Starting…")
        self.bn = BabelNetDisambiguator(key)
        self.wn = WordNetMiner()
        self.dm = DatamuseMiner()
        self.flt = WordFilter(min_freq=8)
        self.asm = Assembler()
        self.bn.probe()

    def _arch(self, name: str):
        return {"A": ArchitectureA, "B": ArchitectureB, "C": ArchitectureC}[name](
            self.bn, self.wn, self.dm, self.flt)

    def generate_one(self, arch_name: str = "A", retries: int = 12,
                     global_used: set = None) -> Puzzle:
        base_used = set(global_used) if global_used else set()
        arch = self._arch(arch_name)

        for attempt in range(retries):
            print(f"\n  [gen-{arch_name}] Attempt {attempt+1}/{retries}…")
            used, groups, others = set(base_used), [], []
            try:
                g1 = arch.level1(used, others)
                if not g1:
                    print("    L1 failed")
                    continue
                used.update(w.text for w in g1.words)
                others.append([w.text for w in g1.words])
                groups.append(g1)
                print(f"    ✓ L1 [{g1.name}]: {[w.text for w in g1.words]}")

                g2 = arch.level2(used, others)
                if not g2:
                    print("    L2 failed")
                    continue
                used.update(w.text for w in g2.words)
                others.append([w.text for w in g2.words])
                groups.append(g2)
                print(f"    ✓ L2 [{g2.name}]: {[w.text for w in g2.words]}")

                g3 = arch.level3(used, others)
                if not g3:
                    print("    L3 failed")
                    continue
                used.update(w.text for w in g3.words)
                others.append([w.text for w in g3.words])
                groups.append(g3)
                print(f"    ✓ L3 [{g3.name}]: {[w.text for w in g3.words]}")

                # L4: pass used set so stems don't duplicate grid words
                g4 = arch.level4(used)
                if not g4:
                    print("    L4 failed")
                    continue
                used.update(w.text for w in g4.words)
                groups.append(g4)
                print(f"    ✓ L4 [{g4.name}]: {[w.text for w in g4.words]}")

                puzzle = self.asm.build(groups, arch_name)
                print(f"    ★ Score: {puzzle.score}/100  arch={arch_name}")
                return puzzle

            except Exception as e:
                print(f"    Error: {e}")

        raise RuntimeError(
            f"Arch {arch_name}: failed after {retries} attempts.")

    def generate_dataset(self, n: int = 5) -> list:
        arch_cycle = ["A", "B", "C"]
        puzzles, global_used = [], set()
        for i in range(n):
            arch = arch_cycle[i % 3]
            arch_obj = {"A": ArchitectureA,
                        "B": ArchitectureB, "C": ArchitectureC}[arch]
            print(f"\n{'='*60}")
            print(f"  Puzzle {i+1}/{n}  |  Architecture {arch}")
            print(f"  {arch_obj.NAME}")
            print(f"{'='*60}")
            try:
                p = self.generate_one(arch_name=arch, global_used=global_used)
                puzzles.append(p)
                global_used.update(p.grid)
                print(f"\n  → {len(global_used)} words used globally so far")
            except Exception as e:
                print(f"  [!] Puzzle {i+1} failed: {e}")
        return puzzles

    def print_puzzle(self, p: Puzzle):
        colors = {1: "🟡 YELLOW", 2: "🟢 GREEN", 3: "🔵 BLUE", 4: "🟣 PURPLE"}
        print(f"\n{'═'*60}")
        print(f"  NYT CONNECTIONS — Architecture {p.arch}")
        arch_names = {"A": ArchitectureA.NAME,
                      "B": ArchitectureB.NAME, "C": ArchitectureC.NAME}
        print(f"  {arch_names[p.arch]}")
        print(f"{'═'*60}\n")
        print("GRID:\n")
        for i in range(0, 16, 4):
            print("  " + "  │  ".join(w.upper().ljust(10)
                  for w in p.grid[i:i+4]))
        print(f"\nScore: {p.score}/100\n\nANSWERS:\n")
        for g in p.groups:
            print(f"  {colors[g.level]} — {g.name}  [{g.pattern}]")
            for w in g.words:
                print(
                    f"    {w.text.upper():12}  poly={w.polysemy}  freq={w.frequency}")
            print()
        print("═"*60)

    def to_jsonl(self, p: Puzzle) -> dict:
        cm = {1: "yellow", 2: "green", 3: "blue", 4: "purple"}
        return {
            "architecture":  p.arch,
            "grid":          p.grid,
            "quality_score": p.score,
            "groups": [{
                "level":     g.level,
                "color":     cm[g.level],
                "category":  g.name,
                "pattern":   g.pattern,
                "wn_synset": g.wn_synset,
                "words": [{"word": w.text, "polysemy": w.polysemy,
                           "frequency": w.frequency} for w in g.words],
            } for g in p.groups],
            "all_words":   [w.text for g in p.groups for w in g.words],
            "word_labels": [g.level for g in p.groups for _ in g.words],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key",     required=True)
    parser.add_argument("--puzzles", type=int, default=5)
    parser.add_argument("--output",  default="dataset.jsonl")
    parser.add_argument("--arch",    default="cycle",
                        help="A, B, C, or cycle (default: cycle A→B→C→...)")
    args = parser.parse_args()

    gen = NYTConnectionsGenerator(key=args.key)

    if args.arch == "cycle":
        puzzles = gen.generate_dataset(n=args.puzzles)
    else:
        puzzles = []
        for _ in range(args.puzzles):
            try:
                puzzles.append(gen.generate_one(arch_name=args.arch))
            except Exception as e:
                print(f"  [!] Failed: {e}")

    if not puzzles:
        print("\n[!] No puzzles generated.")
    else:
        for i, p in enumerate(puzzles):
            print(f"\n{'#'*60}\n  PUZZLE {i+1}  (Architecture {p.arch})\n{'#'*60}")
            gen.print_puzzle(p)
        with open(args.output, "w") as f:
            for p in puzzles:
                f.write(json.dumps(gen.to_jsonl(p)) + "\n")
        print(f"\n✅  {len(puzzles)} puzzles saved → {args.output}")
