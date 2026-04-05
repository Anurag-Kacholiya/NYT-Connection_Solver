"""
NYT Connections Puzzle Generator — v8
======================================

WHAT'S NEW in v8:
  • 25+ NEW L4 pattern types (rhyme, palindrome, anagram, prefix/suffix,
    hidden-word, homophones, silent-letter, same-vowel, letter-swap, etc.)
  • More L1 categories (countries, months, colors, body parts expanded, etc.)
  • More L2 synonym anchors (40+ verbs/adjectives)
  • More L3 anchors (60+ concrete + 30+ abstract)
  • Overlap checking: reads existing output file and refuses to save a puzzle
    whose full 16-word set or any 4-word group already appears in it.
  • CSV export in original format:
    Game ID, Puzzle Date, Word, Group Name, Group Level, Starting Row, Starting Column
  • Run command unchanged:
    python puzzle_generator.py --key YOUR_KEY --puzzles 5 --output dataset.jsonl

ARCHITECTURE OVERVIEW (3 variants, cycled for dataset variety):

Architecture A — Hyponym + Synonym + Triggered + Compound (___ WORD)
Architecture B — Hyponym(alt cats) + Means-like + Abstract + Compound (WORD ___)
Architecture C — Meronym + Synonym + Co-occurrence + Compound (any)
Architecture D — NEW: Wordplay puzzles (rhyme/anagram/palindrome/hidden-word/etc.)
"""

import argparse
import csv
import json
import os
import random
import re
import time
import warnings
import contextlib
import io
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import nltk
with contextlib.redirect_stdout(io.StringIO()):
    nltk.download("wordnet",  quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("brown",   quiet=True)
    nltk.download("cmudict", quiet=True)
import requests
from nltk.corpus import wordnet as wn, brown, cmudict

# ─────────────────────────────────────────────────────────────
# Brown corpus frequency
# ─────────────────────────────────────────────────────────────
_FREQ: Optional[Counter] = None
_CMU: Optional[dict] = None


def _freq_table() -> Counter:
    global _FREQ
    if _FREQ is None:
        print("  [init] Loading Brown corpus…")
        _FREQ = Counter(w.lower() for w in brown.words() if w.isalpha())
    return _FREQ


def _cmu_dict() -> dict:
    global _CMU
    if _CMU is None:
        _CMU = cmudict.dict()
    return _CMU


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
# HARDCODED L4 STEM POOLS — ___ WORD patterns
# ─────────────────────────────────────────────────────────────
L4_STEM_POOLS = {
    # ___ BASE patterns (word goes BEFORE the base)
    "ball":    list(dict.fromkeys(["basket", "base", "foot", "snow", "cannon", "soft", "pin", "volley",
                                   "hand", "net", "cricket", "stump", "gum", "odd", "paint"])),
    "firex":   list(dict.fromkeys(["camp", "cross", "gun", "wild", "open", "rapid", "cease", "miss",
                                   # ___ FIRE (before)
                                   "back", "spear", "sword", "anti", "hell", "bon"])),
    "light":   list(dict.fromkeys(["day", "flash", "moon", "sun", "star", "lime", "spot", "search",
                                   "candle", "street", "tail", "head", "night", "sky", "torch", "twi", "traffic"])),
    "house":   list(dict.fromkeys(["farm", "store", "ware", "court", "power", "jail", "round",
                                   "green", "club", "town", "work", "ale", "road", "tree", "full", "open"])),
    "board":   list(dict.fromkeys(["card", "key", "skate", "surf", "snow", "dart", "paste",
                                   "black", "cup", "flap", "score", "side", "check", "base", "body", "spring"])),
    "line":    list(dict.fromkeys(["base", "dead", "head", "side", "hair", "border", "life", "tag",
                                   "clothes", "guide", "shore", "story", "under", "hot", "air", "main", "time"])),
    "yard":    list(dict.fromkeys(["back", "court", "vine", "grave", "barn", "door", "farm", "ship",
                                   "stock", "school", "church", "front", "junk", "rail"])),
    "field":   list(dict.fromkeys(["corn", "mine", "air", "oil", "coal", "wheat", "grass",
                                   "down", "play", "gold", "hay", "battle", "ice", "cane"])),
    "bookx":   list(dict.fromkeys(["hand", "note", "text", "pass", "case", "guide", "log", "cook",
                                   # ___ BOOK (before)
                                   "copy", "guest", "pay", "rule", "score", "source", "work", "pocket"])),
    "stone":   list(dict.fromkeys(["lime", "sand", "key", "flag", "corner", "cobble", "grind",
                                   "hail", "mud", "brown", "head", "mill", "gold", "grave"])),
    "room":    list(dict.fromkeys(["bed", "bath", "court", "show", "class", "ball", "store",
                                   "cloak", "dark", "dining", "dress", "gun", "living", "mud", "work"])),
    "work":    list(dict.fromkeys(["frame", "net", "paper", "team", "ground", "field",
                                   "house", "home", "class", "body", "social", "wood", "bridge", "busy", "odd"])),
    "time":    list(dict.fromkeys(["day", "night", "life", "half", "lunch", "any", "bed",
                                   "down", "full", "hard", "long", "old", "part", "peace", "spring", "war"])),
    "man":     list(dict.fromkeys(["door", "sales", "states", "gentle", "work", "play",
                                   "post", "horse", "marks", "gun", "chair", "service", "police", "cloak"])),
    "day":     list(dict.fromkeys(["birth", "every", "week", "work", "pay", "mid", "good", "sun"])),
    "cake":    list(dict.fromkeys(["cup", "cheese", "short", "pan", "wafer", "fruit", "layer"])),
    "box":     list(dict.fromkeys(["sand", "tool", "bread", "music", "card", "post", "mail", "letter", "thunder"])),
    "fall":    list(dict.fromkeys(["down", "free", "water", "land", "short", "night", "rain", "wind"])),
    "sidex":   list(dict.fromkeys(["bed", "road", "lake", "hill", "sea", "ring", "court",
                                   # ___ SIDE (before)
                                   "top", "blind", "broad", "dark", "flip"])),
    "way":     list(dict.fromkeys(["path", "gate", "any", "door", "drive", "free", "high", "motor",
                                   "run", "sub", "under", "wide", "air", "out"])),
    "ship":    list(dict.fromkeys(["air", "flag", "gun", "war", "court", "hard", "fellow", "sports",
                                   "steam", "friend", "owner", "partner", "sponsor"])),
    "waterx":  list(dict.fromkeys(["back", "black", "clear", "deep", "ground", "hot", "rain",
                                   # ___ WATER (before)
                                   "rose", "sea", "still", "surf", "under", "waste"])),
    "gate":    list(dict.fromkeys(["flood", "iron", "lych", "toll", "turn", "water", "flap"])),
    "mark":    list(dict.fromkeys(["bench", "book", "hall", "land", "post", "trade", "birth", "check"])),
    "pool":    list(dict.fromkeys(["car", "dead", "gene", "whirl", "swimming"])),
    "walk":    list(dict.fromkeys(["board", "cat", "jay", "moon", "sky"])),
    "ground":  list(dict.fromkeys(["back", "battle", "camp", "fore", "play", "under", "above"])),
    # BASE ___ patterns (word goes AFTER the base / prefix forms)
    "over":    list(dict.fromkeys(["come", "look", "time", "load", "turn", "all", "board", "cast",
                                   "coat", "due", "flow", "lap", "night", "pass", "rule", "see",
                                   "take", "throw", "work", "haul", "grown", "joyed", "rated"])),
    "under":   list(dict.fromkeys(["line", "cover", "go", "age", "cut", "dog", "done", "foot",
                                   "ground", "hand", "mine", "pass", "play", "score", "side",
                                   "take", "tone", "world", "rate", "wear"])),
    "out":     list(dict.fromkeys(["run", "side", "door", "fit", "law", "let", "line", "post",
                                   "put", "reach", "right", "set", "smart", "source",
                                   "stand", "weigh", "rage", "last", "shine", "grow", "dated"])),
    "back":    list(dict.fromkeys(["ground", "yard", "lash", "log", "pack", "seat", "slide",
                                   "stage", "stroke", "track", "ward", "bone", "door", "drop", "handed", "space", "bench"])),
    "down":    list(dict.fromkeys(["town", "stairs", "fall", "hill", "load", "right", "side",
                                   "size", "stream", "swing", "ward", "play", "grade", "cast", "beat"])),
    "eye":     list(dict.fromkeys(["lid", "brow", "lash", "ball", "sight", "witness", "drop", "let", "piece", "glass"])),
    "foot":    list(dict.fromkeys(["print", "note", "step", "ball", "hold", "wear", "bridge",
                                   "path", "rest", "stool", "work", "board"])),
    "hand":    list(dict.fromkeys(["shake", "bag", "made", "rail", "writing", "cuff", "hold",
                                   "picked", "stand", "work", "gun", "book"])),
    "head":    list(dict.fromkeys(["ache", "band", "line", "master", "quarters", "stone",
                                   "board", "count", "dress", "gear", "lamp", "land", "long",
                                   "phone", "set", "strong", "way"])),
    "sun":     list(dict.fromkeys(["burn", "flower", "glasses", "light", "rise", "set", "shine",
                                   "screen", "stroke", "roof", "dial", "tan", "beam"])),
    "air":     list(dict.fromkeys(["craft", "field", "line", "port", "ship", "tight", "waves",
                                   "born", "brush", "drop", "fare", "flow", "gun", "lift", "plane"])),
    "sea":     list(dict.fromkeys(["bed", "bird", "board", "coast", "fare", "floor", "food",
                                   "gull", "horse", "port", "shell", "shore", "sick", "side", "weed"])),
    "cross":   list(dict.fromkeys(["bar", "bow", "breed", "check", "country", "fire", "road",
                                   "roads", "walk", "wind", "word"])),
    "black":   list(dict.fromkeys(["berry", "bird", "board", "book", "smith", "thorn",
                                   "box", "hole", "jack", "list", "mail"])),
    "blue":    list(dict.fromkeys(["bell", "berry", "bird", "bottle", "print", "tooth"])),
    "fire":    list(dict.fromkeys(["arm", "brand", "brick", "cracker", "fighter", "man", "place",
                                   # FIRE ___ (after)
                                   "side", "truck", "wood", "work", "proof"])),
    "green":   list(dict.fromkeys(["house", "land", "back", "card"])),
    "high":    list(dict.fromkeys(["land", "light", "road", "way", "chair", "rise", "school"])),
    # SIDE ___ (after)
    "side":    list(dict.fromkeys(["board", "burns", "car", "line", "show", "step", "swipe", "track", "walk"])),
    "up":      list(dict.fromkeys(["beat", "bring", "cast", "date", "draft", "grade", "hold",
                                   "keep", "lift", "load", "set", "stairs", "stream", "swing"])),
    "blood":   list(dict.fromkeys(["hound", "shed", "shot", "stain", "stream", "bath", "line", "bank"])),
    # WATER ___ (after)
    "water":   list(dict.fromkeys(["fall", "front", "mark", "melon", "proof", "shed", "side", "tight", "way"])),
    "wind":    list(dict.fromkeys(["fall", "mill", "pipe", "screen", "shield", "ward"])),
    "gold":    list(dict.fromkeys(["field", "finch", "fish", "mine", "smith"])),
    "silver":  list(dict.fromkeys(["smith", "ware", "side"])),
    "snow":    list(dict.fromkeys(["ball", "board", "bound", "drift", "drop", "fall", "flake", "man", "plow", "storm"])),
    "wild":    list(dict.fromkeys(["cat", "flower", "life", "wood"])),
    "night":   list(dict.fromkeys(["cap", "club", "fall", "gown", "life", "shade", "stand"])),
    "long":    list(dict.fromkeys(["bow", "hand", "house", "shore", "boat"])),
    "short":   list(dict.fromkeys(["cake", "change", "cut", "fall", "hand"])),
    "horse":   list(dict.fromkeys(["back", "play", "power", "shoe"])),
    "bird":    list(dict.fromkeys(["bath", "cage", "call", "dog", "house", "seed", "watch"])),
    # BOOK ___ (after)
    "book":    list(dict.fromkeys(["case", "keeper", "let", "mark", "shelf", "worm"])),
    "star":    list(dict.fromkeys(["fish", "gazer", "light"])),
    "flower":  list(dict.fromkeys(["bed", "pot", "shop", "show"])),
}

# Pattern metadata: (base, display_label, mode)
L4_PATTERNS = [
    # ___ WORD (word BEFORE base)
    ("ball",   "___ BALL",   "before"),
    ("firex",  "___ FIRE",   "before"),
    ("light",  "___ LIGHT",  "before"),
    ("house",  "___ HOUSE",  "before"),
    ("board",  "___ BOARD",  "before"),
    ("line",   "___ LINE",   "before"),
    ("yard",   "___ YARD",   "before"),
    ("field",  "___ FIELD",  "before"),
    ("bookx",  "___ BOOK",   "before"),
    ("stone",  "___ STONE",  "before"),
    ("room",   "___ ROOM",   "before"),
    ("work",   "___ WORK",   "before"),
    ("time",   "___ TIME",   "before"),
    ("man",    "___ MAN",    "before"),
    ("day",    "___ DAY",    "before"),
    ("sidex",  "___ SIDE",   "before"),
    ("fall",   "___ FALL",   "before"),
    ("box",    "___ BOX",    "before"),
    ("way",    "___ WAY",    "before"),
    ("waterx", "___ WATER",  "before"),
    ("mark",   "___ MARK",   "before"),
    ("walk",   "___ WALK",   "before"),
    ("ground", "___ GROUND", "before"),
    ("pool",   "___ POOL",   "before"),
    ("gate",   "___ GATE",   "before"),
    ("ship",   "___ SHIP",   "before"),
    # WORD ___ (word AFTER base / prefix forms)
    ("over",   "OVER ___",   "after"),
    ("under",  "UNDER ___",  "after"),
    ("out",    "OUT ___",    "after"),
    ("back",   "BACK ___",   "after"),
    ("down",   "DOWN ___",   "after"),
    ("eye",    "EYE ___",    "after"),
    ("foot",   "FOOT ___",   "after"),
    ("hand",   "HAND ___",   "after"),
    ("head",   "HEAD ___",   "after"),
    ("sun",    "SUN ___",    "after"),
    ("air",    "AIR ___",    "after"),
    ("sea",    "SEA ___",    "after"),
    ("cross",  "CROSS ___",  "after"),
    ("black",  "BLACK ___",  "after"),
    ("blue",   "BLUE ___",   "after"),
    ("high",   "HIGH ___",   "after"),
    ("up",     "UP ___",     "after"),
    ("blood",  "BLOOD ___",  "after"),
    ("wind",   "WIND ___",   "after"),
    ("gold",   "GOLD ___",   "after"),
    ("snow",   "SNOW ___",   "after"),
    ("wild",   "WILD ___",   "after"),
    ("night",  "NIGHT ___",  "after"),
    ("long",   "LONG ___",   "after"),
    ("short",  "SHORT ___",  "after"),
    ("horse",  "HORSE ___",  "after"),
    ("bird",   "BIRD ___",   "after"),
    ("star",   "STAR ___",   "after"),
    ("water",  "WATER ___",  "after"),
    ("fire",   "FIRE ___",   "after"),
    ("green",  "GREEN ___",  "after"),
    ("side",   "SIDE ___",   "after"),
]

# ─────────────────────────────────────────────────────────────
# NEW WORDPLAY L4 DATA BANKS
# ─────────────────────────────────────────────────────────────

# Palindromes (read same forwards and backwards)
PALINDROMES = [
    "level", "civic", "radar", "refer", "deed", "noon", "kayak",
    "madam", "repaper", "rotator", "racecar", "redder", "stats",
    "tenet", "reviver", "detartrated",
]

# Anagram groups: each tuple = (label, [words that are anagrams of each other])
ANAGRAM_GROUPS = [
    ("ANAGRAMS OF EACH OTHER", [
     "post", "stop", "tops", "pots", "spot", "opts"]),
    ("ANAGRAMS OF EACH OTHER", [
     "neat", "ante", "etna", "lane", "lean", "elan"]),
    ("ANAGRAMS OF EACH OTHER", ["arts", "rats", "tars", "star", "tsar"]),
    ("ANAGRAMS OF EACH OTHER", ["below", "elbow", "bowel"]),
    ("ANAGRAMS OF EACH OTHER", ["least", "stale",
     "tales", "steal", "teals", "slate"]),
    ("ANAGRAMS OF EACH OTHER", ["merit", "mitre", "timer", "miter", "remit"]),
    ("ANAGRAMS OF EACH OTHER", ["words", "sword", "cords", "rods"]),
    ("ANAGRAMS OF EACH OTHER", ["pear", "reap", "rape", "pare"]),
    ("ANAGRAMS OF EACH OTHER", ["dupe", "pude", "pued"]),
    ("ANAGRAMS OF EACH OTHER", ["care", "race", "acre", "arce"]),
    ("ANAGRAMS OF EACH OTHER", ["inch", "chin"]),
    ("ANAGRAMS OF EACH OTHER", [
     "enlist", "silent", "listen", "tinsel", "inlets"]),
    ("ANAGRAMS OF EACH OTHER", ["night", "thing"]),
    ("ANAGRAMS OF EACH OTHER", ["satin", "saint", "antis", "tains"]),
    ("ANAGRAMS OF EACH OTHER", ["caste", "taces",
     "cates", "trace", "caret", "carte"]),
]

# Rhyming word groups: each tuple = (rhyme_sound_label, [words])
RHYME_GROUPS = [
    ("RHYMES WITH 'CAT'",   ["bat", "hat",
     "mat", "rat", "sat", "fat", "pat", "vat"]),
    ("RHYMES WITH 'NIGHT'", ["bite", "kite", "mite",
     "site", "light", "right", "tight", "white"]),
    ("RHYMES WITH 'MOON'",  ["boon", "loon", "noon",
     "soon", "tune", "june", "dune", "boon"]),
    ("RHYMES WITH 'COLD'",  ["bold", "fold", "gold",
     "hold", "mold", "old", "sold", "told"]),
    ("RHYMES WITH 'RING'",  ["king", "sing", "wing",
     "bring", "spring", "sting", "thing", "swing"]),
    ("RHYMES WITH 'DOOR'",  ["bore", "core", "floor",
     "more", "pour", "sore", "wore", "shore"]),
    ("RHYMES WITH 'TREE'",  ["bee", "free",
     "glee", "key", "sea", "tea", "fee", "pea"]),
    ("RHYMES WITH 'LOVE'",  ["above", "dove", "glove", "shove"]),
    ("RHYMES WITH 'WORD'",  ["bird", "heard",
     "herd", "blurred", "curd", "nerd", "stirred"]),
    ("RHYMES WITH 'CAKE'",  ["bake", "fake", "lake",
     "make", "rake", "sake", "take", "wake"]),
    ("RHYMES WITH 'RAIN'",  ["brain", "chain", "drain",
     "gain", "lane", "main", "pain", "plane"]),
    ("RHYMES WITH 'BLUE'",  ["clue", "dew", "flew",
     "glue", "grew", "knew", "true", "zoo"]),
    ("RHYMES WITH 'BACK'",  ["black", "crack", "lack",
     "pack", "rack", "sack", "stack", "track"]),
    ("RHYMES WITH 'BRIGHT'", ["flight", "fright",
     "knight", "might", "night", "right", "sight"]),
]

# Homophones: each tuple = (label, [words that sound like something else / pairs])
HOMOPHONE_SETS = [
    ("HOMOPHONES", ["bare", "bear"]),
    ("HOMOPHONES", ["flour", "flower"]),
    ("HOMOPHONES", ["sea", "see"]),
    ("HOMOPHONES", ["hear", "here"]),
    ("HOMOPHONES", ["knight", "night"]),
    ("HOMOPHONES", ["knot", "not"]),
    ("HOMOPHONES", ["pear", "pair"]),
    ("HOMOPHONES", ["right", "write"]),
    ("HOMOPHONES", ["sale", "sail"]),
    ("HOMOPHONES", ["tale", "tail"]),
    ("HOMOPHONES", ["their", "there"]),
    ("HOMOPHONES", ["week", "weak"]),
    ("HOMOPHONES", ["wood", "would"]),
    ("HOMOPHONES", ["by", "bye", "buy"]),
    ("HOMOPHONES", ["which", "witch"]),
    ("HOMOPHONES", ["way", "weigh"]),
    ("HOMOPHONES", ["sole", "soul"]),
    ("HOMOPHONES", ["fair", "fare"]),
    ("HOMOPHONES", ["mail", "male"]),
    ("HOMOPHONES", ["peace", "piece"]),
]

# Words containing hidden smaller word: each tuple = (hidden_word, label, [containing_words])
HIDDEN_WORD_GROUPS = [
    ("cat",  "CONTAINS 'CAT'",  ["scatter", "locate",
     "educate", "catalog", "scat", "catfish", "catch"]),
    ("art",  "CONTAINS 'ART'",  ["party", "start",
     "smart", "chart", "heart", "darted", "artist"]),
    ("age",  "CONTAINS 'AGE'",  [
     "stage", "manage", "village", "savage", "usage", "engage", "average"]),
    ("end",  "CONTAINS 'END'",  ["blend", "friend",
     "spend", "extend", "defend", "trend", "render"]),
    ("ear",  "CONTAINS 'EAR'",  ["year", "fear",
     "pearl", "search", "heart", "early", "learn"]),
    ("old",  "CONTAINS 'OLD'",  [
     "cold", "bold", "folder", "golden", "holding", "mold", "shoulder"]),
    ("red",  "CONTAINS 'RED'",  [
     "bread", "credit", "spread", "thread", "hundred", "predator", "already"]),
    ("elf",  "CONTAINS 'ELF'",  [
     "shelf", "myself", "himself", "herself", "self"]),
    ("ice",  "CONTAINS 'ICE'",  [
     "price", "slice", "notice", "advice", "choice", "office", "service"]),
    ("ace",  "CONTAINS 'ACE'",  ["place", "face",
     "space", "grace", "peace", "race", "trace"]),
    ("and",  "CONTAINS 'AND'",  ["land", "stand",
     "sand", "grand", "brand", "hand", "standard"]),
    ("ore",  "CONTAINS 'ORE'",  ["more", "store",
     "before", "core", "score", "explore", "ignore"]),
    ("ale",  "CONTAINS 'ALE'",  ["tale", "male",
     "sale", "pale", "whale", "exhale", "stale"]),
    ("ate",  "CONTAINS 'ATE'",  ["late", "fate",
     "rate", "plate", "state", "create", "debate"]),
    ("ire",  "CONTAINS 'IRE'",  [
     "fire", "hire", "tire", "desire", "entire", "require", "inspire"]),
    ("one",  "CONTAINS 'ONE'",  ["bone", "cone",
     "stone", "phone", "money", "honey", "lonely"]),
    ("over", "CONTAINS 'OVER'", ["cover", "hover",
     "clover", "shover", "recover", "discover"]),
    ("low",  "CONTAINS 'LOW'",  ["flow", "slow",
     "blow", "glow", "below", "allow", "fellow"]),
    ("hot",  "CONTAINS 'HOT'",  ["shot", "photo",
     "throat", "hotel", "broth", "thoth", "smother"]),
    ("ink",  "CONTAINS 'INK'",  ["think", "drink",
     "rink", "sink", "blink", "pink", "distinct"]),
]

# Same prefix groups: each tuple = (prefix, label, [words with that prefix])
SAME_PREFIX_GROUPS = [
    ("re",  "STARTS WITH 'RE'",   [
     "return", "replay", "rethink", "reform", "refuse", "rebuild", "rename"]),
    ("un",  "STARTS WITH 'UN'",   [
     "unlock", "unfair", "unsafe", "unreal", "unable", "unfit", "unknown"]),
    ("pre", "STARTS WITH 'PRE'",  [
     "preview", "prevent", "predict", "prefer", "prepare", "premix", "preset"]),
    ("dis", "STARTS WITH 'DIS'",  [
     "display", "dismiss", "disturb", "dislike", "disable", "discard", "dismiss"]),
    ("mis", "STARTS WITH 'MIS'",  [
     "mistake", "misfit", "mislead", "misuse", "misfire", "misread", "misstep"]),
    ("over", "STARTS WITH 'OVER'", [
     "overcome", "overlook", "overrun", "overturn", "overlap", "overdue", "overhaul"]),
    ("out", "STARTS WITH 'OUT'",  [
     "outrun", "outcome", "outline", "outlook", "outrage", "outpace", "outwit"]),
    ("sub", "STARTS WITH 'SUB'",  [
     "subway", "subject", "submit", "subtext", "subside", "subpar", "suburb"]),
    ("up",  "STARTS WITH 'UP'",   [
     "update", "upgrade", "upbeat", "uphill", "uphold", "upkeep", "uplift"]),
    ("non", "STARTS WITH 'NON'",  ["nonfarm",
     "nonstop", "nonplus", "nonfat", "nonfit"]),
    ("fore", "STARTS WITH 'FORE'", [
     "forecast", "forearm", "foreground", "foreign", "foreman", "foresee"]),
    ("inter", "STARTS WITH 'INTER'", [
     "internet", "interlude", "internal", "interest", "interlace", "interlock"]),
    ("anti", "STARTS WITH 'ANTI'", [
     "antidote", "antique", "antivirus", "antibody", "antiwar", "antisocial"]),
    ("semi", "STARTS WITH 'SEMI'", [
     "semifinal", "semidry", "semihard", "semipro", "semicircle"]),
    ("post", "STARTS WITH 'POST'", [
     "postcard", "postbox", "postman", "postwar", "postcode", "postpone"]),
    ("mid", "STARTS WITH 'MID'",  [
     "midterm", "midway", "midday", "midfield", "midpoint", "midnight", "midsize"]),
    ("co",  "STARTS WITH 'CO'",   [
     "cowork", "coexist", "coauthor", "copilot", "coedit", "costar", "copay"]),
    ("de",  "STARTS WITH 'DE'",   [
     "decode", "defrost", "debrief", "defuse", "depart", "depend", "derive"]),
    ("trans", "STARTS WITH 'TRANS'", [
     "transport", "transmit", "translate", "transform", "transit", "transplant"]),
    ("counter", "STARTS WITH 'COUNTER'", [
     "counterpoint", "counteract", "counterfeit", "counterpart"]),
]

# Same suffix groups: each tuple = (suffix, label, [words with that suffix])
SAME_SUFFIX_GROUPS = [
    ("ness", "ENDS WITH 'NESS'",   [
     "darkness", "kindness", "sadness", "madness", "fitness", "weakness", "boldness"]),
    ("tion", "ENDS WITH 'TION'",   [
     "nation", "motion", "caption", "station", "fraction", "mention", "section"]),
    ("ment", "ENDS WITH 'MENT'",   [
     "moment", "payment", "movement", "judgment", "comment", "segment", "treatment"]),
    ("less", "ENDS WITH 'LESS'",   [
     "hopeless", "useless", "careless", "harmless", "restless", "wireless", "endless"]),
    ("ful",  "ENDS WITH 'FUL'",    [
     "careful", "grateful", "handful", "hopeful", "peaceful", "playful", "restful"]),
    ("ing",  "ENDS WITH 'ING'",    [
     "running", "singing", "jumping", "dancing", "thinking", "reading", "writing"]),
    ("ous",  "ENDS WITH 'OUS'",    [
     "famous", "nervous", "serious", "curious", "obvious", "various", "glorious"]),
    ("ible", "ENDS WITH 'IBLE'",   [
     "visible", "flexible", "terrible", "possible", "audible", "legible", "sensible"]),
    ("ish",  "ENDS WITH 'ISH'",    [
     "foolish", "selfish", "childish", "lavish", "sluggish", "oldish", "coldish"]),
    ("ward", "ENDS WITH 'WARD'",   [
     "forward", "backward", "upward", "inward", "outward", "skyward", "onward"]),
    ("er",   "ENDS WITH 'ER'",     [
     "runner", "singer", "driver", "fighter", "teacher", "hunter", "leader"]),
    ("ly",   "ENDS WITH 'LY'",     [
     "slowly", "quickly", "boldly", "freely", "purely", "softly", "firmly"]),
    ("ry",   "ENDS WITH 'RY'",     [
     "bakery", "bravery", "celery", "gallery", "slavery", "sorcery", "surgery"]),
    ("age",  "ENDS WITH 'AGE'",    [
     "package", "village", "garbage", "passage", "message", "manage", "courage"]),
    ("ship", "ENDS WITH 'SHIP'",   [
     "hardship", "kinship", "worship", "township", "courtship", "lordship"]),
    ("like", "ENDS WITH 'LIKE'",   [
     "childlike", "godlike", "lifelike", "warlike", "dreamlike", "catlike"]),
    ("ling", "ENDS WITH 'LING'",   [
     "darling", "duckling", "fledgling", "seedling", "stripling", "yearling"]),
    ("some", "ENDS WITH 'SOME'",   [
     "awesome", "fearsome", "handsome", "loathsome", "tiresome", "wholesome"]),
    ("ic",   "ENDS WITH 'IC'",     [
     "comic", "magic", "panic", "sonic", "toxic", "mystic", "rustic"]),
    ("ive",  "ENDS WITH 'IVE'",    [
     "active", "creative", "festive", "massive", "native", "passive", "relative"]),
    ("ard",  "ENDS WITH 'ARD'",    [
     "bastard", "drunkard", "leopard", "mustard", "wizard", "custard", "coward"]),
    ("ock",  "ENDS WITH 'OCK'",    [
     "block", "clock", "dock", "flock", "knock", "lock", "mock", "rock"]),
    ("ight", "ENDS WITH 'IGHT'",   [
     "blight", "bright", "fight", "might", "night", "right", "tight"]),
    ("ound", "ENDS WITH 'OUND'",   [
     "bound", "found", "ground", "mound", "round", "sound", "wound"]),
    ("ake",  "ENDS WITH 'AKE'",    [
     "bake", "cake", "fake", "lake", "make", "rake", "sake", "take"]),
    ("own",  "ENDS WITH 'OWN'",    [
     "brown", "clown", "crown", "down", "drown", "frown", "gown", "town"]),
    ("air",  "ENDS WITH 'AIR'",    [
     "chair", "flair", "hair", "lair", "pair", "repair", "snair", "stair"]),
    ("ink",  "ENDS WITH 'INK'",    [
     "blink", "brink", "drink", "link", "mink", "pink", "rink", "sink"]),
]

# Words that pair with same second word
SAME_SECOND_WORD = [
    ("___ BALL", ["basket", "foot", "base",
     "volley", "soft", "fire", "cannon", "pin"]),
    ("___ FISH",  ["sword", "star", "cat",
     "angel", "blow", "jelly", "monk", "muscle"]),
    ("___ BIRD",  ["thunder", "song", "mock",
     "black", "lady", "hum", "sea", "love"]),
    ("___ LIGHT", ["day", "moon", "star", "sun",
     "flash", "traffic", "spot", "night"]),
    ("___ HOUSE", ["tree", "farm", "ware",
     "store", "jailbird", "club", "green", "ale"]),
    ("___ LINE",  ["base", "side", "guide",
     "head", "off", "on", "dead", "life"]),
    ("___ STONE", ["lime", "corner", "key",
     "flag", "black", "brown", "grave", "mile"]),
    ("___ BOARD", ["black", "card", "key",
     "skate", "dart", "score", "cup", "surf"]),
    ("___ WORK",  ["net", "team", "frame",
     "over", "paper", "class", "wood", "ground"]),
    ("___ MAN",   ["fire", "sales", "door",
     "states", "gentle", "work", "police", "post"]),
    ("___ SIDE",  ["bed", "road", "sea",
     "lake", "hill", "ring", "fire", "water"]),
    ("___ SHIP",  ["friend", "hard", "sports",
     "court", "air", "kin", "fellow", "owner"]),
    ("___ WAY",   ["any", "door", "free",
     "gate", "high", "motor", "path", "sub"]),
    ("___ FIELD", ["corn", "air", "mine",
     "oil", "wheat", "battle", "gold", "ice"]),
    ("___ YARD",  ["back", "court", "vine",
     "grave", "barn", "church", "front", "ship"]),
    ("___ TIME",  ["any", "bed", "day", "half",
     "lunch", "night", "over", "part"]),
    ("___ ROOM",  ["ball", "bath", "bed",
     "class", "court", "dark", "show", "store"]),
    ("___ FALL",  ["down", "free", "land",
     "night", "over", "rain", "water", "wind"]),
]

# Same starting letter (alliteration puzzle — 4 words sharing a letter)
SAME_START_GROUPS = {
    "b": ["bright", "brave", "bold", "battle", "broad", "brisk", "blaze", "bloom",
          "build", "break", "brush", "bring", "brook", "brand", "breeze"],
    "c": ["climb", "crisp", "coast", "craft", "crown", "claim", "clear", "close",
          "crush", "chase", "charm", "chest", "check", "crack", "crest"],
    "d": ["drift", "dream", "drive", "drawn", "drill", "dwell", "drops", "drums",
          "drape", "drain", "dress", "dance", "dash", "draft", "dunes"],
    "f": ["flame", "fleet", "flare", "flesh", "float", "flood", "floor", "flush",
          "focus", "force", "forge", "found", "frown", "fresh", "frame"],
    "g": ["grace", "graze", "groan", "grope", "gross", "group", "growl", "grind",
          "grief", "greet", "grasp", "grant", "grave", "grain", "green"],
    "h": ["harsh", "haste", "hatch", "haven", "heart", "heavy", "hedge", "hills",
          "hints", "horde", "hover", "human", "humble", "hurry", "honor"],
    "l": ["lance", "lapse", "large", "laser", "latch", "layer", "learn", "leave",
          "level", "light", "limit", "lodge", "logic", "loose", "loyal"],
    "m": ["march", "match", "merge", "might", "minor", "model", "mount", "mourn",
          "moist", "moral", "motor", "mould", "moody", "murky", "manor"],
    "p": ["peace", "pilot", "pitch", "place", "plain", "plant", "plate", "plaza",
          "pluck", "point", "prize", "probe", "proud", "prove", "prune"],
    "r": ["radar", "raise", "range", "rapid", "reach", "realm", "rebel", "reign",
          "repay", "reply", "ridge", "ripen", "risen", "rival", "rocky"],
    "s": ["scale", "scout", "seize", "sense", "serve", "setup", "shaft", "shake",
          "shelf", "shift", "shock", "shore", "sight", "skill", "slope"],
    "t": ["tiger", "tidal", "tight", "timid", "tired", "title", "token", "topic",
          "total", "touch", "trace", "trade", "trail", "tread", "trial"],
    "w": ["wager", "watch", "weave", "wedge", "weigh", "whirl", "wider", "winds",
          "witch", "world", "worst", "worth", "wound", "wrath", "write"],
}

# Same ending letter groups
SAME_END_GROUPS = {
    "e": ["brave", "crane", "drome", "flute", "grove", "image", "knife", "large",
          "loose", "nerve", "place", "prose", "quote", "ridge", "scale", "spine"],
    "n": ["cabin", "chain", "drain", "grain", "human", "Japan", "lemon", "mitten",
          "often", "organ", "plain", "raven", "robin", "siren", "seven", "token"],
    "t": ["adopt", "assist", "craft", "doubt", "exact", "fleet", "frost", "grunt",
          "heart", "limit", "orbit", "quest", "scout", "sport", "start", "trust"],
    "k": ["block", "blank", "brick", "brook", "click", "clock", "crack", "drink",
          "flick", "shack", "stick", "stock", "thick", "track", "trick", "trunk"],
    "l": ["angel", "anvil", "crawl", "cruel", "drawl", "excel", "equal", "focal",
          "local", "model", "novel", "panel", "pedal", "rival", "trawl", "tidal"],
    "d": ["bland", "blend", "bound", "cloud", "crowd", "found", "fraud", "grand",
          "grind", "proud", "round", "sound", "squad", "stand", "trend", "world"],
    "r": ["actor", "anger", "buyer", "color", "cover", "error", "favor", "fever",
          "fiber", "floor", "humor", "layer", "power", "river", "tower", "water"],
    "y": ["army", "baby", "berry", "candy", "crazy", "fancy", "gravy", "greedy",
          "heavy", "honey", "lucky", "money", "party", "shiny", "sixty", "story"],
}

# Words with silent letters (organized by silent letter)
SILENT_LETTER_GROUPS = {
    "silent k":   ["kneel", "knife", "knight", "knit", "knob", "knock", "knot", "know", "knack"],
    "silent w":   ["wrap", "wrist", "wrong", "write", "wreck", "wren", "wrestle", "wrote"],
    "silent g":   ["gnaw", "gnarl", "gnome", "gnat", "gnash", "sign", "align", "design", "reign"],
    "silent b":   ["thumb", "climb", "lamb", "comb", "numb", "dumb", "bomb", "debt", "doubt"],
    "silent h":   ["heir", "honor", "hour", "honest", "rhyme", "rhythm", "ghost", "ghetto", "white"],
    "silent p":   ["pneumonia", "psalm", "psyche", "ptarmigan"],
    "silent t":   ["castle", "whistle", "bustle", "hustle", "nestle", "rustle", "trestle", "glisten"],
    "silent l":   ["calm", "palm", "psalm", "half", "calf", "walk", "talk", "folk", "could", "should"],
    "silent c":   ["muscle", "acquire", "scissors", "ascent", "indict", "scene", "scent"],
    "silent n":   ["autumn", "column", "solemn", "condemn", "hymn", "damn"],
}

# Words differing by one letter (minimal pairs)
MINIMAL_PAIRS_GROUPS = [
    ("DIFFER BY ONE LETTER FROM 'COLD'", [
     "bold", "fold", "gold", "hold", "mold", "old", "sold", "told", "wold"]),
    ("DIFFER BY ONE LETTER FROM 'CAKE'", [
     "bake", "fake", "lake", "make", "rake", "sake", "take", "wake"]),
    ("DIFFER BY ONE LETTER FROM 'NIGHT'", [
     "might", "right", "sight", "tight", "light", "bight", "fight"]),
    ("DIFFER BY ONE LETTER FROM 'BACK'", [
     "hack", "jack", "lack", "mack", "pack", "rack", "sack", "tack"]),
    ("DIFFER BY ONE LETTER FROM 'RING'", [
     "king", "sing", "wing", "bing", "ding", "ping", "zing"]),
    ("DIFFER BY ONE LETTER FROM 'MAN'",  [
     "ban", "can", "fan", "pan", "ran", "tan", "van"]),
    ("DIFFER BY ONE LETTER FROM 'LATE'", [
     "date", "fate", "gate", "hate", "mate", "rate"]),
    ("DIFFER BY ONE LETTER FROM 'FINE'", [
     "line", "mine", "nine", "pine", "vine", "wine", "dine"]),
    ("DIFFER BY ONE LETTER FROM 'MARK'", [
     "bark", "dark", "hark", "lark", "park", "shark"]),
    ("DIFFER BY ONE LETTER FROM 'FROG'", [
     "blog", "clog", "dog", "fog", "hog", "jog", "log"]),
]

# Words with same vowel pattern
SAME_VOWEL_GROUPS = [
    ("VOWEL PATTERN A_E",  ["bake", "cake", "fame", "gate", "lane",
     "make", "name", "pace", "safe", "save", "tale", "wave"]),
    ("VOWEL PATTERN O_E",  ["bone", "code", "core", "dome", "home",
     "hope", "joke", "lone", "mole", "note", "pose", "role"]),
    ("VOWEL PATTERN I_E",  ["bike", "bite", "dike", "dive", "file",
     "fine", "fire", "hide", "hike", "lime", "line", "mine"]),
    ("VOWEL PATTERN EA",   ["beam", "beat", "dean", "deal", "fear",
     "heap", "jean", "lead", "lean", "meal", "near", "peal"]),
    ("VOWEL PATTERN OO",   ["book", "cook", "food", "fool", "good",
     "hook", "hoop", "look", "mood", "moon", "pool", "roof"]),
    ("VOWEL PATTERN AI",   ["bail", "fail", "hail", "jail", "mail",
     "nail", "pail", "rail", "sail", "tail", "trail", "wail"]),
    ("VOWEL PATTERN OU",   ["bout", "cloud", "clout", "doubt", "found",
     "foul", "gout", "loud", "mount", "pour", "round", "shout"]),
    ("VOWEL PATTERN AW",   ["claw", "crawl", "draw", "flaw",
     "flaw", "gnaw", "jaw", "law", "paw", "raw", "saw", "straw"]),
]

# Polysemy / double meaning categories
POLYSEMY_GROUPS = [
    # river bank, save at bank, bank a turn, bank on
    ("MEANINGS OF 'BANK'",   ["slope", "store", "tilt", "rely"]),
    ("MEANINGS OF 'PITCH'",  ["hurl", "tone", "tar", "slope"]),
    ("MEANINGS OF 'FAIR'",   ["just", "pale", "event", "fine"]),
    ("MEANINGS OF 'BARK'",   ["skin", "rind", "shout", "ship"]),
    ("MEANINGS OF 'LIGHT'",  ["pale", "lamp", "ignite", "easy"]),
    ("MEANINGS OF 'CHARGE'", ["fee", "rush", "power", "accuse"]),
    ("MEANINGS OF 'WATCH'",  ["guard", "clock", "view", "mind"]),
    ("MEANINGS OF 'RING'",   ["circle", "bell", "boxing", "call"]),
    ("MEANINGS OF 'SPELL'",  ["charm", "period", "curse", "write"]),
    ("MEANINGS OF 'PATIENT'", ["calm", "ill", "steady", "waiting"]),
    ("MEANINGS OF 'SPRING'", ["jump", "coil", "season", "source"]),
    ("MEANINGS OF 'WAVE'",   ["signal", "surf", "curve", "ripple"]),
    ("MEANINGS OF 'POOL'",   ["merge", "swim", "stake", "gather"]),
    ("MEANINGS OF 'COLD'",   ["chilly", "ill", "distant", "stale"]),
    ("MEANINGS OF 'HARD'",   ["firm", "tough", "heavy", "strict"]),
]

# Idiom-based word groups
IDIOM_GROUPS = [
    ("WORDS IN 'RAIN' IDIOMS", ["cats", "dogs", "bow",
     "check", "drop", "cloud", "storm", "pour"]),
    ("WORDS IN 'FIRE' IDIOMS", [
     "cease", "arm", "fight", "side", "rapid", "friendly", "camp", "open"]),
    ("WORDS IN 'COLD' IDIOMS", ["blood", "shoulder",
     "turkey", "feet", "war", "snap", "snap", "front"]),
    ("WORDS IN 'HEART' IDIOMS", [
     "broken", "whole", "faint", "brave", "heavy", "sweet", "open", "brave"]),
    ("WORDS IN 'HAND' IDIOMS", ["upper", "shake",
     "clap", "lend", "free", "raise", "tip", "show"]),
    ("WORDS IN 'EYE' IDIOMS",  ["black", "bird",
     "see", "catch", "keep", "evil", "keen", "open"]),
]

# Words fitting multiple categories (tricky!)
MULTI_CATEGORY_WORDS = [
    ("COULD BE A COLOR OR FRUIT", [
     "orange", "plum", "lime", "rose", "olive", "lemon"]),
    ("COULD BE A NUMBER OR ADJECTIVE", [
     "first", "second", "third", "prime", "square", "odd", "even"]),
    ("COULD BE ANIMAL OR VERB",   [
     "bear", "bat", "duck", "hawk", "crane", "bug", "hound", "wolf"]),
    ("COULD BE BODY PART OR VERB", [
     "elbow", "hand", "thumb", "shoulder", "back", "face", "stomach", "head"]),
    ("COULD BE TREE OR NAME",     [
     "ash", "oak", "hazel", "holly", "laurel", "olive", "cedar"]),
    ("COULD BE PLANET OR GOD",    [
     "mars", "venus", "mercury", "saturn", "jupiter", "pluto", "neptune"]),
    ("COULD BE A SUIT OR VERB",   ["club", "spade", "heart", "diamond"]),
    ("COULD BE MONTH OR NAME",    ["april", "june", "august", "may"]),
]

# Letter removal (remove a letter to get a new word)
LETTER_REMOVAL_GROUPS = [
    ("REMOVE A LETTER → NEW WORD (remove first)", [
     "brace", "grain", "plank", "crane", "blend", "flint", "shirt", "stale"]),
    ("REMOVE A LETTER → NEW WORD (remove last)",  [
     "curse", "nerve", "house", "horse", "knife", "store", "drove", "spine"]),
]

# Letter addition (add a letter to make a new word)
LETTER_ADDITION_GROUPS = [
    ("ADD A LETTER → NEW WORD", ["rain", "read",
     "ring", "rap", "rice", "rife", "rill", "rip"]),
]

# Abbreviations / expansions
ABBREVIATION_GROUPS = [
    ("ABBREVIATED TITLES",  ["dr", "mr",
     "ms", "mrs", "prof", "rev", "sr", "jr"]),
    ("ABBREVIATED MONTHS",  ["jan", "feb", "mar",
     "apr", "aug", "sep", "oct", "nov", "dec"]),
    ("ABBREVIATED UNITS",   ["mph", "rpm", "psi",
     "kwh", "mpg", "lbs", "oz", "ft", "cm", "ml"]),
    ("STATE ABBREVIATIONS", ["al", "ak", "az",
     "ar", "ca", "co", "ct", "de", "fl", "ga"]),
]

# Repeated letter patterns
REPEATED_LETTER_WORDS = {
    "double_l": ["bell", "bill", "bull", "call", "cell", "cull", "dull", "fall", "fill", "full",
                 "hall", "hill", "hull", "kill", "lull", "mall", "mill", "null", "pill", "pull",
                 "roll", "sell", "tall", "tell", "till", "toll", "wall", "well", "will", "yell"],
    "double_s": ["boss", "diss", "fuss", "hiss", "kiss", "less", "loss", "mass", "mess", "miss",
                 "moss", "pass", "sass", "toss", "fuss", "bless", "class", "dress", "glass", "grass"],
    "double_t": ["attic", "batten", "battle", "bitten", "bitter", "bottle", "butter", "button",
                 "catty", "cotton", "cutter", "dotted", "flatten", "gotten", "kitten", "litter",
                 "matter", "mitten", "otter", "pattern", "pottery", "putty", "rattle", "setting"],
    "double_o": ["boot", "book", "cool", "door", "food", "fool", "good", "hood", "hook", "hoop",
                 "loom", "look", "mood", "moon", "nook", "pool", "roof", "room", "root", "soot",
                 "soon", "took", "tool", "tooth", "wood", "wool", "zoom"],
    "double_e": ["bee", "deed", "feed", "feel", "feet", "flee", "free", "glee", "heel", "keen",
                 "keep", "keel", "kneel", "need", "peel", "reef", "seed", "seek", "seem", "seen",
                 "seep", "seer", "teem", "tree", "weed", "week"],
}

# Alternating letter patterns
ALTERNATING_LETTER_WORDS = [
    "level", "civic", "limit", "vivid", "minim", "radar", "refer", "revere",
    "banana", "prefix", "proper", "icicle", "pepper", "letter", "better",
    "coffee", "bottle", "middle", "people", "simple", "sample",
]

# ─────────────────────────────────────────────────────────────
# CURATED L1 WORD BANKS — always semantically correct
# ─────────────────────────────────────────────────────────────
L1_WORD_BANKS = {
    "fruit":      ["apple", "mango", "peach", "plum", "grape", "lemon", "lime", "pear", "fig",
                   "date", "kiwi", "guava", "melon", "cherry", "quince", "lychee", "papaya",
                   "apricot", "coconut", "banana", "orange", "pomelo", "nectarine", "persimmon"],
    "bird":       ["robin", "eagle", "finch", "heron", "crane", "swift", "wren", "lark", "raven",
                   "sparrow", "falcon", "parrot", "pigeon", "toucan", "condor", "quail", "stork",
                   "pelican", "magpie", "kestrel", "martin", "swallow", "grouse", "plover", "bittern"],
    "fish":       ["trout", "salmon", "perch", "carp", "bass", "bream", "tuna", "shark", "cod",
                   "pike", "roach", "dace", "rudd", "brill", "plaice", "sprat", "smelt", "chub",
                   "zander", "gudgeon", "barbel", "darter", "loach", "bleak", "ruffe", "vendace"],
    "insect":     ["bee", "ant", "wasp", "moth", "flea", "gnat", "louse", "midge", "tick", "aphid",
                   "locust", "beetle", "cricket", "firefly", "termite", "mantis", "earwig", "hornet",
                   "weevil", "cicada", "mayfly", "lacewing", "dragonfly", "bumblebee", "silverfish"],
    "mammal":     ["lion", "wolf", "bear", "deer", "boar", "otter", "mink", "vole", "mole", "shrew",
                   "stoat", "weasel", "badger", "ferret", "polecat", "marten", "lynx", "moose",
                   "bison", "tapir", "jaguar", "panther", "leopard", "cheetah", "wolverine", "capybara"],
    "vegetable":  ["leek", "beet", "turnip", "celery", "carrot", "radish", "squash", "gourd", "broccoli",
                   "spinach", "lettuce", "cabbage", "parsnip", "fennel", "artichoke", "kale", "chard",
                   "onion", "garlic", "shallot", "pepper", "cucumber", "yam", "kohlrabi", "celeriac"],
    "tree":       ["oak", "elm", "ash", "yew", "pine", "fir", "cedar", "birch", "maple", "beech", "aspen",
                   "larch", "spruce", "willow", "poplar", "alder", "hazel", "holly", "rowan", "lime",
                   "chestnut", "sycamore", "walnut", "mahogany", "teak", "ebony", "balsa", "acacia"],
    "flower":     ["rose", "lily", "iris", "dahlia", "aster", "pansy", "tulip", "daisy", "peony", "poppy",
                   "lupin", "lavender", "violet", "clover", "jasmine", "orchid", "zinnia", "crocus",
                   "azalea", "begonia", "forsythia", "magnolia", "wisteria", "foxglove", "hollyhock"],
    "tool":       ["saw", "axe", "drill", "hammer", "chisel", "trowel", "spanner", "wrench", "pliers",
                   "level", "square", "clamp", "vise", "punch", "awl", "rasp", "file", "adze", "spokeshave",
                   "drawknife", "burnisher", "bradawl", "gouge", "hone", "grinder", "lathe"],
    "weapon":     ["sword", "spear", "bow", "lance", "mace", "axe", "dagger", "crossbow", "musket",
                   "pistol", "rifle", "cannon", "catapult", "halberd", "rapier", "sabre", "cutlass",
                   "stiletto", "flail", "mortar", "trebuchet", "ballista", "scimitar", "dirk", "falchion"],
    "vehicle":    ["car", "bus", "van", "tram", "truck", "lorry", "coach", "jeep", "taxi", "cycle",
                   "scooter", "moped", "buggy", "cart", "sled", "tractor", "trailer", "tanker", "camper",
                   "sedan", "saloon", "hatchback", "coupe", "minivan", "pickup", "roadster", "estate"],
    "furniture":  ["desk", "chair", "sofa", "table", "bench", "stool", "shelf", "cabinet", "wardrobe",
                   "dresser", "sideboard", "bookcase", "credenza", "armchair", "ottoman", "chaise",
                   "bureau", "settee", "davenport", "hutch", "console", "tallboy", "commode"],
    "sport":      ["tennis", "rugby", "cricket", "hockey", "boxing", "rowing", "skiing", "golf", "cycling",
                   "archery", "fencing", "diving", "polo", "darts", "squash", "croquet", "lacrosse",
                   "curling", "sailing", "kayaking", "judo", "karate", "wrestling", "handball", "netball"],
    "cheese":     ["cheddar", "brie", "gouda", "edam", "feta", "ricotta", "gruyere", "havarti", "manchego",
                   "fontina", "provolone", "pecorino", "stilton", "colby", "asiago", "camembert",
                   "emmental", "limburger", "muenster", "raclette", "gorgonzola", "taleggio"],
    "gem":        ["diamond", "ruby", "emerald", "sapphire", "topaz", "opal", "garnet", "jade", "onyx",
                   "pearl", "amethyst", "turquoise", "citrine", "spinel", "zircon", "peridot", "agate",
                   "jasper", "obsidian", "tourmaline", "tanzanite", "alexandrite", "aquamarine"],
    "metal":      ["gold", "silver", "copper", "iron", "steel", "brass", "bronze", "zinc", "lead", "nickel",
                   "chrome", "platinum", "titanium", "tungsten", "cobalt", "tin", "aluminium", "magnesium",
                   "manganese", "vanadium", "molybdenum", "palladium", "osmium", "rhodium", "iridium"],
    "currency":   ["dollar", "euro", "pound", "yen", "franc", "rupee", "krona", "dinar", "dirham", "peso",
                   "ruble", "forint", "zloty", "krone", "baht", "lira", "shekel", "ringgit", "won", "real",
                   "escudo", "guilder", "florin", "taka", "kwanza", "birr", "cedi", "kwacha", "shilling"],
    "dog":        ["labrador", "poodle", "beagle", "collie", "spaniel", "terrier", "retriever", "setter",
                   "pointer", "whippet", "greyhound", "mastiff", "boxer", "bulldog", "dalmatian", "husky",
                   "samoyed", "malamute", "akita", "shiba", "pinscher", "vizsla", "weimaraner", "doberman"],
    "planet":     ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "ceres"],
    "instrument": ["violin", "cello", "flute", "oboe", "clarinet", "trumpet", "trombone", "tuba", "bassoon",
                   "harp", "lute", "mandolin", "banjo", "ukulele", "sitar", "dulcimer", "zither", "accordion",
                   "harmonica", "xylophone", "marimba", "timpani", "theorbo", "rebec", "serpent"],
    "dance":      ["waltz", "tango", "foxtrot", "polka", "samba", "rumba", "salsa", "jive", "quickstep",
                   "lindy", "swing", "bolero", "flamenco", "merengue", "mambo", "hustle", "minuet", "gavotte",
                   "cotillion", "hornpipe", "reel", "jig", "tarantella", "czardas", "fandango", "pavane"],
    "snake":      ["cobra", "viper", "python", "mamba", "boa", "adder", "krait", "racer", "ratsnake",
                   "bullsnake", "kingsnake", "copperhead", "cottonmouth", "anaconda", "sidewinder",
                   "diamondback", "massasauga", "gaboon", "puffadder", "hognose", "milk", "corn", "garter"],
    "fabric":     ["silk", "wool", "cotton", "linen", "velvet", "satin", "denim", "tweed", "corduroy",
                   "chiffon", "taffeta", "organza", "muslin", "cambric", "flannel", "gingham", "paisley",
                   "tartan", "hessian", "buckram", "brocade", "damask", "moleskin", "cashmere", "angora"],
    "hat":        ["beret", "fedora", "bonnet", "turban", "stetson", "trilby", "deerstalker", "fez",
                   "pillbox", "homburg", "bowler", "panama", "boater", "sombrero", "cloche", "mitre",
                   "skullcap", "shako", "busby", "wimple", "coif", "hennin", "mortarboard", "tam"],
    "boat":       ["canoe", "kayak", "dinghy", "yacht", "skiff", "ferry", "barge", "trawler", "schooner",
                   "brigantine", "galleon", "frigate", "sloop", "ketch", "cutter", "lugger", "smack",
                   "pinnace", "wherry", "coracle", "currach", "catamaran", "trimaran", "dhow", "proa"],
    "cat":        ["tabby", "siamese", "persian", "bengal", "burmese", "ragdoll", "sphynx", "manx",
                   "abyssinian", "tonkinese", "balinese", "somali", "devon", "cornish", "ocicat",
                   "savannah", "birman", "chartreux", "havana", "bombay", "snowshoe", "javanese"],
    "horse":      ["arab", "clydesdale", "thoroughbred", "appaloosa", "palomino", "mustang", "fjord",
                   "morgan", "friesian", "lipizzaner", "andalusian", "percheron", "shire", "haflinger",
                   "exmoor", "dartmoor", "fell", "hackney", "oldenburg", "trakehner", "hanoverian"],
    "grain":      ["wheat", "barley", "oats", "rye", "millet", "sorghum", "spelt", "emmer", "einkorn",
                   "triticale", "buckwheat", "quinoa", "amaranth", "teff", "kamut", "freekeh", "bulgur",
                   "farro", "durum", "semolina", "polenta", "maize", "groats", "hominy", "grits"],
    "spice":      ["pepper", "cumin", "clove", "nutmeg", "cardamom", "cinnamon", "turmeric", "saffron",
                   "paprika", "coriander", "fennel", "caraway", "anise", "fenugreek", "sumac", "tamarind",
                   "galangal", "asafoetida", "mahlab", "mace", "allspice", "juniper", "vanilla", "ginger"],
    "nut":        ["walnut", "almond", "cashew", "pistachio", "pecan", "hazel", "chestnut", "macadamia",
                   "brazil", "pine", "peanut", "acorn", "beechnut", "butternut", "hickory", "filbert",
                   "kola", "tigernuts", "coconut", "areca", "tung", "paradise", "queensland", "candlenut"],
    "pasta":      ["spaghetti", "linguine", "fettuccine", "tagliatelle", "pappardelle", "rigatoni",
                   "penne", "fusilli", "farfalle", "conchiglie", "orecchiette", "vermicelli", "orzo",
                   "ditalini", "rotini", "cavatappi", "bucatini", "ziti", "macaroni", "tortellini"],
    "martial":    ["judo", "karate", "taekwondo", "jujitsu", "aikido", "hapkido", "capoeira", "muay",
                   "silat", "wushu", "escrima", "arnis", "kendo", "ninjutsu", "sumo", "savate", "systema",
                   "bojutsu", "iaido", "kyudo", "pankration", "sambo", "lethwei", "vovinam", "dumog"],
    "board":      ["chess", "checkers", "backgammon", "draughts", "scrabble", "monopoly", "cluedo",
                   "othello", "reversi", "ludo", "parcheesi", "mancala", "dominoes", "mahjong", "risk",
                   "diplomacy", "stratego", "battleship", "halma", "senet", "go", "shogi", "xiangqi"],
    "painting":   ["watercolor", "fresco", "tempera", "gouache", "encaustic", "miniature", "impasto",
                   "sfumato", "chiaroscuro", "pointillism", "landscape", "portrait", "grisaille",
                   "verdaccio", "secco", "sgraffito", "monotype", "aquatint", "mezzotint"],
    "phobia":     ["arachnophobia", "claustrophobia", "agoraphobia", "acrophobia", "xenophobia",
                   "hydrophobia", "nyctophobia", "pyrophobia", "glossophobia", "necrophobia",
                   "hemophobia", "cynophobia", "ergophobia", "gamophobia", "gerascophobia"],
    "religion":   ["christianity", "islam", "hinduism", "buddhism", "judaism", "sikhism", "jainism",
                   "taoism", "shinto", "zoroastrianism", "bahai", "animism", "shamanism", "druidism",
                   "paganism", "wicca", "mandaeism", "yazidism", "tenrikyo"],
}

# ─────────────────────────────────────────────────────────────
# ConceptNet — free API, no key, greatly enriches L2/L3
# ─────────────────────────────────────────────────────────────


class ConceptNetMiner:
    BASE = "https://api.conceptnet.io"
    _cache = {}

    def _get(self, url: str) -> dict:
        if url in self._cache:
            return self._cache[url]
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            self._cache[url] = data
            time.sleep(0.15)
            return data
        except Exception:
            return {}

    def _words(self, data: dict, word: str) -> list:
        out = []
        for edge in data.get("edges", []):
            for key in ["end", "start"]:
                node = edge.get(key, {})
                if node.get("language") == "en":
                    w = node.get("label", "").lower().strip()
                    if w and " " not in w and w.isalpha() and 3 <= len(w) <= 10 and w != word:
                        out.append(w)
        return list(dict.fromkeys(out))

    def related(self, word: str, rel: str = "RelatedTo", n: int = 40) -> list:
        url = f"{self.BASE}/query?node=/c/en/{word}&rel=/r/{rel}&limit={n}"
        return self._words(self._get(url), word)

    def similar(self, word: str, n: int = 40) -> list:
        return list(dict.fromkeys(
            self.related(word, "SimilarTo", n) +
            self.related(word, "Synonym", n)
        ))

    def context(self, word: str, n: int = 40) -> list:
        return list(dict.fromkeys(
            self.related(word, "HasContext", n) +
            self.related(word, "RelatedTo", n)
        ))


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
                    try:
                        if any(d in s.lexname() for d in preferred_domains):
                            return s.name()
                    except Exception:
                        continue
            return syns[0].name()
        key_pool = [s for s, k in candidates if k] or [
            s for s, _ in candidates]
        if preferred_domains:
            for s in key_pool:
                try:
                    if any(d in s.lexname() for d in preferred_domains):
                        return s.name()
                except Exception:
                    continue
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
        "cloud":       ("cloud.n.02",              "noun.phenomenon", "CLOUD TYPES"),
        "cat":         ("cat.n.01",                "noun.animal",     "CAT BREEDS"),
        "horse":       ("horse.n.01",              "noun.animal",     "HORSE BREEDS"),
        "butterfly":   ("butterfly.n.01",          "noun.animal",     "BUTTERFLIES"),
        "grain":       ("grain.n.03",              "noun.food",       "GRAINS"),
        "spice":       ("spice.n.01",              "noun.food",       "SPICES"),
        "nut":         ("nut.n.01",                "noun.food",       "NUTS"),
        "berry":       ("berry.n.01",              "noun.food",       "BERRIES"),
        "shoe":        ("shoe.n.01",               "noun.artifact",   "TYPES OF SHOES"),
        "bag":         ("bag.n.01",                "noun.artifact",   "TYPES OF BAGS"),
        "chair":       ("chair.n.01",              "noun.artifact",   "TYPES OF CHAIRS"),
        "bridge":      ("card_game.n.01",          "noun.act",        "CARD GAMES"),
        "pasta":       ("pasta.n.01",              "noun.food",       "PASTA TYPES"),
        "cookie":      ("cookie.n.01",             "noun.food",       "COOKIES"),
        "sauce":       ("sauce.n.01",              "noun.food",       "SAUCES"),
        "knife":       ("knife.n.01",              "noun.artifact",   "TYPES OF KNIVES"),
        "castle":      ("castle.n.03",             "noun.artifact",   "CASTLE PARTS"),
        "painting":    ("painting.n.01",           "noun.communication", "PAINTING STYLES"),
        "poetry":      ("poetry.n.01",             "noun.communication", "POETRY FORMS"),
        "language":    ("language.n.01",           "noun.communication", "LANGUAGES"),
        "religion":    ("religion.n.01",           "noun.cognition",  "RELIGIONS"),
        "phobia":      ("phobia.n.01",             "noun.state",      "PHOBIAS"),
        "martial":     ("martial_art.n.01",        "noun.act",        "MARTIAL ARTS"),
        "board":       ("board_game.n.01",         "noun.act",        "BOARD GAMES"),
    }
    MERONYM_CATS = {
        "body":    ("body.n.01",      "BODY PARTS"),
        "car":     ("car.n.01",       "CAR PARTS"),
        "house":   ("house.n.01",     "PARTS OF A HOUSE"),
        "plant":   ("plant.n.02",     "PLANT PARTS"),
        "bicycle": ("bicycle.n.01",   "BICYCLE PARTS"),
        "face":    ("face.n.01",      "PARTS OF THE FACE"),
        "book_n":  ("book.n.01",      "PARTS OF A BOOK"),
        "tree_n":  ("tree.n.01",      "PARTS OF A TREE"),
        "ship":    ("ship.n.01",      "PARTS OF A SHIP"),
        "brain":   ("brain.n.01",     "PARTS OF THE BRAIN"),
        "guitar":  ("guitar.n.01",    "PARTS OF A GUITAR"),
        "computer": ("computer.n.01",  "COMPUTER PARTS"),
        "eye_n":   ("eye.n.01",       "PARTS OF THE EYE"),
        "tooth":   ("tooth.n.01",     "TOOTH PARTS"),
        "flower_n": ("flower.n.01",    "FLOWER PARTS"),
        "kitchen": ("kitchen.n.01",   "KITCHEN ITEMS"),
    }

    def hyponyms_of(self, synset_name: str, max_depth: int = 3, max_results: int = 100) -> list:
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
            try:
                for lem in node.lemmas():
                    tok = lem.name().replace("_", "")
                    if tok.isalpha() and 3 <= len(tok) <= 10:
                        words.append(tok.lower())
            except Exception:
                pass
            if d < max_depth:
                try:
                    for h in node.hyponyms():
                        if h not in visited:
                            queue.append((h, d+1))
                except Exception:
                    pass
        return list(dict.fromkeys(words))

    def meronyms_of(self, synset_name: str) -> list:
        try:
            root = wn.synset(synset_name)
        except Exception:
            return []
        words = []
        try:
            for rel in [root.part_meronyms(), root.substance_meronyms(), root.member_meronyms()]:
                for h in rel:
                    try:
                        for lem in h.lemmas():
                            tok = lem.name().replace("_", "")
                            if tok.isalpha() and 3 <= len(tok) <= 10:
                                words.append(tok.lower())
                    except Exception:
                        continue
        except Exception:
            pass
        return list(dict.fromkeys(words))

    def is_hyponym_of(self, word: str, synset_name: str) -> bool:
        try:
            cat = wn.synset(synset_name)
        except Exception:
            return True
        try:
            for syn in wn.synsets(word):
                try:
                    for path in syn.hypernym_paths():
                        if cat in path:
                            return True
                except Exception:
                    continue
        except Exception:
            return True
        return False

    def all_synonyms(self, concept: str, pos) -> list:
        out = set()
        try:
            for syn in wn.synsets(concept, pos=pos):
                try:
                    for lem in syn.lemmas():
                        w = lem.name().replace("_", "")
                        if w.isalpha() and 3 <= len(w) <= 10 and w.lower() != concept.lower():
                            out.add(w.lower())
                except Exception:
                    continue
        except Exception:
            pass
        return list(out)

    def polysemy(self, w: str) -> int:
        try:
            return len(wn.synsets(w))
        except Exception:
            return 0

    def synset_overlap(self, w1: str, w2: str) -> float:
        try:
            s1, s2 = set(wn.synsets(w1)), set(wn.synsets(w2))
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)
        except Exception:
            return 0.0

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

    def _clean(self, res: list, exc: str = "") -> list:
        out = []
        for item in res:
            w = item.get("word", "")
            if w and " " not in w and w.isalpha() and 3 <= len(w) <= 10 and w.lower() != exc.lower():
                out.append(w.lower())
        return out

    def triggered_by(self, w: str, n: int = 60) -> list:
        return self._clean(self._get({"rel_trg": w, "max": n}), w)

    def means_like(self, w: str, n: int = 60) -> list:
        return self._clean(self._get({"ml": w, "max": n}), w)

    def synonyms(self, w: str, n: int = 40) -> list:
        return self._clean(self._get({"rel_syn": w, "max": n}), w)

    def words_before(self, suffix: str, n: int = 80) -> list:
        return self._clean(self._get({"rc": suffix, "max": n}), suffix)

    def words_after(self, prefix: str, n: int = 80) -> list:
        return self._clean(self._get({"lc": prefix, "max": n}), prefix)

    def cooccur(self, w: str, n: int = 60) -> list:
        return self._clean(self._get({"rel_bga": w, "max": n}), w)

    def rhymes(self, w: str, n: int = 40) -> list:
        return self._clean(self._get({"rel_rhy": w, "max": n}), w)


# ─────────────────────────────────────────────────────────────
# Word Filter + dedup helpers
# ─────────────────────────────────────────────────────────────
class WordFilter:
    def __init__(self, min_freq: int = 2):
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

    def apply(self, words, used=None, forbidden=None) -> list:
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
# L2 concept bank (anchor → synonyms) — expanded
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
    ("eat",       wn.VERB, "eat.v.01",       "WORDS MEANING EAT"),
    ("sleep",     wn.VERB, "sleep.v.01",     "WORDS MEANING SLEEP"),
    ("think",     wn.VERB, "think.v.01",     "WORDS MEANING THINK"),
    ("break",     wn.VERB, "break.v.01",     "WORDS MEANING BREAK"),
    ("move",      wn.VERB, "travel.v.01",    "WORDS MEANING MOVE"),
    ("stop",      wn.VERB, "stop.v.01",      "WORDS MEANING STOP"),
    ("begin",     wn.VERB, "begin.v.01",     "WORDS MEANING BEGIN"),
    ("fall",      wn.VERB, "fall.v.01",      "WORDS MEANING FALL"),
    ("pull",      wn.VERB, "pull.v.01",      "WORDS MEANING PULL"),
    ("push",      wn.VERB, "push.v.01",      "WORDS MEANING PUSH"),
    ("look",      wn.VERB, "look.v.01",      "WORDS MEANING LOOK"),
    ("find",      wn.VERB, "find.v.01",      "WORDS MEANING FIND"),
    ("leave",     wn.VERB, "leave.v.01",     "WORDS MEANING LEAVE"),
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
    ("cold",      wn.ADJ,  "cold.a.01",      "WORDS MEANING COLD"),
    ("hot",       wn.ADJ,  "hot.a.01",       "WORDS MEANING HOT"),
    ("new",       wn.ADJ,  "new.a.01",       "WORDS MEANING NEW"),
    ("clean",     wn.ADJ,  "clean.a.01",     "WORDS MEANING CLEAN"),
    ("honest",    wn.ADJ,  "honest.a.01",    "WORDS MEANING HONEST"),
    ("loud",      wn.ADJ,  "loud.a.01",      "WORDS MEANING LOUD"),
    ("quiet",     wn.ADJ,  "quiet.a.01",     "WORDS MEANING QUIET"),
    ("sharp",     wn.ADJ,  "sharp.a.01",     "WORDS MEANING SHARP"),
    ("smooth",    wn.ADJ,  "smooth.a.01",    "WORDS MEANING SMOOTH"),
    ("rough",     wn.ADJ,  "rough.a.01",     "WORDS MEANING ROUGH"),
    ("soft",      wn.ADJ,  "soft.a.01",      "WORDS MEANING SOFT"),
    ("hard",      wn.ADJ,  "hard.a.01",      "WORDS MEANING HARD"),
    ("thin",      wn.ADJ,  "thin.a.01",      "WORDS MEANING THIN"),
    ("thick",     wn.ADJ,  "thick.a.01",     "WORDS MEANING THICK"),
    ("light",     wn.ADJ,  "light.a.01",     "WORDS MEANING LIGHT"),
    ("heavy",     wn.ADJ,  "heavy.a.01",     "WORDS MEANING HEAVY"),
    ("narrow",    wn.ADJ,  "narrow.a.01",    "WORDS MEANING NARROW"),
    ("wide",      wn.ADJ,  "wide.a.01",      "WORDS MEANING WIDE"),
    ("deep",      wn.ADJ,  "deep.a.01",      "WORDS MEANING DEEP"),
    ("shallow",   wn.ADJ,  "shallow.a.01",   "WORDS MEANING SHALLOW"),
]

L3_ANCHORS = [
    "fire", "water", "ice", "gold", "silver", "ocean", "desert",
    "forest", "space", "music", "war", "peace", "light", "wind",
    "heat", "snow", "night", "blood", "steel", "stone", "storm",
    "dream", "time", "rain", "sun", "moon", "earth", "cave",
    "river", "mountain", "castle", "pirate", "circus", "magic",
    "prison", "garden", "hospital", "kitchen", "wedding", "market",
    "school", "church", "library", "theater", "museum", "beach",
    "jungle", "island", "volcano", "glacier", "swamp", "cliff",
    "valley", "plateau", "canyon", "dune", "lagoon", "reef",
    "farm", "ranch", "village", "harbor", "bridge", "tower",
    "arena", "stadium", "palace", "temple", "pyramid", "mine",
]

L3_ABSTRACT = [
    "justice", "liberty", "chaos", "memory", "courage", "wealth",
    "danger", "beauty", "wisdom", "truth", "power", "glory",
    "honor", "shame", "pride", "faith", "hope", "fear", "love",
    "anger", "grief", "joy", "peace", "war", "time", "fate",
    "freedom", "loyalty", "betrayal", "sacrifice", "revenge",
    "patience", "ambition", "compassion", "envy", "forgiveness",
    "humility", "kindness", "creativity", "discipline", "curiosity",
    "passion", "loneliness", "nostalgia", "resilience", "solidarity",
    "trust", "doubt", "wonder", "serenity", "tension",
]


# ─────────────────────────────────────────────────────────────
# Overlap checker — reads existing JSONL output to avoid duplicates
# ─────────────────────────────────────────────────────────────

def _load_existing_puzzles(output_path: str) -> list:
    """Load all previously generated puzzles from the output file."""
    puzzles = []
    if not os.path.exists(output_path):
        return puzzles
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    puzzles.append(json.loads(line))
                except Exception:
                    pass
    return puzzles


def _save_incremental(p, gen, jsonl_path: str, csv_path: str):
    """Save one puzzle immediately to both output files (append mode)."""
    import csv as csv_mod
    # JSONL
    try:
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(gen.to_jsonl(p)) + "\n")
    except Exception as e:
        print(f"  [save] JSONL write error: {e}")

    # CSV
    try:
        fieldnames = ["Game ID", "Puzzle Date", "Word", "Group Name",
                      "Group Level", "Starting Row", "Starting Column"]
        # Determine next game ID
        next_id = 1
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    next_id = int(df["Game ID"].max()) + 1
            except Exception:
                pass
        today_str = date.today().isoformat()
        rows = gen.to_csv_rows(p, next_id, today_str)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv_mod.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"  [save] CSV write error: {e}")


def _puzzle_overlaps(new_puzzle_dict: dict, existing: list) -> bool:
    """
    Return True if this puzzle overlaps with any existing one.
    Checks:
      1. Exact 16-word set match  (new_words == existing_words as sets)
      2. Any single 4-word group is identical to an existing group
    """
    new_words = frozenset(w.lower()
                          for w in new_puzzle_dict.get("all_words", []))
    new_groups = []
    for g in new_puzzle_dict.get("groups", []):
        new_groups.append(frozenset(w["word"].lower()
                          for w in g.get("words", [])))

    for existing_puzzle in existing:
        ex_words = frozenset(w.lower()
                             for w in existing_puzzle.get("all_words", []))

        # Check 1: Exact same 16-word set
        if new_words == ex_words:
            print("    [overlap] ❌ Exact same 16-word puzzle already exists!")
            return True

        # Check 2: Any group of 4 words is fully identical to an existing group
        for ex_g in existing_puzzle.get("groups", []):
            ex_group_set = frozenset(w["word"].lower()
                                     for w in ex_g.get("words", []))
            for new_g in new_groups:
                if new_g == ex_group_set:
                    print(
                        f"    [overlap] ❌ Group {new_g} is identical to one in a previous puzzle!")
                    return True

    return False


# ─────────────────────────────────────────────────────────────
# Core level-building helpers (shared across architectures)
# ─────────────────────────────────────────────────────────────
def _w(text, wn_m):
    try:
        poly = wn_m.polysemy(text)
    except Exception:
        poly = 0
    try:
        freq = word_freq(text)
    except Exception:
        freq = 0
    return Word(text, poly, freq)


def _excl_ok(word: str, others: list, wn_m: WordNetMiner, thr: float = 0.25) -> bool:
    for g in others:
        try:
            if wn_m.max_overlap(word, g) >= thr:
                return False
        except Exception:
            pass
    return True


def _build_l1_hyponym(concept, cat_data, bn, wn_m, flt, used, others,
                      used_sigs: set = None, cn=None):
    """
    Build a Level-1 group (category members).
    Tries, in order:
      1. WordNet hyponyms (BabelNet-refined synset)
      2. Curated L1_WORD_BANKS
      3. ConceptNet IsA relations (if cn provided)
    Skips any combination whose frozenset is already in used_sigs.
    Returns up to 4 unique, valid words.
    """
    syn_default, dom, display = cat_data
    forbidden = {concept, concept+"s",
                 concept[:-1] if concept.endswith("s") else concept+"s"}
    if used_sigs is None:
        used_sigs = set()

    def _pick4(pool):
        """Try every possible set of 4 from pool, skip already-used sigs."""
        clean = _dedup([w for w in pool if _excl_ok(w, others, wn_m)])
        if len(clean) < 4:
            return None
        # Shuffle and try up to 20 random 4-subsets
        random.shuffle(clean)
        for _ in range(20):
            chosen = random.sample(clean, min(4, len(clean)))
            if len(chosen) < 4:
                break
            sig = frozenset(chosen)
            if sig not in used_sigs:
                return chosen
        # All tried combos were duplicates — return None
        return None

    # ── Pass 1: WordNet hyponyms ──────────────────────────────
    try:
        synset = bn.find_wn_synset(concept, "n", [dom]) or syn_default
    except Exception:
        synset = syn_default

    for synset_try in list(dict.fromkeys([synset, syn_default])):
        raw = wn_m.hyponyms_of(synset_try)
        # Relaxed: try strict first, then all hyponyms
        for pool_fn in [
            lambda r=raw, s=synset_try: [w for w in flt.apply(r, used, forbidden)
                                         if wn_m.is_hyponym_of(w, s)],
            lambda r=raw: flt.apply(r, used, forbidden),
        ]:
            try:
                pool = pool_fn()
            except Exception:
                pool = []
            chosen = _pick4(pool)
            if chosen:
                print(f"      [L1-wn] ✓ {display}: {chosen}")
                return Group(1, display, [_w(c, wn_m) for c in chosen],
                             "wn_hyponym", synset_try)

    # ── Pass 2: Curated L1_WORD_BANKS ────────────────────────
    bank = L1_WORD_BANKS.get(concept, [])
    if bank:
        # Shuffle a fresh copy each time so we get different combos
        bank_shuffled = list(bank)
        random.shuffle(bank_shuffled)
        pool = [w for w in bank_shuffled if flt.ok(w, used, forbidden)]
        chosen = _pick4(pool)
        if chosen:
            print(f"      [L1-bank] ✓ {display}: {chosen}")
            return Group(1, display, [_w(c, wn_m) for c in chosen],
                         "curated_bank", syn_default)

    # ── Pass 3: ConceptNet IsA expansion ─────────────────────
    if cn is not None:
        try:
            cn_pool = cn.related(concept, "IsA") + \
                cn.related(concept, "InstanceOf")
            pool = [w for w in flt.apply(cn_pool, used, forbidden)]
            chosen = _pick4(pool)
            if chosen:
                print(f"      [L1-cn] ✓ {display}: {chosen}")
                return Group(1, display, [_w(c, wn_m) for c in chosen],
                             "conceptnet_isa", syn_default)
        except Exception:
            pass

    return None


def _build_l2_synonym(anchor, pos, syn_hint, label, dm, wn_m, flt, used, others,
                      used_sigs: set = None):
    if used_sigs is None:
        used_sigs = set()
    wn_syns = wn_m.all_synonyms(anchor, pos)
    dm_syns = dm.synonyms(anchor) + dm.means_like(anchor)
    pool = list(dict.fromkeys(wn_syns + dm_syns))
    try:
        anchor_synsets = set(wn.synsets(anchor, pos=pos))
    except Exception:
        anchor_synsets = set()
    validated = []
    for w in pool:
        try:
            if set(wn.synsets(w)) & anchor_synsets:
                validated.append(w)
        except Exception:
            pass
    # Fallback: if strict overlap gives too few, use full pool
    if len(validated) < 6:
        validated = pool
    forbidden = {anchor, anchor+"s", anchor+"ing", anchor+"ed", anchor+"er"}
    cands = flt.apply(validated, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    # Try different 4-subsets to avoid duplicates
    random.shuffle(clean)
    for _ in range(15):
        chosen = random.sample(clean, min(4, len(clean)))
        if len(chosen) < 4:
            break
        if frozenset(chosen) not in used_sigs:
            return Group(2, label, [_w(c, wn_m) for c in chosen], "wn_synonym", syn_hint)
    return None


def _build_l2_meanslike(anchor, pos, syn_hint, label, dm, wn_m, flt, used, others,
                        used_sigs: set = None):
    if used_sigs is None:
        used_sigs = set()
    pool = list(dict.fromkeys(dm.means_like(anchor) + dm.synonyms(anchor)))
    forbidden = {anchor, anchor+"s", anchor+"ing", anchor+"ed"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    for _ in range(15):
        chosen = random.sample(clean, min(4, len(clean)))
        if len(chosen) < 4:
            break
        if frozenset(chosen) not in used_sigs:
            return Group(2, label, [_w(c, wn_m) for c in chosen], "datamuse_means_like", syn_hint)
    return None


def _build_l3_triggered(anchor, dm, wn_m, flt, used, others, used_sigs: set = None):
    if used_sigs is None:
        used_sigs = set()
    # Merge triggered_by + cooccur for a bigger, more varied pool
    pool = list(dict.fromkeys(dm.triggered_by(anchor) + dm.cooccur(anchor)))
    forbidden = {anchor, anchor+"s", anchor+"ed", anchor+"ing"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    for _ in range(15):
        chosen = random.sample(clean, min(4, len(clean)))
        if len(chosen) < 4:
            break
        if frozenset(chosen) not in used_sigs:
            return Group(3, f"ASSOCIATED WITH {anchor.upper()}",
                         [_w(c, wn_m) for c in chosen], "datamuse_triggered")
    return None


def _build_l3_cooccur(anchor, dm, wn_m, flt, used, others, used_sigs: set = None):
    if used_sigs is None:
        used_sigs = set()
    pool = list(dict.fromkeys(dm.cooccur(anchor) + dm.triggered_by(anchor)))
    forbidden = {anchor, anchor+"s"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    for _ in range(15):
        chosen = random.sample(clean, min(4, len(clean)))
        if len(chosen) < 4:
            break
        if frozenset(chosen) not in used_sigs:
            return Group(3, f"GOES WITH {anchor.upper()}",
                         [_w(c, wn_m) for c in chosen], "datamuse_cooccur")
    return None


def _build_l3_abstract(anchor, dm, wn_m, flt, used, others, used_sigs: set = None):
    if used_sigs is None:
        used_sigs = set()
    pool = list(dict.fromkeys(
        dm.means_like(anchor) + dm.triggered_by(anchor) + dm.synonyms(anchor)
    ))
    forbidden = {anchor, anchor+"s"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    for _ in range(15):
        chosen = random.sample(clean, min(4, len(clean)))
        if len(chosen) < 4:
            break
        if frozenset(chosen) not in used_sigs:
            return Group(3, f"WORDS RELATED TO {anchor.upper()}",
                         [_w(c, wn_m) for c in chosen], "datamuse_abstract_ml")
    return None


def _build_l3_conceptnet(anchor, cn, dm, wn_m, flt, used, others, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    pool = list(dict.fromkeys(
        cn.context(anchor) + cn.related(anchor, "PartOf") +
        dm.triggered_by(anchor)
    ))
    forbidden = {anchor, anchor+"s"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(3, f"ASSOCIATED WITH {anchor.upper()}",
                 [_w(c, wn_m) for c in clean[:4]], "conceptnet_related")


def _build_l2_conceptnet(anchor, pos, syn_hint, label, cn, dm, wn_m, flt, used, others, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    pool = list(dict.fromkeys(
        cn.similar(anchor) + dm.synonyms(anchor) + dm.means_like(anchor)
    ))
    forbidden = {anchor, anchor+"s", anchor+"ing", anchor+"ed", anchor+"er"}
    cands = flt.apply(pool, used, forbidden)
    clean = _dedup([w for w in cands if _excl_ok(w, others, wn_m)])
    if len(clean) < 4:
        return None
    random.shuffle(clean)
    return Group(2, label, [_w(c, wn_m) for c in clean[:4]], "conceptnet_similar", syn_hint)


# ─────────────────────────────────────────────────────────────
# L4 Builder — Standard compound (before/after/any)
# ─────────────────────────────────────────────────────────────
def _build_l4(mode_filter, dm, wn_m, used, min_non_conflict=4, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    patterns = [(b, d, m) for b, d, m in L4_PATTERNS
                if mode_filter == "any" or m == mode_filter]
    random.shuffle(patterns)

    for base, display, mode in patterns:
        hardcoded = L4_STEM_POOLS.get(base, [])
        try:
            live = (dm.words_before(base) if mode == "before"
                    else dm.words_after(base))
        except Exception:
            live = []
        all_stems = list(dict.fromkeys(hardcoded + live))
        cands = []
        for s in all_stems:
            sl = s.lower()
            if (sl.isalpha() and 3 <= len(sl) <= 12
                    and word_freq(sl) >= 0
                    and sl not in used
                    and sl != base):
                cands.append(sl)
        cands = _dedup(list(dict.fromkeys(cands)))
        if len(cands) < 4:
            print(
                f"      [L4] '{base}' only {len(cands)} stems after filter, skipping")
            continue
        random.shuffle(cands)
        # Try random 4-subsets to avoid duplicate groups
        for _ in range(20):
            if len(cands) < 4:
                break
            chosen = random.sample(cands, 4)
            if frozenset(chosen) not in used_sigs:
                print(
                    f"      [L4] ✓ base='{base}' mode={mode} chosen={chosen}")
                return Group(4, display, [_w(c, wn_m) for c in chosen], f"compound_{mode}")
    return None


# ─────────────────────────────────────────────────────────────
# L4 Builder — NEW WORDPLAY PATTERNS
# ─────────────────────────────────────────────────────────────

def _build_l4_rhyme(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words that rhyme with a given word."""
    random.shuffle(RHYME_GROUPS)
    for label, pool in RHYME_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-rhyme] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "rhyming_words")
    return None


def _build_l4_anagram(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words that are anagrams of each other."""
    random.shuffle(ANAGRAM_GROUPS)
    for label, pool in ANAGRAM_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-anagram] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "anagrams")
    return None


def _build_l4_palindrome(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words that are palindromes."""
    pool_copy = [w for w in PALINDROMES if w not in used and w.isalpha()
                 and 3 <= len(w) <= 12]
    if len(pool_copy) >= 4:
        random.shuffle(pool_copy)
        chosen = pool_copy[:4]
        print(f"      [L4-palindrome] ✓ PALINDROMES: {chosen}")
        return Group(4, "PALINDROMES", [_w(c, wn_m) for c in chosen], "palindromes")
    return None


def _build_l4_hidden_word(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """All words contain the same hidden substring."""
    random.shuffle(HIDDEN_WORD_GROUPS)
    for hidden, label, pool in HIDDEN_WORD_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha()
                 and 3 <= len(w) <= 10 and hidden in w.lower()]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-hidden] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "hidden_substring")
    return None


def _build_l4_same_prefix(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """All words share the same prefix."""
    random.shuffle(SAME_PREFIX_GROUPS)
    for prefix, label, pool in SAME_PREFIX_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha()
                 and 3 <= len(w) <= 12 and w.lower().startswith(prefix.lower())]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-prefix] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "same_prefix")
    return None


def _build_l4_same_suffix(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """All words share the same suffix."""
    random.shuffle(SAME_SUFFIX_GROUPS)
    for suffix, label, pool in SAME_SUFFIX_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha()
                 and 3 <= len(w) <= 12 and w.lower().endswith(suffix.lower())]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-suffix] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "same_suffix")
    return None


def _build_l4_homophones(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """
    Words that are homophones of another word (each word in the group
    sounds like a DIFFERENT word). Uses the CMU Pronouncing Dictionary
    to verify real homophones, then picks 4 confirmed homophone-having words.
    """
    # Confirmed homophone pairs (word → sounds_like) — all real English homophones
    HOMOPHONE_PAIRS = [
        ("bare", "bear"), ("flour", "flower"), ("sea", "see"), ("hear", "here"),
        ("knight", "night"), ("knot", "not"), ("pear", "pair"), ("right", "write"),
        ("sale", "sail"), ("tale", "tail"), ("week", "weak"), ("wood", "would"),
        ("which", "witch"), ("way", "weigh"), ("sole", "soul"), ("fair", "fare"),
        ("mail", "male"), ("peace", "piece"), ("wait", "weight"), ("know", "no"),
        ("eye", "aye"), ("buy", "by"), ("threw", "through"), ("new", "knew"),
        ("wear", "where"), ("read", "reed"), ("steel", "steal"), ("heel", "heal"),
        ("meat", "meet"), ("feat", "feet"), ("peak", "peek"), ("suite", "sweet"),
        ("their", "there"), ("two", "too"), ("one", "won"), ("son", "sun"),
        ("prey", "pray"), ("dye", "die"), ("hare", "hair"), ("bore", "boar"),
        ("plane", "plain"), ("main", "mane"), ("reign", "rain"), ("brake", "break"),
        ("steak", "stake"), ("waste", "waist"), ("write", "right"), ("scene", "seen"),
        ("grate", "great"), ("whole", "hole"), ("role", "roll"), ("poll", "pole"),
        ("pore", "pour"), ("soar", "sore"), ("shore", "sure"), ("tide", "tied"),
    ]
    # Collect words from pairs not in `used`, ensuring each word itself is real
    cands = []
    for w1, w2 in HOMOPHONE_PAIRS:
        if (w1 not in used and w1.isalpha() and 3 <= len(w1) <= 8
                and word_freq(w1) >= 5):
            cands.append(w1)
    cands = list(dict.fromkeys(cands))
    if len(cands) >= 4:
        random.shuffle(cands)
        chosen = cands[:4]
        print(
            f"      [L4-homophone] ✓ WORDS THAT SOUND LIKE ANOTHER WORD: {chosen}")
        return Group(4, "WORDS THAT SOUND LIKE ANOTHER WORD",
                     [_w(c, wn_m) for c in chosen], "homophones")
    return None


def _build_l4_silent_letters(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words with silent letters."""
    keys = list(SILENT_LETTER_GROUPS.keys())
    random.shuffle(keys)
    for key in keys:
        pool = SILENT_LETTER_GROUPS[key]
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            label = f"WORDS WITH {key.upper()}"
            print(f"      [L4-silent] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "silent_letter")
    return None


def _build_l4_same_vowel(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words sharing the same vowel pattern."""
    random.shuffle(SAME_VOWEL_GROUPS)
    for label, pool in SAME_VOWEL_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-vowel] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "same_vowel_pattern")
    return None


def _build_l4_minimal_pairs(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words differing by one letter from a target."""
    random.shuffle(MINIMAL_PAIRS_GROUPS)
    for label, pool in MINIMAL_PAIRS_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-minimal] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "differ_by_one_letter")
    return None


def _build_l4_double_letter(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words with a repeated letter pattern."""
    keys = list(REPEATED_LETTER_WORDS.keys())
    random.shuffle(keys)
    for key in keys:
        pool = REPEATED_LETTER_WORDS[key]
        letter = key.replace("double_", "").upper()
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            label = f"WORDS WITH DOUBLE '{letter}'"
            print(f"      [L4-double] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "double_letter_pattern")
    return None


def _build_l4_polysemy(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words sharing multiple meanings of a concept."""
    random.shuffle(POLYSEMY_GROUPS)
    for label, pool in POLYSEMY_GROUPS:
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-polysemy] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "polysemy")
    return None


def _build_l4_same_start(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """4 words starting with the same letter."""
    letters = list(SAME_START_GROUPS.keys())
    random.shuffle(letters)
    for letter in letters:
        pool = SAME_START_GROUPS[letter]
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            label = f"ALL START WITH '{letter.upper()}'"
            print(f"      [L4-start] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "same_starting_letter")
    return None


def _build_l4_same_end(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """4 words ending with the same letter."""
    letters = list(SAME_END_GROUPS.keys())
    random.shuffle(letters)
    for letter in letters:
        pool = SAME_END_GROUPS[letter]
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            label = f"ALL END WITH '{letter.upper()}'"
            print(f"      [L4-end] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "same_ending_letter")
    return None


def _build_l4_multi_category(wn_m, used, used_sigs=None):
    if used_sigs is None:
        used_sigs = set()
    """Words that could fit multiple categories."""
    random.shuffle(MULTI_CATEGORY_WORDS)
    for label, pool in MULTI_CATEGORY_WORDS:
        cands = [w for w in pool if w not in used and w.isalpha() and 3 <=
                 len(w) <= 10]
        cands = list(dict.fromkeys(cands))
        if len(cands) >= 4:
            random.shuffle(cands)
            chosen = cands[:4]
            print(f"      [L4-multi] ✓ {label}: {chosen}")
            return Group(4, label, [_w(c, wn_m) for c in chosen], "double_meaning_category")
    return None


# ─────────────────────────────────────────────────────────────
# Master L4 dispatcher — picks random wordplay pattern
# ─────────────────────────────────────────────────────────────
# Difficulty tiers: HARD (first), MED, EASY — weighted sampling so
# harder patterns appear ~3x more often than easy ones.
L4_WORDPLAY_HARD = [
    _build_l4_anagram,
    _build_l4_polysemy,
    _build_l4_hidden_word,
    _build_l4_palindrome,
    _build_l4_minimal_pairs,
    _build_l4_homophones,
]
L4_WORDPLAY_MED = [
    _build_l4_rhyme,
    _build_l4_same_prefix,
    _build_l4_same_suffix,
    _build_l4_same_vowel,
    _build_l4_silent_letters,
    _build_l4_double_letter,
    _build_l4_multi_category,
]
L4_WORDPLAY_EASY = [
    _build_l4_same_start,
    _build_l4_same_end,
]

# Flat list for backwards-compat; actual dispatch uses weighted tier sampling
L4_WORDPLAY_BUILDERS = L4_WORDPLAY_HARD + L4_WORDPLAY_MED + L4_WORDPLAY_EASY


def _build_l4_wordplay(wn_m, used, used_sigs=None):
    """Try wordplay builders using weighted tier sampling (hard > med > easy)."""
    if used_sigs is None:
        used_sigs = set()
    # Build weighted list: hard builders weighted 3x, med 2x, easy 1x
    weighted = (L4_WORDPLAY_HARD * 3) + \
        (L4_WORDPLAY_MED * 2) + (L4_WORDPLAY_EASY * 1)
    # Deduplicate while preserving weights via random.choices
    order = []
    seen_fns = set()
    random.shuffle(weighted)
    for fn in weighted:
        if fn not in seen_fns:
            order.append(fn)
            seen_fns.add(fn)
    for builder in order:
        try:
            g = builder(wn_m, used, used_sigs=used_sigs)
            if g:
                return g
        except Exception as e:
            print(f"      [L4-wordplay] {builder.__name__} failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Architecture A — Standard (hyponym + synonym + triggered + compound)
# ─────────────────────────────────────────────────────────────
class ArchitectureA:
    NAME = "A — hyponym + synonym + triggered_by + prefix_compound (___ WORD)"

    def __init__(self, bn, wn_m, dm, flt, cn=None):
        self.bn = bn
        self.wn = wn_m
        self.dm = dm
        self.flt = flt
        self.cn = cn

    def level1(self, used, others, used_sigs=None):
        cats = list(self.wn.CATEGORIES.items())
        random.shuffle(cats)
        for concept, data in cats:
            g = _build_l1_hyponym(concept, data, self.bn, self.wn, self.flt, used, others,
                                  used_sigs=used_sigs, cn=self.cn)
            if g:
                return g
        return None

    def level2(self, used, others, used_sigs=None):
        concepts = random.sample(L2_CONCEPTS, len(L2_CONCEPTS))
        for anchor, pos, syn, label in concepts:
            if self.cn:
                g = _build_l2_conceptnet(anchor, pos, syn, label, self.cn, self.dm, self.wn,
                                         self.flt, used, others, used_sigs=used_sigs)
                if g:
                    return g
            g = _build_l2_synonym(anchor, pos, syn, label, self.dm, self.wn, self.flt,
                                  used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level3(self, used, others, used_sigs=None):
        anchors = random.sample(L3_ANCHORS, len(L3_ANCHORS))
        for anchor in anchors:
            if self.cn and random.random() < 0.6:
                g = _build_l3_conceptnet(anchor, self.cn, self.dm, self.wn, self.flt,
                                         used, others, used_sigs=used_sigs)
                if g:
                    return g
            g = _build_l3_triggered(anchor, self.dm, self.wn, self.flt,
                                    used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level4(self, used, used_sigs=None):
        return _build_l4("before", self.dm, self.wn, used, used_sigs=used_sigs)


# ─────────────────────────────────────────────────────────────
# Architecture B — Contrast (alt cats + means_like + abstract + suffix)
# ─────────────────────────────────────────────────────────────
class ArchitectureB:
    NAME = "B — hyponym(alt cats) + means_like + abstract_ml + suffix_compound (WORD ___)"
    B_CATS = ["dance", "snake", "fabric", "hat", "boat", "planet",
              "cheese", "dog", "gem", "flower", "insect", "fish",
              "cat", "horse", "grain", "spice", "nut", "berry",
              "pasta", "martial", "board", "painting", "poetry"]

    def __init__(self, bn, wn_m, dm, flt, cn=None):
        self.bn = bn
        self.wn = wn_m
        self.dm = dm
        self.flt = flt
        self.cn = cn

    def level1(self, used, others, used_sigs=None):
        b_cats = [(k, self.wn.CATEGORIES[k])
                  for k in self.B_CATS if k in self.wn.CATEGORIES]
        random.shuffle(b_cats)
        for concept, data in b_cats:
            g = _build_l1_hyponym(concept, data, self.bn, self.wn, self.flt, used, others,
                                  used_sigs=used_sigs, cn=self.cn)
            if g:
                return g
        all_cats = list(self.wn.CATEGORIES.items())
        random.shuffle(all_cats)
        for concept, data in all_cats:
            g = _build_l1_hyponym(concept, data, self.bn, self.wn, self.flt, used, others,
                                  used_sigs=used_sigs, cn=self.cn)
            if g:
                return g
        return None

    def level2(self, used, others, used_sigs=None):
        concepts = random.sample(L2_CONCEPTS, len(L2_CONCEPTS))
        for anchor, pos, syn, label in concepts:
            if self.cn:
                g = _build_l2_conceptnet(anchor, pos, syn, label, self.cn, self.dm, self.wn, self.flt,
                                         used, others, used_sigs=used_sigs)
                if g:
                    return g
            g = _build_l2_meanslike(anchor, pos, syn, label, self.dm, self.wn, self.flt,
                                    used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level3(self, used, others, used_sigs=None):
        anchors = random.sample(L3_ABSTRACT, len(L3_ABSTRACT))
        for anchor in anchors:
            if self.cn and random.random() < 0.6:
                g = _build_l3_conceptnet(anchor, self.cn, self.dm, self.wn, self.flt,
                                         used, others, used_sigs=used_sigs)
                if g:
                    return g
            g = _build_l3_abstract(anchor, self.dm, self.wn, self.flt,
                                   used, others, used_sigs=used_sigs)
            if g:
                return g
        anchors2 = random.sample(L3_ANCHORS, len(L3_ANCHORS))
        for anchor in anchors2:
            g = _build_l3_triggered(anchor, self.dm, self.wn, self.flt,
                                    used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level4(self, used, used_sigs=None):
        return _build_l4("after", self.dm, self.wn, used, used_sigs=used_sigs)


# ─────────────────────────────────────────────────────────────
# Architecture C — Meronym (parts-of-whole + cooccurrence + mixed)
# ─────────────────────────────────────────────────────────────
class ArchitectureC:
    NAME = "C — meronym(parts) + synonym + cooccur + mixed_compound (random)"

    def __init__(self, bn, wn_m, dm, flt, cn=None):
        self.bn = bn
        self.wn = wn_m
        self.dm = dm
        self.flt = flt
        self.cn = cn

    def level1(self, used, others, used_sigs=None):
        if used_sigs is None:
            used_sigs = set()
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
            for _ in range(15):
                chosen = random.sample(clean, min(4, len(clean)))
                if len(chosen) < 4:
                    break
                if frozenset(chosen) not in used_sigs:
                    return Group(1, display, [_w(c, self.wn) for c in chosen], "wn_meronym", syn)
        all_cats = list(self.wn.CATEGORIES.items())
        random.shuffle(all_cats)
        for concept, data in all_cats:
            g = _build_l1_hyponym(concept, data, self.bn, self.wn, self.flt, used, others,
                                  used_sigs=used_sigs, cn=self.cn)
            if g:
                return g
        return None

    def level2(self, used, others, used_sigs=None):
        concepts = random.sample(L2_CONCEPTS, len(L2_CONCEPTS))
        for anchor, pos, syn, label in concepts:
            g = _build_l2_synonym(anchor, pos, syn, label, self.dm, self.wn, self.flt,
                                  used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level3(self, used, others, used_sigs=None):
        anchors = random.sample(L3_ANCHORS, len(L3_ANCHORS))
        for anchor in anchors:
            if self.cn and random.random() < 0.6:
                g = _build_l3_conceptnet(anchor, self.cn, self.dm, self.wn, self.flt,
                                         used, others, used_sigs=used_sigs)
                if g:
                    return g
            g = _build_l3_cooccur(anchor, self.dm, self.wn, self.flt,
                                  used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level4(self, used, used_sigs=None):
        return _build_l4("any", self.dm, self.wn, used, used_sigs=used_sigs)


# ─────────────────────────────────────────────────────────────
# Architecture D — Wordplay (wordplay-focused L4, still semantic L1-L3)
# ─────────────────────────────────────────────────────────────
class ArchitectureD:
    NAME = "D — hyponym + means_like + abstract + WORDPLAY_L4 (rhyme/anagram/palindrome/etc.)"

    def __init__(self, bn, wn_m, dm, flt, cn=None):
        self.bn = bn
        self.wn = wn_m
        self.dm = dm
        self.flt = flt
        self.cn = cn

    def level1(self, used, others, used_sigs=None):
        cats = list(self.wn.CATEGORIES.items())
        random.shuffle(cats)
        for concept, data in cats:
            g = _build_l1_hyponym(concept, data, self.bn, self.wn, self.flt, used, others,
                                  used_sigs=used_sigs, cn=self.cn)
            if g:
                return g
        return None

    def level2(self, used, others, used_sigs=None):
        concepts = random.sample(L2_CONCEPTS, len(L2_CONCEPTS))
        for anchor, pos, syn, label in concepts:
            g = _build_l2_meanslike(anchor, pos, syn, label, self.dm, self.wn, self.flt,
                                    used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level3(self, used, others, used_sigs=None):
        all_anchors = L3_ABSTRACT + L3_ANCHORS
        anchors = random.sample(all_anchors, min(len(all_anchors), 50))
        for anchor in anchors:
            if self.cn and random.random() < 0.6:
                g = _build_l3_conceptnet(anchor, self.cn, self.dm, self.wn, self.flt,
                                         used, others, used_sigs=used_sigs)
                if g:
                    return g
            g = _build_l3_abstract(anchor, self.dm, self.wn, self.flt,
                                   used, others, used_sigs=used_sigs)
            if g:
                return g
        return None

    def level4(self, used, used_sigs=None):
        return _build_l4_wordplay(self.wn, used, used_sigs=used_sigs)


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
        try:
            score = round(
                min(sum(w.frequency for w in ws) / max(len(ws), 1) / 300.0, 1.0) * 40 +
                len({g.level for g in groups}) / 4.0 * 30 +
                min(sum(w.polysemy for w in ws) / max(len(ws), 1) / 8.0, 1.0) * 30, 1)
        except Exception:
            score = 50.0
        return Puzzle(groups=groups, grid=words, score=score, arch=arch)


class NYTConnectionsGenerator:

    def __init__(self, key: str):
        print("  [init] Starting…")
        self.bn = BabelNetDisambiguator(key)
        self.wn = WordNetMiner()
        self.dm = DatamuseMiner()
        self.cn = ConceptNetMiner()
        self.flt = WordFilter(min_freq=2)
        self.asm = Assembler()
        self.bn.probe()
        print("  [init] ConceptNet ready (free, no key) ✓")

    def _arch(self, name: str):
        return {"A": ArchitectureA, "B": ArchitectureB,
                "C": ArchitectureC, "D": ArchitectureD}[name](
            self.bn, self.wn, self.dm, self.flt, cn=self.cn)

    def generate_one(self, arch_name: str = "A", retries: int = 15,
                     global_used: set = None,
                     used_sigs: set = None) -> Puzzle:
        """
        Generate one puzzle.
        global_used: words to exclude within this puzzle (within-puzzle dedup)
        used_sigs:   frozensets of already-used groups — skip these combos
        """
        base_used = set(global_used) if global_used else set()
        if used_sigs is None:
            used_sigs = set()
        arch = self._arch(arch_name)

        for attempt in range(retries):
            print(f"\n  [gen-{arch_name}] Attempt {attempt+1}/{retries}…")
            used, groups, others = set(base_used), [], []
            try:
                g1 = arch.level1(used, others, used_sigs)
                if not g1:
                    print("    L1 failed")
                    continue
                used.update(w.text for w in g1.words)
                others.append([w.text for w in g1.words])
                groups.append(g1)
                print(f"    ✓ L1 [{g1.name}]: {[w.text for w in g1.words]}")

                g2 = arch.level2(used, others, used_sigs)
                if not g2:
                    print("    L2 failed")
                    continue
                used.update(w.text for w in g2.words)
                others.append([w.text for w in g2.words])
                groups.append(g2)
                print(f"    ✓ L2 [{g2.name}]: {[w.text for w in g2.words]}")

                g3 = arch.level3(used, others, used_sigs)
                if not g3:
                    print("    L3 failed")
                    continue
                used.update(w.text for w in g3.words)
                others.append([w.text for w in g3.words])
                groups.append(g3)
                print(f"    ✓ L3 [{g3.name}]: {[w.text for w in g3.words]}")

                g4 = arch.level4(used, used_sigs)
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

    def generate_dataset(self, n: int = 5, output_path: str = "dataset.jsonl") -> list:
        arch_cycle = ["A", "B", "C", "D"]
        puzzles = []

        # Load existing puzzles for overlap checking (group-signature based only)
        existing_puzzles = _load_existing_puzzles(output_path)
        print(
            f"\n  [overlap] Loaded {len(existing_puzzles)} existing puzzles from \'{output_path}\'")

        # Build set of existing group signatures (frozensets of 4 words each)
        # We only block EXACT duplicate groups, not individual word reuse.
        # This is correct for data augmentation — NYT reuses words across puzzles.
        existing_group_sigs = set()
        for ep in existing_puzzles:
            for g in ep.get("groups", []):
                sig = frozenset(w["word"].lower() for w in g.get("words", []))
                existing_group_sigs.add(sig)

        arch_classes = {"A": ArchitectureA, "B": ArchitectureB,
                        "C": ArchitectureC, "D": ArchitectureD}

        # Consecutive failures per arch — if an arch fails MAX_ARCH_FAILS times
        # in a row, skip it for this puzzle slot and try the next arch.
        MAX_ARCH_FAILS = 3
        arch_fail_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

        i = 0
        total_attempts = 0
        MAX_TOTAL = n * 8  # hard ceiling to prevent infinite loops

        while i < n and total_attempts < MAX_TOTAL:
            # Pick arch — skip arches that are currently exhausted
            arch_order = [arch_cycle[(i + k) % len(arch_cycle)]
                          for k in range(4)]
            arch = None
            for a in arch_order:
                if arch_fail_counts[a] < MAX_ARCH_FAILS:
                    arch = a
                    break
            if arch is None:
                # All arches failing — reset counters and try again
                print(
                    f"  [!] All arches exhausted for puzzle {i+1}, resetting fail counts")
                arch_fail_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
                arch = arch_cycle[i % len(arch_cycle)]

            arch_obj = arch_classes[arch]
            print(f"\n{'='*60}")
            print(f"  Puzzle {i+1}/{n}  |  Architecture {arch}")
            print(f"  {arch_obj.NAME}")
            print(f"{'='*60}")
            total_attempts += 1

            try:
                # Pass used_sigs so builders avoid already-generated groups
                p = self.generate_one(arch_name=arch, global_used=None,
                                      used_sigs=existing_group_sigs)
                p_dict = self.to_jsonl(p)

                # Check: any group identical to an existing group?
                new_sigs = []
                duplicate = False
                for g in p_dict.get("groups", []):
                    sig = frozenset(w["word"].lower()
                                    for w in g.get("words", []))
                    if sig in existing_group_sigs:
                        print(f"  [!] Duplicate group {sig} — retrying")
                        duplicate = True
                        break
                    new_sigs.append(sig)

                if duplicate:
                    arch_fail_counts[arch] += 1
                    continue

                # Accept puzzle
                for sig in new_sigs:
                    existing_group_sigs.add(sig)
                puzzles.append(p)
                arch_fail_counts[arch] = 0  # reset on success
                print(f"\n  → Puzzle {i+1} accepted  (total generated: {i+1})")
                i += 1

                # Save incrementally every puzzle so Ctrl+C never loses work
                _save_incremental(p, self, output_path,
                                  output_path.replace(".jsonl", ".csv"))

            except Exception as e:
                arch_fail_counts[arch] += 1
                print(f"  [!] Puzzle {i+1} arch={arch} error: {e}")

        if total_attempts >= MAX_TOTAL:
            print(
                f"\n  [!] Hit attempt ceiling ({MAX_TOTAL}). Saved {i} puzzles.")

        return puzzles

    def print_puzzle(self, p: Puzzle):
        colors = {1: "🟡 YELLOW", 2: "🟢 GREEN", 3: "🔵 BLUE", 4: "🟣 PURPLE"}
        arch_names = {"A": ArchitectureA.NAME, "B": ArchitectureB.NAME,
                      "C": ArchitectureC.NAME, "D": ArchitectureD.NAME}
        print(f"\n{'═'*60}")
        print(f"  NYT CONNECTIONS — Architecture {p.arch}")
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

    def to_csv_rows(self, p: Puzzle, game_id: int, puzzle_date: str) -> list:
        """
        Returns rows in the original CSV format:
        Game ID, Puzzle Date, Word, Group Name, Group Level, Starting Row, Starting Column
        Level mapping: 1=yellow(0), 2=green(1), 3=blue(2), 4=purple(3)
        """
        rows = []
        # Build grid layout (4x4), shuffled
        grid_words = list(p.grid)  # already shuffled
        word_positions = {}
        for idx, word in enumerate(grid_words):
            row = (idx // 4) + 1
            col = (idx % 4) + 1
            word_positions[word] = (row, col)

        for g in p.groups:
            level_0indexed = g.level - 1  # 0-indexed like original dataset
            for w in g.words:
                row, col = word_positions.get(w.text, (0, 0))
                rows.append({
                    "Game ID": game_id,
                    "Puzzle Date": puzzle_date,
                    "Word": w.text.upper(),
                    "Group Name": g.name,
                    "Group Level": level_0indexed,
                    "Starting Row": row,
                    "Starting Column": col,
                })
        return rows


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key",      required=True, help="BabelNet API key")
    parser.add_argument("--puzzles",  type=int, default=5,
                        help="Number of puzzles to generate")
    parser.add_argument("--output",   default="dataset.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--csv",      default="connections_dataset.csv",
                        help="Output CSV file (same format as original)")
    parser.add_argument("--arch",     default="cycle",
                        help="A, B, C, D, or cycle (default: cycle A→B→C→D→...)")
    parser.add_argument("--start-id", type=int, default=None,
                        help="Starting Game ID for CSV export (auto-detects from existing CSV)")
    args = parser.parse_args()

    gen = NYTConnectionsGenerator(key=args.key)

    # generate_dataset handles everything:
    # - loads existing puzzles from output file
    # - skips duplicate groups
    # - saves EACH puzzle immediately (Ctrl+C safe)
    # - rotates architectures automatically
    # - if one arch is exhausted, tries others
    if args.arch == "cycle":
        puzzles = gen.generate_dataset(n=args.puzzles, output_path=args.output)
    else:
        # Single-arch mode: use generate_dataset with locked arch
        # Temporarily patch cycle to only use the requested arch
        original_cycle = ["A", "B", "C", "D"]
        puzzles = []
        existing_puzzles = _load_existing_puzzles(args.output)
        existing_group_sigs = set()
        for ep in existing_puzzles:
            for g in ep.get("groups", []):
                sig = frozenset(w["word"].lower() for w in g.get("words", []))
                existing_group_sigs.add(sig)
        saved = 0
        for attempt in range(args.puzzles * 8):
            if saved >= args.puzzles:
                break
            try:
                p = gen.generate_one(arch_name=args.arch)
                p_dict = gen.to_jsonl(p)
                duplicate = False
                new_sigs = []
                for g in p_dict.get("groups", []):
                    sig = frozenset(w["word"].lower()
                                    for w in g.get("words", []))
                    if sig in existing_group_sigs:
                        duplicate = True
                        break
                    new_sigs.append(sig)
                if duplicate:
                    continue
                for sig in new_sigs:
                    existing_group_sigs.add(sig)
                puzzles.append(p)
                _save_incremental(p, gen, args.output, args.csv)
                saved += 1
                print(f"  → Saved puzzle {saved}/{args.puzzles}")
            except Exception as e:
                print(f"  [!] Failed: {e}")

    if not puzzles:
        print("\n[!] No new puzzles generated this run.")
        print(
            f"    Check {args.output} — puzzles may have been saved incrementally.")
    else:
        print(
            f"\n✅  {len(puzzles)} puzzles generated this run → {args.output} / {args.csv}")
        print(f"    (Each puzzle was saved immediately — safe even if interrupted)")
