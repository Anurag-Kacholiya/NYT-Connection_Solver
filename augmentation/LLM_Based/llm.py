from google import genai
import os
import argparse


parser = argparse.ArgumentParser(description="Generate NYT Connections puzzles using Gemini")
parser.add_argument("n", type=int, help="Number of puzzles to generate")
parser.add_argument("--output", default="connections_dataset.csv", help="Output CSV file")

args = parser.parse_args()
num_puzzles = args.n
output_file = args.output

# Initialize client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt_template = """
LLM Prompt for Generating NYT Connections Puzzle Data
You are a puzzle designer for the New York Times "Connections" word game. Your job is to generate complete, high-quality puzzles that follow the exact structure and difficulty patterns of the real game.

GAME STRUCTURE (STRICT)
Every puzzle has exactly:

16 words total, arranged in a 4×4 grid
4 groups of exactly 4 words each
One group at each difficulty level: 0 (Yellow/easiest), 1 (Green), 2 (Blue), 3 (Purple/hardest)
All words are in UPPERCASE
~97% of words are single words (multi-word entries are rare)
Word length: median 5 characters, typically 3–8 characters
OUTPUT FORMAT
For each puzzle, output in this exact CSV format (same as the NYT dataset):

Game ID,Puzzle Date,Word,Group Name,Group Level,Starting Row,Starting Column
Each puzzle has 16 rows (4 groups × 4 words). The Starting Row (1-4) and Starting Column (1-4) define the shuffled grid position — assign these so the 4 words of each group are NOT adjacent.
LEVEL-BY-LEVEL DESIGN RULES
LEVEL 0 — YELLOW (Easiest) — 93% plain category names
Group Name Style: Short, direct, literal. Plain English noun category or synonym description. Median 14 characters.

✅ "WET WEATHER", "FOOTWEAR", "STREAMING SERVICES", "SHADES OF RED"
✅ Synonym style: "ENORMOUS", "MOVE QUICKLY", "DECEIVE", "COMPLAIN"
❌ Never use wordplay, blanks (___), homophones, or clever puns
Word Selection:

All 4 words should OBVIOUSLY belong to the category for any native English speaker
Use common, everyday vocabulary
But 1-2 words should have alternate meanings that could confuse solvers into thinking they belong to another group in the puzzle (this is critical for misdirection)
Examples from real puzzles:

STREAMING SERVICES → PRIME, PEACOCK, HULU, NETFLIX
WET WEATHER → SNOW, HAIL, RAIN, SLEET
SNEAKER BRANDS → PUMA, NIKE, REEBOK, ADIDAS
FRUITS → FIG, LIME, APRICOT, GRAPE
ENDORSE → CHAMPION, SECOND, BACK, SUPPORT
PURSUIT → HUNT, SEARCH, CHASE, QUEST
PARTS OF A CAR → HOOD, TRUNK, TIRE, BUMPER
LEVEL 1 — GREEN (Easy-Medium) — 89% direct categories, slightly more abstract
Group Name Style: Still mostly direct, but categories are more specific, abstract, or require moderate cultural knowledge. May use "KINDS OF", "PARTS OF A", "THINGS THAT" frames (~7%).

✅ "BABY ANIMALS", "NBA TEAMS", "PALINDROMES", "CHESS TERMS"
✅ "FEATURES OF A SKI RESORT", "SYNONYMS FOR EAT", "METRIC PREFIXES"
❌ No wordplay or linguistic tricks
Word Selection:

Every word should be polysemous — it should have at least one meaning outside this group
This is the primary misdirection level. Words should make solvers hesitate
The category itself is recognizable once you see it, but individual words pull your attention elsewhere
Examples:

NBA TEAMS → HEAT, BUCKS, JAZZ, NETS (all have non-basketball meanings)
BABY ANIMALS → CUB, JOEY, CALF, KID ("kid" = child)
CONDIMENTS → TARTAR, RELISH, KETCHUP, MAYO ("relish" = enjoy)
CHESS TERMS → BISHOP, CASTLE, CHECK, GAMBIT (all have other meanings)
MOVE QUICKLY → ZIP, DASH, DART, BOLT (all are also nouns)
PARTS OF A BOOK → LEAVES, COVER, JACKET, SPINE (all dual-meaning)
MUSICALS BEGINNING WITH "C" → CATS, CAROUSEL, CHICAGO, CABARET
LEVEL 2 — BLUE (Medium-Hard) — 82% direct but niche, wordplay emerging
Group Name Style: More complex, longer (median 18 chars). Niche trivia, specific pop culture, or light linguistic patterns start appearing.

✅ "FORD MODELS", "YOGA POSES", "BOXING PUNCHES", "SLANG FOR MONEY"
✅ "THINGS THAT CAN RUN, ANNOYINGLY", "WORDS DERIVED FROM JAPANESE"
✅ Light wordplay: "SILENT 'G'", "WORDS BEFORE 'NUT'", "BEGINNING WITH DOUBLE LETTERS"
✅ Pop culture: "EDDIE MURPHY ROLES", "'PEANUTS' CHARACTERS", "BEST PICTURE WINNERS SINCE 2000"
Word Selection:

Words should be MAXIMALLY AMBIGUOUS — they must plausibly fit 2-3 groups
Can include proper nouns for pop culture categories
The most reused L2 words in real puzzles: CHECK, BALL, BLUE, ROCK, TWIST, CARD, POUND, SAW, HOOK — all heavily polysemous
Examples:

SILENT "G" → GNOME, GNOCCHI, GNAW, GNAT
CLEANING VERBS → DUST, MOP, SWEEP, VACUUM
SYNONYMS FOR SAD → BLUE, GLUM, DOWN, LOW (BLUE/DOWN are traps)
FORD MODELS → BRONCO, ESCAPE, EXPLORER, MUSTANG
YOGA POSES → BOW, CAMEL, CHAIR, COBRA
BIRDS → SWIFT, CARDINAL, LARK, JAY (all have alternate meanings)
LOSE IT, WITH "OUT" → BUG, WIG, FLIP, FREAK
LEVEL 3 — PURPLE (Hardest) — Only 35% plain categories; 65% wordplay
This level is FUNDAMENTALLY DIFFERENT. The connection is usually linguistic/structural, not semantic. The 4 words should look completely unrelated on the surface.
Pattern Distribution (must follow these proportions):

33% Compound words (___ X or X ___): Words that form compounds with a hidden word
8% Hidden words at start/end: Common words that secretly start/end with members of a category
6% Words in phrases: Words extracted from specific titles, quotes, or cultural sets
4% Homophones: Words that SOUND LIKE members of a category
4% Letter manipulation: Known items with letters added/removed/changed
3% Polysemy ("What X might mean"): Different meanings of one word
35% Other tricky categories: Things with metaphorical connections, cultural references requiring lateral thinking
LEVEL 3 PATTERN DETAILS:
COMPOUND PREFIX (___ WORD) — Most common pattern
Group name: "___ CAKE", "___ FISH", "___ CLOCK", "___ HOUSE"
Each word + hidden word = recognized compound/phrase.

___ MAN SUPERHEROES → SPIDER, IRON, SUPER, BAT
___ CAKE → CARROT, SPONGE, COFFEE, POUND
___ HOUSE → BIRD, DOG, FIRE, PEN
___ CLOCK → ALARM, GRANDFATHER, BIOLOGICAL, CUCKOO
___ STICK → MEMORY, SELFIE, HOCKEY, FISH
COMPOUND SUFFIX (WORD ___) — Second most common
Group name: "FIRE ___", "MR. ___", "DIRTY ___", "FULL ___"
Hidden word + each word = recognized compound/phrase.

FIRE ___ → CRACKER, FLY, PLACE, WORK
MR. ___ → BEAN, CLEAN, PEANUT, FOX
DIRTY ___ → LAUNDRY, MARTINI, JOKE, DOZEN
GOLD ___ → BOND, RUSH, LEAF, MINE
FULL ___ → MONTY, MOON, CIRCLE, HOUSE
HOMOPHONES
Group name: "[CATEGORY] HOMOPHONES"
Words that sound like members of the named category but are spelled differently.

LETTER HOMOPHONES → SEA (C), WHY (Y), ARE (R), QUEUE (Q)
FRUIT HOMOPHONES → LYME (lime), PAIR (pear), PLUMB (plum), MELLON (melon)
NUMBER HOMOPHONES → WON (one), TOO (two), ATE (eight), FOR (four)
COLOR HOMOPHONES → READ (red), BLEW (blue), ROWS (rose), CHORAL (coral)
HIDDEN WORDS AT START/END
Group name: "STARTING WITH [CATEGORY]", "ENDING WITH [CATEGORY]"
Common words that contain members of a category embedded in their letters.

STARTING WITH ANIMALS → CATASTROPHE (CAT), BEARISH (BEAR), RAMPAGE (RAM), COWARD (COW)
STARTS OF U.S. COINS → DIM (dime), QUART (quarter), PEN (penny), NICK (nickel)
ENDING WITH CLOTHING → WINDSOCK (sock), TURNCOAT (coat), FOXGLOVE (glove), GUMSHOE (shoe)
WORDS BEGINNING WITH INSTRUMENTS → CELLOPHANE (cello), ORGANISM (organ), HARPOON (harp), BASSINET (bass)
LETTER MANIPULATION
Group name: "[ITEMS] MINUS [WHAT]", "[ITEMS] PLUS [WHAT]"
Known proper nouns with consistent letter transformations applied.

BAND NAMES MINUS COLORS → DAY (Green Day), STRIPES (White Stripes), FLOYD (Pink Floyd), SABBATH (Black Sabbath)
BAND NAMES MINUS NUMBERS → SUM (Sum 41), MAROON (Maroon 5), U (U2), BLINK (Blink-182)
COLORS MINUS THEIR FIRST LETTERS → INK (pink), OLD (gold), LACK (black), RANGE (orange)
POLYSEMY (What "X" might mean)
Group name: WHAT "[WORD]" MIGHT MEAN
4 words each representing a completely different meaning of the same word.

WHAT "GREEN" MIGHT MEAN → FRESH, ENVIOUS, NAIVE, UNWELL
WHAT "X" MIGHT MEAN → ADULT, TEN, TIMES, KISS
WHAT "K" MIGHT MEAN → OKAY, KELVIN, THOUSAND, POTASSIUM
WHAT "POP" MIGHT MEAN → MAINSTREAM, DAD, BURST, SODA
WORDS IN PHRASES/TITLES
Group name: "FIRST WORDS IN [SOURCE]", "WORDS BEFORE [WORD]"

FIRST WORDS IN RAPPERS' NAMES → FOXY (Foxy Brown), KILLER (Killer Mike), GUCCI (Gucci Mane), NOTORIOUS (Notorious B.I.G.)
WORDS BEFORE DAYS OF THE WEEK → ASH (Ash Wednesday), BLACK (Black Friday), CYBER (Cyber Monday), FAT (Fat Tuesday)
WORDS SPELLED WITH ROMAN NUMERALS → MIX (M+I+X), MILD (M+I+L+D), LIVID, DILL
CRITICAL MISDIRECTION RULES
The puzzle's quality comes from CROSS-GROUP CONFUSION. Follow these rules:

Include trap words: At least 3-4 words in the puzzle should plausibly fit a DIFFERENT group. For example, if one group is "FRUITS" and another is "CLUE CHARACTERS", PLUM appears to be a fruit but actually belongs to Clue characters.
Use polysemous words: The most-reused words across real puzzles are: BALL, RING, HEART, WING, BAR, BLUE, STAR, FISH, LOVE, SPLIT, KING, BABY, FLY, ROCK, EGG, DRIVE, STICK, HOOK, JACK, SUN. These work because they have 5+ meanings.
Never have two groups with overlapping themes: Don't have both FRUITS and VEGETABLES. But DO have words that create bridges (e.g., a color word like BLUE in a non-color group).
Level 3 words should look random: The 4 Purple words should never obviously go together. SPIDER, IRON, SUPER, BAT look unrelated until you realize they all precede "MAN".
Design Purple FIRST, then build the rest around it: This ensures the hardest group has maximum misdirection.
COMPLETE PUZZLE EXAMPLES
Example 1 (Classic structure):
L0 [SNEAKER BRANDS]: PUMA, NIKE, REEBOK, ADIDAS
L1 [MUSICALS BEGINNING WITH "C"]: CATS, CAROUSEL, CHICAGO, CABARET
L2 [CLEANING VERBS]: DUST, MOP, SWEEP, VACUUM
L3 [___ MAN SUPERHEROES]: SPIDER, IRON, SUPER, BAT
Note: CATS (musical or animal?), PUMA (sneaker or animal?), DUST (cleaning or what collects?)

Example 2 (Homophones):
L0 [FOOTWEAR]: PUMP, LOAFER, BOOT, SNEAKER
L1 [UNITS OF LENGTH]: FOOT, LEAGUE, YARD, MILE
L2 [MAGAZINES]: TIME, US, PEOPLE, ESSENCE
L3 [LETTER HOMOPHONES]: SEA (C), WHY (Y), ARE (R), QUEUE (Q)
Note: FOOT (footwear or length?), TIME (magazine or concept?), PEOPLE (magazine or noun?)

Example 3 (Hidden words):
L0 [PARTS OF AN AIRPORT]: RUNWAY, TERMINAL, HANGAR, TARMAC
L1 [LEGAL TERMS]: LAWSUIT, CLAIM, COMPLAINT, ACTION
L2 [THINGS A JUGGLER JUGGLES]: CLUB, BEANBAG, TORCH, RING
L3 [WORDS ENDING IN CLOTHING]: WINDSOCK (sock), TURNCOAT (coat), FOXGLOVE (glove), GUMSHOE (shoe)
Note: CLUB (juggling or nightclub?), RING (juggling or jewelry?), ACTION (legal or movie?)

Example 4 (Polysemy):
L0 [FRUITS]: FIG, LIME, APRICOT, GRAPE
L1 [LUXURIOUS]: GRAND, DELUXE, LAVISH, OPULENT
L2 [BEST ACTRESS OSCAR WINNERS]: SWANK, STONE, FOSTER, BERRY
L3 [WHAT "GREEN" MIGHT MEAN]: FRESH, ENVIOUS, NAIVE, UNWELL
Note: LIME (fruit or color?), BERRY (fruit or actress?), STONE (luxurious/gemstone or actress?)

Example 5 (Letter manipulation):
L0 [MODES OF TRANSPORTATION]: PLANE, BOAT, CAR, TRAIN
L1 [NBA PLAYERS]: SUN, THUNDER, KING, MAGIC
L2 [FAST FOOD CHAINS]: SONIC, CHECKERS, SUBWAY, OUTBACK
L3 [BAND NAMES MINUS COLORS]: DAY (Green Day), STRIPES (White Stripes), FLOYD (Pink Floyd), SABBATH (Black Sabbath)
Note: KING (NBA or royalty?), MAGIC (NBA or illusion?), TRAIN (transport or music?)

Example 6 (Compound suffix + strong misdirection):
L0 [PURSUIT]: HUNT, SEARCH, CHASE, QUEST
L1 [ROMANTIC LETTER SIGN-OFFS]: KISSES, ALWAYS, YOURS, LOVE
L2 [GROUP WITHIN A GROUP]: WING, PARTY, CAMP, SIDE
L3 [GOLD ___]: BOND, RUSH, LEAF, MINE
Note: MINE (gold mine or possessive?), BOND (gold bond or connection?), PARTY (group or celebration?), WING (group or body part?)
GENERATION INSTRUCTIONS

Start with Level 3 (Purple). Pick one of the L3 pattern types using the distribution: ~33% compound, ~8% hidden words, ~6% words-in-phrases, ~4% homophones, ~4% letter manipulation, ~3% polysemy, ~35% other tricky.
Then design Level 0 (Yellow) — an obvious category. Make sure 1-2 of its words could be confused with the L3 group.
Then Level 1 (Green) — a category requiring slightly more thought. Use polysemous words that create bridges to other groups.
Then Level 2 (Blue) — niche knowledge or light wordplay. Words should be maximally ambiguous.
Verify misdirection: Check that at least 3-4 words across the puzzle could plausibly fit in a different group.
Assign grid positions: Place each word at a unique (row, column) position from (1,1) to (4,4). Ensure no group's 4 words share the same row or column.
Also Ensure one most imporant thing: in every puzzle each word must belong to only 1 subgroup and words should not be repeated in single puzzle
Output each puzzle in the CSV format specified above.

Now generate {n} complete NYT Connections puzzles.
Output in CSV format.
"""

prompt = prompt_template.format(n=num_puzzles)

# Call Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

generated_text = response.text.replace("```csv", "").replace("```", "")



file_exists = os.path.exists(output_file)

with open(output_file, "a") as f:
    # if not file_exists:
    #     f.write("Game ID,Puzzle Date,Word,Group Name,Group Level,Starting Row,Starting Column\n")
    f.write(generated_text + "\n")



print("Generated puzzles saved to:", output_file)
print(generated_text)