# GloVe 300d + KMeans Baseline Evaluation Results

## How to Run this Evaluation
To reproduce these results, navigate to the `Baseline-0-Glove-KMeans` directory and execute:

```bash
python Glove_Kmeans.py
```

## Model Architecture
You can view the detailed system architecture for this GloVe-based baseline here: [architecture.html](architecture.html)

## Methodology
The evaluation of this GloVe-based baseline follows a structured 4-stage pipeline:

1. **Static Embedding Retrieval**: Words are mapped to 300-dimensional vectors using the pre-trained GloVe 6B dataset.
2. **Greedy Simulation Environment**: To mimic human gameplay, the model iteratively selects the highest-similarity word sets.
3. **Iterative Error Handling**: The simulation accounts for 4 lives and 'one-away' feedback, allowing for a realistic assessment of iterative performance.
4. **Metric Calculation**: Performance is measured across perfect solves, group intersections, and red-herring susceptibility.

---

Evaluated on 100 randomly selected games.

## Game-by-Game Breakdown

### Game 1
**Ground Truth Groups:**
- ['CUE', 'PROMPT', 'SIGNAL', 'WORD']
- ['CLUTCH', 'MESSENGER', 'SATCHEL', 'TOTE']
- ['BOXER', 'BRIEF', 'DRAWER', 'SHORT']
- ['FREEDOM', 'LATITUDE', 'LICENSE', 'SLACK']

**Predictions Sequence:**
- Incorrect: ['BRIEF', 'PROMPT', 'SHORT', 'WORD']
- Incorrect: ['BRIEF', 'CUE', 'SHORT', 'WORD']
- Incorrect: ['BRIEF', 'CUE', 'PROMPT', 'SHORT']
- One away: ['CUE', 'PROMPT', 'SHORT', 'WORD']

**Matched Groups: 0/4**

### Game 2
**Ground Truth Groups:**
- ['BOAT', 'BREAD', 'SLUG', 'SPLIT']
- ['CROSS', 'PRICKLY', 'SORE', 'SOUR']
- ['CHAR', 'FLUKE', 'PIKE', 'SKATE']
- ['COBBLER', 'CRISP', 'CRUMBLE', 'TART']

**Predictions Sequence:**
- One away: ['COBBLER', 'CRISP', 'SOUR', 'TART']
- One away: ['CRISP', 'CRUMBLE', 'SOUR', 'TART']
- One away: ['COBBLER', 'CRUMBLE', 'SOUR', 'TART']
- Incorrect: ['CRISP', 'PRICKLY', 'SOUR', 'TART']

**Matched Groups: 0/4**

### Game 3
**Ground Truth Groups:**
- ['BALL', 'MOVIE', 'SCHOOL', 'VITAMIN']
- ['LASER', 'PLUCK', 'THREAD', 'WAX']
- ['HONEYCOMB', 'ORGANISM', 'SOLAR PANEL', 'SPREADSHEET']
- ['COIL', 'SPOOL', 'WIND', 'WRAP']

**Predictions Sequence:**
- Incorrect: ['PLUCK', 'SOLAR PANEL', 'THREAD', 'WRAP']
- Incorrect: ['PLUCK', 'SOLAR PANEL', 'WIND', 'WRAP']
- One away: ['PLUCK', 'THREAD', 'WAX', 'WRAP']
- Incorrect: ['SOLAR PANEL', 'THREAD', 'WIND', 'WRAP']

**Matched Groups: 0/4**

### Game 4
**Ground Truth Groups:**
- ['RING', 'STICK', 'TENDER', 'WING']
- ['DRIFT', 'IDEA', 'MESSAGE', 'POINT']
- ['KIND', 'SORT', 'STYLE', 'TYPE']
- ['COOL', 'NICE', 'SICK', 'SWEET']

**Predictions Sequence:**
- One away: ['IDEA', 'KIND', 'SORT', 'TYPE']
- Incorrect: ['IDEA', 'KIND', 'NICE', 'SORT']
- Incorrect: ['IDEA', 'KIND', 'MESSAGE', 'SORT']
- Incorrect: ['IDEA', 'KIND', 'POINT', 'SORT']

**Matched Groups: 0/4**

### Game 5
**Ground Truth Groups:**
- ['BAROQUE', 'COMPLEX', 'ELABORATE', 'INVOLVED']
- ['CRUX', 'ESSENCE', 'HEART', 'SUBSTANCE']
- ['ARROW', 'BULLET', 'CHECKBOX', 'HYPHEN']
- ['ANGRY', 'BETRAY', 'CRUCIFIX', 'HYBRID']

**Predictions Sequence:**
- One away: ['COMPLEX', 'ELABORATE', 'ESSENCE', 'INVOLVED']
- One away: ['ESSENCE', 'HEART', 'INVOLVED', 'SUBSTANCE']
- Incorrect: ['ELABORATE', 'ESSENCE', 'INVOLVED', 'SUBSTANCE']
- Incorrect: ['COMPLEX', 'ESSENCE', 'INVOLVED', 'SUBSTANCE']

**Matched Groups: 0/4**

### Game 6
**Ground Truth Groups:**
- ['COO', 'GOBBLE', 'HONK', 'HOOT']
- ['SCARF', 'SHAWL', 'STOLE', 'WRAP']
- ['APP', 'GOO', 'MOTOR', 'SAM']
- ['BEAR', 'STAND', 'SWALLOW', 'TAKE']

**Predictions Sequence:**
- Correct: ['BEAR', 'STAND', 'SWALLOW', 'TAKE']
- Correct: ['SCARF', 'SHAWL', 'STOLE', 'WRAP']
- Correct: ['COO', 'GOBBLE', 'HONK', 'HOOT']
- Correct: ['APP', 'GOO', 'MOTOR', 'SAM']

**Matched Groups: 4/4**

### Game 7
**Ground Truth Groups:**
- ['LEDGER', 'LOG', 'RECORD', 'REGISTER']
- ['BROWN', 'DREW', 'HARDY', 'HOLMES']
- ['BALE', 'HORSE', 'PITCHFORK', 'TROUGH']
- ['CANOPY', 'DAY', 'MURPHY', 'WATER']

**Predictions Sequence:**
- One away: ['BROWN', 'DREW', 'HOLMES', 'MURPHY']
- Incorrect: ['BROWN', 'DAY', 'HOLMES', 'MURPHY']
- Incorrect: ['BROWN', 'DAY', 'DREW', 'MURPHY']
- Incorrect: ['DAY', 'DREW', 'HOLMES', 'MURPHY']

**Matched Groups: 0/4**

### Game 8
**Ground Truth Groups:**
- ['BEER', 'DIP', 'PIZZA', 'WINGS']
- ['AIR', 'AURA', 'HALO', 'VIBE']
- ['DWELL', 'HARP', 'INSIST', 'LINGER']
- ['BERNIE', 'PARISH', 'RIGATONI', 'ROMEO']

**Predictions Sequence:**
- One away: ['AURA', 'DWELL', 'INSIST', 'LINGER']
- Incorrect: ['AURA', 'DWELL', 'INSIST', 'VIBE']
- One away: ['DIP', 'DWELL', 'INSIST', 'LINGER']
- One away: ['AURA', 'DWELL', 'HALO', 'VIBE']

**Matched Groups: 0/4**

### Game 9
**Ground Truth Groups:**
- ['GRIT', 'HEART', 'NERVE', 'PLUCK']
- ['CONSTRUCTION', 'GRAPH', 'TISSUE', 'WAX']
- ['CARPET', 'CLAY', 'GRASS', 'HARD']
- ['ARTERY', 'AVENUE', 'DRAG', 'DRIVE']

**Predictions Sequence:**
- Incorrect: ['DRAG', 'DRIVE', 'HARD', 'HEART']
- Incorrect: ['DRAG', 'DRIVE', 'HARD', 'PLUCK']
- Incorrect: ['CONSTRUCTION', 'DRAG', 'DRIVE', 'HARD']
- Incorrect: ['DRIVE', 'GRIT', 'HARD', 'PLUCK']

**Matched Groups: 0/4**

### Game 10
**Ground Truth Groups:**
- ['CONE', 'CUBE', 'PYRAMID', 'SPHERE']
- ['CUCUMBER', 'EGGPLANT', 'PEPPER', 'TOMATO']
- ['JOB', 'LIMA', 'MOBILE', 'POLISH']
- ['MAC', 'PAD', 'PHONE', 'POD']

**Predictions Sequence:**
- Correct: ['CUCUMBER', 'EGGPLANT', 'PEPPER', 'TOMATO']
- Incorrect: ['JOB', 'MOBILE', 'PAD', 'PHONE']
- One away: ['JOB', 'LIMA', 'MOBILE', 'PHONE']
- Incorrect: ['JOB', 'MOBILE', 'PHONE', 'SPHERE']
- Incorrect: ['CONE', 'JOB', 'MOBILE', 'PHONE']

**Matched Groups: 1/4**

### Game 11
**Ground Truth Groups:**
- ['LIFT', 'LODGE', 'MOGUL', 'SLOPE']
- ['CLASS', 'HOMEROOM', 'LUNCH', 'RECESS']
- ['ANGLE', 'CYCLE', 'DENT', 'POD']
- ['EMOJI', 'GINKGO', 'KARAOKE', 'TYCOON']

**Predictions Sequence:**
- Incorrect: ['DENT', 'EMOJI', 'LIFT', 'SLOPE']
- Incorrect: ['ANGLE', 'EMOJI', 'LIFT', 'SLOPE']
- Incorrect: ['DENT', 'EMOJI', 'LIFT', 'POD']
- Incorrect: ['CYCLE', 'DENT', 'EMOJI', 'LIFT']

**Matched Groups: 0/4**

### Game 12
**Ground Truth Groups:**
- ['GATOR', 'SOONER', 'VOLUNTEER', 'WOLVERINE']
- ['BANANAS', 'COMMANDO', 'FIGURE', 'ROGUE']
- ['BERET', 'DERBY', 'PILLBOX', 'SNAPBACK']
- ['CAP', 'COVER', 'PLUG', 'SEAL']

**Predictions Sequence:**
- One away: ['COVER', 'PLUG', 'SEAL', 'SOONER']
- One away: ['CAP', 'COVER', 'PLUG', 'SOONER']
- Incorrect: ['CAP', 'COVER', 'FIGURE', 'SOONER']
- One away: ['CAP', 'COVER', 'SEAL', 'SOONER']

**Matched Groups: 0/4**

### Game 13
**Ground Truth Groups:**
- ['BALLOT', 'ROSTER', 'SLATE', 'TICKET']
- ['CAPSULE', 'CREAM', 'SYRUP', 'TABLET']
- ['COAT', 'GREEN', 'POD', 'SOUP']
- ['BUFFER', 'CUSHION', 'PAD', 'SHIELD']

**Predictions Sequence:**
- Incorrect: ['COAT', 'CREAM', 'SOUP', 'SYRUP']
- One away: ['COAT', 'CREAM', 'GREEN', 'SOUP']
- Incorrect: ['CREAM', 'GREEN', 'SOUP', 'SYRUP']
- Correct: ['BUFFER', 'CUSHION', 'PAD', 'SHIELD']
- Incorrect: ['COAT', 'CREAM', 'GREEN', 'SYRUP']

**Matched Groups: 1/4**

### Game 14
**Ground Truth Groups:**
- ['CEREAL', 'OMELET', 'PANCAKE', 'WAFFLE']
- ['EGG', 'JUROR', 'MONTH', 'ROSE']
- ['BACON', 'CLOSE', 'MUNCH', 'WHISTLER']
- ['CENTURY', 'DECADE', 'MILLENNIUM', 'YEAR']

**Predictions Sequence:**
- Incorrect: ['CLOSE', 'DECADE', 'MONTH', 'YEAR']
- One away: ['CENTURY', 'DECADE', 'MONTH', 'YEAR']
- Incorrect: ['DECADE', 'MONTH', 'ROSE', 'YEAR']
- One away: ['DECADE', 'MILLENNIUM', 'MONTH', 'YEAR']

**Matched Groups: 0/4**

### Game 15
**Ground Truth Groups:**
- ['BEAKER', 'DROPPER', 'GOGGLES', 'MICROSCOPE']
- ['CENTER', 'HEART', 'HUB', 'NUCLEUS']
- ['CELL', 'FORMULA', 'SHEET', 'SORT']
- ['BODY', 'LOVE', 'ROMANCE', 'SIGN']

**Predictions Sequence:**
- Incorrect: ['HEART', 'LOVE', 'ROMANCE', 'SORT']
- One away: ['LOVE', 'ROMANCE', 'SIGN', 'SORT']
- Incorrect: ['HEART', 'LOVE', 'SIGN', 'SORT']
- Incorrect: ['BODY', 'HEART', 'LOVE', 'SORT']

**Matched Groups: 0/4**

### Game 16
**Ground Truth Groups:**
- ['CAP', 'COVER', 'LID', 'TOP']
- ['HARBOR', 'HARP', 'HOODED', 'MONK']
- ['AVIATOR', 'CAT EYE', 'WAYFARER', 'WRAPAROUND']
- ['BASSOON', 'CLARINET', 'FLUTE', 'OBOE']

**Predictions Sequence:**
- Correct: ['BASSOON', 'CLARINET', 'FLUTE', 'OBOE']
- One away: ['CAP', 'CAT EYE', 'COVER', 'TOP']
- Incorrect: ['CAT EYE', 'COVER', 'MONK', 'TOP']
- One away: ['CAP', 'CAT EYE', 'COVER', 'LID']
- Incorrect: ['CAP', 'CAT EYE', 'COVER', 'MONK']

**Matched Groups: 1/4**

### Game 17
**Ground Truth Groups:**
- ['ACT', 'BIT', 'ROUTINE', 'SET']
- ['COCOA', 'COFFEE', 'MATE', 'TEA']
- ['DIRTY', 'DRY', 'TWIST', 'UP']
- ['BORING', 'DULL', 'MUNDANE', 'VANILLA']

**Predictions Sequence:**
- Incorrect: ['BIT', 'SET', 'TWIST', 'UP']
- One away: ['BIT', 'BORING', 'DULL', 'MUNDANE']
- Incorrect: ['BIT', 'BORING', 'DULL', 'TWIST']
- Incorrect: ['BIT', 'BORING', 'SET', 'UP']

**Matched Groups: 0/4**

### Game 18
**Ground Truth Groups:**
- ['HEYDAY', 'PINNACLE', 'PRIME', 'ZENITH']
- ['BAYWATCH', 'COOTIE', 'HERRING', 'KINGPIN']
- ['KALEIDOSCOPE', 'PEACOCK', 'RAINBOW', 'SUNSET']
- ['BOLD', 'HIGHLIGHT', 'STRIKETHROUGH', 'UNDERLINE']

**Predictions Sequence:**
- One away: ['HEYDAY', 'HIGHLIGHT', 'STRIKETHROUGH', 'UNDERLINE']
- One away: ['HIGHLIGHT', 'KALEIDOSCOPE', 'STRIKETHROUGH', 'UNDERLINE']
- One away: ['HEYDAY', 'PINNACLE', 'STRIKETHROUGH', 'ZENITH']
- Incorrect: ['HEYDAY', 'HIGHLIGHT', 'KALEIDOSCOPE', 'STRIKETHROUGH']

**Matched Groups: 0/4**

### Game 19
**Ground Truth Groups:**
- ['BEST', 'CHAMP', 'GOAT', 'LEGEND']
- ['HARE', 'I', 'MUSSEL', 'NAVAL']
- ['BARGE', 'CRAFT', 'SHIP', 'VESSEL']
- ['ABOUT', 'CONCERNING', 'ON', 'TOWARD']

**Predictions Sequence:**
- Correct: ['BARGE', 'CRAFT', 'SHIP', 'VESSEL']
- One away: ['ABOUT', 'BEST', 'CONCERNING', 'ON']
- Incorrect: ['ABOUT', 'BEST', 'I', 'ON']
- One away: ['ABOUT', 'CONCERNING', 'I', 'ON']
- Correct: ['ABOUT', 'CONCERNING', 'ON', 'TOWARD']
- One away: ['BEST', 'CHAMP', 'I', 'LEGEND']

**Matched Groups: 2/4**

### Game 20
**Ground Truth Groups:**
- ['BUNNY', 'EGG', 'ISLAND', 'SUNDAY']
- ['COUNTER', 'MIXER', 'RANGE', 'SINK']
- ['BASH', 'BLOWOUT', 'PARTY', 'SHINDIG']
- ['CRICKET', 'PUPPET', 'WHALE', 'WOODCARVER']

**Predictions Sequence:**
- One away: ['BASH', 'BLOWOUT', 'SHINDIG', 'SINK']
- Incorrect: ['BASH', 'COUNTER', 'SHINDIG', 'SINK']
- One away: ['COUNTER', 'RANGE', 'SINK', 'SUNDAY']
- One away: ['BASH', 'BLOWOUT', 'BUNNY', 'SHINDIG']

**Matched Groups: 0/4**

### Game 21
**Ground Truth Groups:**
- ['ARES', 'EARS', 'SEAR', 'SERA']
- ['BOWL', 'BUNNY', 'DEVIL', 'JACKET']
- ['ARE', 'ELLE', 'QUE', 'QUEUE']
- ['GNOME', 'GRILL', 'SHED', 'SPRINKLER']

**Predictions Sequence:**
- One away: ['BUNNY', 'DEVIL', 'EARS', 'JACKET']
- Incorrect: ['ARE', 'BUNNY', 'DEVIL', 'EARS']
- Incorrect: ['ARE', 'DEVIL', 'EARS', 'SHED']
- Incorrect: ['ARE', 'DEVIL', 'EARS', 'JACKET']

**Matched Groups: 0/4**

### Game 22
**Ground Truth Groups:**
- ['DOG', 'HEEL', 'JERK', 'SNAKE']
- ['COME', 'DOWN', 'SIT', 'STAY']
- ['ARCH', 'BALL', 'SOLE', 'TOE']
- ['BASS', 'HARP', 'HORN', 'ORGAN']

**Predictions Sequence:**
- Correct: ['COME', 'DOWN', 'SIT', 'STAY']
- One away: ['BALL', 'HEEL', 'SOLE', 'TOE']
- Correct: ['BASS', 'HARP', 'HORN', 'ORGAN']
- Incorrect: ['BALL', 'DOG', 'HEEL', 'TOE']
- One away: ['ARCH', 'HEEL', 'SOLE', 'TOE']
- One away: ['DOG', 'HEEL', 'SNAKE', 'TOE']

**Matched Groups: 2/4**

### Game 23
**Ground Truth Groups:**
- ['BONUS', 'EXTRA', 'GRAVY', 'ICING']
- ['FESTIVUS', 'REGIFTING', 'SHRINKAGE', 'YADA YADA']
- ['CONDENSATION', 'FREEZING', 'MELTING', 'VAPORIZATION']
- ['CONTRACTION', 'FOOT', 'POSSESSIVE', 'QUOTE']

**Predictions Sequence:**
- One away: ['FREEZING', 'MELTING', 'SHRINKAGE', 'VAPORIZATION']
- One away: ['CONDENSATION', 'MELTING', 'SHRINKAGE', 'VAPORIZATION']
- One away: ['CONDENSATION', 'FREEZING', 'MELTING', 'SHRINKAGE']
- Incorrect: ['FREEZING', 'ICING', 'MELTING', 'SHRINKAGE']

**Matched Groups: 0/4**

### Game 24
**Ground Truth Groups:**
- ['GUSTO', 'PASSION', 'RELISH', 'ZEST']
- ['BRICK', 'FISH TANK', 'MICROWAVE', 'SHOEBOX']
- ['JENNY', 'LIME', 'MORTAR', 'PICKLE']
- ['BEAUCOUP', 'MOLTO', 'MUCHO', 'MULTI']

**Predictions Sequence:**
- Correct: ['GUSTO', 'PASSION', 'RELISH', 'ZEST']
- Incorrect: ['BRICK', 'FISH TANK', 'LIME', 'PICKLE']
- Incorrect: ['FISH TANK', 'JENNY', 'PICKLE', 'SHOEBOX']
- Incorrect: ['BRICK', 'FISH TANK', 'LIME', 'MORTAR']
- One away: ['FISH TANK', 'JENNY', 'LIME', 'PICKLE']

**Matched Groups: 1/4**

### Game 25
**Ground Truth Groups:**
- ['GEAR', 'HANDLEBAR', 'PEDAL', 'WHEEL']
- ['BEARD', 'GOATEE', 'MUSTACHE', 'STUBBLE']
- ['DOG', 'FOLLOW', 'TAIL', 'TRACK']
- ['DOUBT', 'MOVIE', 'SHADOW', 'VOTE']

**Predictions Sequence:**
- Correct: ['BEARD', 'GOATEE', 'MUSTACHE', 'STUBBLE']
- Incorrect: ['DOUBT', 'FOLLOW', 'SHADOW', 'TRACK']
- One away: ['DOUBT', 'FOLLOW', 'MOVIE', 'SHADOW']
- One away: ['DOUBT', 'FOLLOW', 'SHADOW', 'VOTE']
- Incorrect: ['DOUBT', 'FOLLOW', 'MOVIE', 'TRACK']

**Matched Groups: 1/4**

### Game 26
**Ground Truth Groups:**
- ['COOLER', 'LANTERN', 'SLEEPING BAG', 'TENT']
- ['CAMPER', 'HOUR', 'MEAL', 'MEDIUM']
- ['BARB', 'DIG', 'DISS', 'JAB']
- ['MINUTE', 'SLIGHT', 'SMALL', 'WEE']

**Predictions Sequence:**
- Incorrect: ['CAMPER', 'SLEEPING BAG', 'SMALL', 'TENT']
- Incorrect: ['HOUR', 'SLEEPING BAG', 'SMALL', 'TENT']
- Incorrect: ['MEAL', 'SLEEPING BAG', 'SMALL', 'TENT']
- Incorrect: ['SLEEPING BAG', 'SLIGHT', 'SMALL', 'TENT']

**Matched Groups: 0/4**

### Game 27
**Ground Truth Groups:**
- ['AD-LIB', 'FREESTYLE', 'IMPROV', 'MAKE UP']
- ['COSMETIC', 'EXTERNAL', 'SHALLOW', 'SURFACE']
- ['BABBLE', 'COO', 'CRAWL', 'NURSE']
- ['BUTTERFLY', 'DOMINO', 'PLACEBO', 'SIDE']

**Predictions Sequence:**
- Incorrect: ['MAKE UP', 'SHALLOW', 'SIDE', 'SURFACE']
- Incorrect: ['CRAWL', 'MAKE UP', 'SIDE', 'SURFACE']
- Incorrect: ['CRAWL', 'MAKE UP', 'SHALLOW', 'SIDE']
- Incorrect: ['EXTERNAL', 'MAKE UP', 'SIDE', 'SURFACE']

**Matched Groups: 0/4**

### Game 28
**Ground Truth Groups:**
- ['DRIVER', 'EBAY', 'FINLET', 'FLAKE']
- ['ADDRESS', 'EMAIL', 'NAME', 'NUMBER']
- ['APPEAL', 'BID', 'CALL', 'PETITION']
- ['FISHEYE', 'MACRO', 'TELEPHOTO', 'ZOOM']

**Predictions Sequence:**
- Incorrect: ['ADDRESS', 'CALL', 'FINLET', 'NAME']
- Incorrect: ['CALL', 'FINLET', 'NAME', 'NUMBER']
- Incorrect: ['ADDRESS', 'CALL', 'FINLET', 'NUMBER']
- One away: ['APPEAL', 'CALL', 'FINLET', 'PETITION']

**Matched Groups: 0/4**

### Game 29
**Ground Truth Groups:**
- ['GRANDSTAND', 'POSTURE', 'SHOWBOAT', 'SWAGGER']
- ['FINE', 'HUNKY-DORY', 'OK', 'SWELL']
- ['CHARM', 'FRIENDSHIP', 'ID', 'TENNIS']
- ['LAW', 'MEDICINE', 'SELF-CARE', 'WITCHCRAFT']

**Predictions Sequence:**
- Correct: ['FINE', 'HUNKY-DORY', 'OK', 'SWELL']
- One away: ['CHARM', 'POSTURE', 'SHOWBOAT', 'SWAGGER']
- One away: ['CHARM', 'GRANDSTAND', 'SHOWBOAT', 'SWAGGER']
- Incorrect: ['CHARM', 'FRIENDSHIP', 'POSTURE', 'SWAGGER']
- Incorrect: ['CHARM', 'FRIENDSHIP', 'SHOWBOAT', 'SWAGGER']

**Matched Groups: 1/4**

### Game 30
**Ground Truth Groups:**
- ['CANDY', 'CHAPSTICK', 'CHARGING CABLE', 'MAGAZINE']
- ['ASSESS', 'CHARGE', 'FINE', 'LEVY']
- ['EPIC', 'FEATURE', 'FILM SERIES', 'SHORT']
- ['DIRT', 'HIGH', 'ROCKY', 'SILK']

**Predictions Sequence:**
- One away: ['CHARGING CABLE', 'FEATURE', 'FILM SERIES', 'SHORT']
- Incorrect: ['CHARGE', 'CHARGING CABLE', 'FEATURE', 'FILM SERIES']
- Correct: ['EPIC', 'FEATURE', 'FILM SERIES', 'SHORT']
- Incorrect: ['CHARGE', 'CHARGING CABLE', 'FINE', 'HIGH']
- One away: ['CHARGE', 'CHARGING CABLE', 'FINE', 'LEVY']

**Matched Groups: 1/4**

### Game 31
**Ground Truth Groups:**
- ['AIOLI', 'BARBECUE', 'MARINARA', 'RANCH']
- ['EXAMINE', 'GRILL', 'PROBE', 'QUESTION']
- ['BILL', 'INVOICE', 'RECEIPT', 'STATEMENT']
- ['ANCESTRY', 'BROTH', 'CATTLE', 'MERCHANDISE']

**Predictions Sequence:**
- One away: ['EXAMINE', 'PROBE', 'QUESTION', 'STATEMENT']
- Incorrect: ['BILL', 'EXAMINE', 'QUESTION', 'STATEMENT']
- Incorrect: ['EXAMINE', 'QUESTION', 'RECEIPT', 'STATEMENT']
- One away: ['BILL', 'EXAMINE', 'PROBE', 'QUESTION']

**Matched Groups: 0/4**

### Game 32
**Ground Truth Groups:**
- ['CONFER', 'GRANT', 'PRESENT', 'VEST']
- ['DRAW', 'EVEN', 'SQUARE', 'TIE']
- ['DOODLE', 'DOZE', 'PASS NOTES', 'SPACE']
- ['DEGREE', 'MARVEL', 'PAC-MAN', 'PAINT']

**Predictions Sequence:**
- Incorrect: ['DRAW', 'EVEN', 'PASS NOTES', 'PRESENT']
- Incorrect: ['EVEN', 'GRANT', 'PASS NOTES', 'PRESENT']
- Incorrect: ['EVEN', 'PASS NOTES', 'PRESENT', 'TIE']
- Incorrect: ['EVEN', 'PAINT', 'PASS NOTES', 'PRESENT']

**Matched Groups: 0/4**

### Game 33
**Ground Truth Groups:**
- ['FLUKE', 'MULLET', 'SOLE', 'TANG']
- ['POUND', 'REAL', 'SOL', 'YUAN']
- ['BUTTER', 'DIVA', 'SOLO', 'WORLD']
- ['GOING', 'ONCE', 'SOLD', 'TWICE']

**Predictions Sequence:**
- One away: ['GOING', 'ONCE', 'REAL', 'TWICE']
- One away: ['GOING', 'ONCE', 'TWICE', 'WORLD']
- Correct: ['GOING', 'ONCE', 'SOLD', 'TWICE']
- Incorrect: ['DIVA', 'FLUKE', 'REAL', 'SOLE']
- Incorrect: ['FLUKE', 'REAL', 'SOLE', 'WORLD']

**Matched Groups: 1/4**

### Game 34
**Ground Truth Groups:**
- ['BUY', 'DEAL', 'STEAL', 'VALUE']
- ['BALL', 'CHALK', 'CUE', 'RACK']
- ['BIT', 'DASH', 'PINCH', 'TOUCH']
- ['FACE', 'MAC', 'MATCH', 'POCKET']

**Predictions Sequence:**
- Incorrect: ['BIT', 'CUE', 'FACE', 'TOUCH']
- Incorrect: ['BIT', 'DEAL', 'FACE', 'TOUCH']
- Incorrect: ['BALL', 'BIT', 'CUE', 'TOUCH']
- Incorrect: ['BALL', 'BIT', 'FACE', 'TOUCH']

**Matched Groups: 0/4**

### Game 35
**Ground Truth Groups:**
- ['DEFENSE', 'EXCUSE', 'OUT', 'REASON']
- ['BLAST', 'KNOCK', 'SLAM', 'TRASH']
- ['DATE', 'DEAD', 'DIRTY', 'DOH']
- ['APOLOGIES', 'OOPS', 'PARDON', 'SORRY']

**Predictions Sequence:**
- One away: ['EXCUSE', 'OUT', 'REASON', 'SORRY']
- One away: ['DATE', 'EXCUSE', 'OUT', 'REASON']
- One away: ['EXCUSE', 'KNOCK', 'OUT', 'REASON']
- Incorrect: ['APOLOGIES', 'EXCUSE', 'REASON', 'SORRY']

**Matched Groups: 0/4**

### Game 36
**Ground Truth Groups:**
- ['AIR', 'FEELING', 'MOOD', 'QUALITY']
- ['CUTE', 'FRESH', 'SMART', 'WISE']
- ['MERMAID', 'PRINCE', 'RASCALS', 'TRAMP']
- ['BAR', 'BEL', 'LUX', 'MOLE']

**Predictions Sequence:**
- One away: ['CUTE', 'FEELING', 'SMART', 'WISE']
- One away: ['CUTE', 'QUALITY', 'SMART', 'WISE']
- Incorrect: ['FEELING', 'QUALITY', 'SMART', 'WISE']
- One away: ['CUTE', 'SMART', 'TRAMP', 'WISE']

**Matched Groups: 0/4**

### Game 37
**Ground Truth Groups:**
- ['BENCHMARK', 'GAUGE', 'STANDARD', 'YARDSTICK']
- ['BET', 'CALL', 'CHECK', 'FOLD']
- ['COMBINATION', 'DRY', 'NORMAL', 'OILY']
- ['KENT', 'MARY', 'MASS', 'WASH']

**Predictions Sequence:**
- Incorrect: ['CALL', 'CHECK', 'COMBINATION', 'NORMAL']
- Incorrect: ['CHECK', 'COMBINATION', 'NORMAL', 'STANDARD']
- Incorrect: ['CALL', 'CHECK', 'COMBINATION', 'STANDARD']
- One away: ['BET', 'CALL', 'CHECK', 'COMBINATION']

**Matched Groups: 0/4**

### Game 38
**Ground Truth Groups:**
- ['BALL', 'FORMAL', 'HOP', 'RAVE']
- ['EXCLAMATION POINTS', 'HEART', 'QUESTION MARK', 'THUMBS UP']
- ['BAD REVIEW', 'CAMERA MOVEMENT', 'COOKWARE', 'SATYR']
- ['DOT', 'PERIOD', 'POINT', 'TITTLE']

**Predictions Sequence:**
- Incorrect: ['BAD REVIEW', 'CAMERA MOVEMENT', 'QUESTION MARK', 'THUMBS UP']
- Incorrect: ['BAD REVIEW', 'POINT', 'QUESTION MARK', 'THUMBS UP']
- One away: ['EXCLAMATION POINTS', 'POINT', 'QUESTION MARK', 'THUMBS UP']
- One away: ['BAD REVIEW', 'EXCLAMATION POINTS', 'QUESTION MARK', 'THUMBS UP']

**Matched Groups: 0/4**

### Game 39
**Ground Truth Groups:**
- ['PLANT', 'PRUNE', 'WATER', 'WEED']
- ['CASTLE', 'PAPER', 'PIPER', 'STONE']
- ['FLAT', 'MARY JANE', 'MULE', 'SLIDE']
- ['DWINDLE', 'FADE', 'PETER', 'TAPER']

**Predictions Sequence:**
- Incorrect: ['FADE', 'FLAT', 'PAPER', 'SLIDE']
- Incorrect: ['DWINDLE', 'FADE', 'FLAT', 'SLIDE']
- One away: ['DWINDLE', 'FADE', 'SLIDE', 'TAPER']
- Incorrect: ['FLAT', 'PAPER', 'SLIDE', 'STONE']

**Matched Groups: 0/4**

### Game 40
**Ground Truth Groups:**
- ['BLINK', 'MAROON', 'SUM', 'U']
- ['BOUNTY', 'CRUNCH', 'HEATH', 'MILKY WAY']
- ['ANVIL', 'COCHLEA', 'HAMMER', 'STIRRUP']
- ['DREAMCAST', 'GENESIS', 'SWITCH', 'WII']

**Predictions Sequence:**
- Incorrect: ['BLINK', 'HAMMER', 'MILKY WAY', 'SWITCH']
- Incorrect: ['ANVIL', 'HAMMER', 'MILKY WAY', 'SWITCH']
- Incorrect: ['BLINK', 'MAROON', 'MILKY WAY', 'SWITCH']
- Incorrect: ['ANVIL', 'BLINK', 'HAMMER', 'MILKY WAY']

**Matched Groups: 0/4**

### Game 41
**Ground Truth Groups:**
- ['FIRE', 'HEAT', 'KICK', 'SPICE']
- ['BRUSH', 'CAPE', 'CLIPPERS', 'GEL']
- ['BOOM', 'CRASH', 'ROAR', 'THUNDER']
- ['BASEBALL', 'MAGIC', 'SET', 'TAROT']

**Predictions Sequence:**
- Incorrect: ['BRUSH', 'CRASH', 'FIRE', 'SET']
- Incorrect: ['BRUSH', 'FIRE', 'MAGIC', 'SET']
- Incorrect: ['FIRE', 'MAGIC', 'SET', 'THUNDER']
- Incorrect: ['CRASH', 'FIRE', 'MAGIC', 'SET']

**Matched Groups: 0/4**

### Game 42
**Ground Truth Groups:**
- ['FISH', 'JUSTICE', 'LIBRA', 'SOLFEGE']
- ['ARENA', 'BOWL', 'COLISEUM', 'DOME']
- ['ALLEY', 'COURT', 'DRIVE', 'LANE']
- ['FOUNTAIN', 'GUTTER', 'TEAPOT', 'WHALE']

**Predictions Sequence:**
- One away: ['ALLEY', 'DRIVE', 'FOUNTAIN', 'LANE']
- One away: ['ALLEY', 'DRIVE', 'GUTTER', 'LANE']
- Incorrect: ['ALLEY', 'FOUNTAIN', 'GUTTER', 'LANE']
- One away: ['ALLEY', 'ARENA', 'DRIVE', 'LANE']

**Matched Groups: 0/4**

### Game 43
**Ground Truth Groups:**
- ['BLUE', 'GREEN', 'WHITE', 'YELLOW']
- ['MAIN', 'PARAMOUNT', 'PRIME', 'SUPREME']
- ['GRANDSTAND', 'PEACOCK', 'POSTURE', 'STRUT']
- ['CHAIN', 'COVER', 'LOVE', 'SCARLET']

**Predictions Sequence:**
- Correct: ['BLUE', 'GREEN', 'WHITE', 'YELLOW']
- One away: ['COVER', 'LOVE', 'PEACOCK', 'SCARLET']
- Incorrect: ['COVER', 'LOVE', 'MAIN', 'PEACOCK']
- One away: ['CHAIN', 'COVER', 'LOVE', 'MAIN']
- Incorrect: ['COVER', 'GRANDSTAND', 'LOVE', 'MAIN']

**Matched Groups: 1/4**

### Game 44
**Ground Truth Groups:**
- ['ASSOCIATION', 'GUILD', 'LEAGUE', 'UNION']
- ['CLINGY', 'SLEEK', 'SLINKY', 'SNUG']
- ['APOLOGY', 'DUBBING', 'PRAYER', 'PROPOSAL']
- ['CONTRACTION', 'FOOT', 'POSSESSIVE', 'QUOTATION']

**Predictions Sequence:**
- Correct: ['CLINGY', 'SLEEK', 'SLINKY', 'SNUG']
- One away: ['ASSOCIATION', 'GUILD', 'PROPOSAL', 'UNION']
- One away: ['ASSOCIATION', 'LEAGUE', 'PROPOSAL', 'UNION']
- Incorrect: ['APOLOGY', 'ASSOCIATION', 'PROPOSAL', 'UNION']
- Incorrect: ['ASSOCIATION', 'FOOT', 'PROPOSAL', 'UNION']

**Matched Groups: 1/4**

### Game 45
**Ground Truth Groups:**
- ['DOUBLE', 'SPARE', 'STRIKE', 'TURKEY']
- ['CURVE', 'SNAKE', 'WEAVE', 'WIND']
- ['BREAK', 'LEAVE', 'REST', 'VACATION']
- ['BLOW', 'CAT', 'GOLD', 'SWORD']

**Predictions Sequence:**
- One away: ['BREAK', 'LEAVE', 'REST', 'SPARE']
- Correct: ['BREAK', 'LEAVE', 'REST', 'VACATION']
- One away: ['BLOW', 'DOUBLE', 'SPARE', 'STRIKE']
- Incorrect: ['BLOW', 'DOUBLE', 'SPARE', 'SWORD']
- Incorrect: ['BLOW', 'DOUBLE', 'SPARE', 'WIND']

**Matched Groups: 1/4**

### Game 46
**Ground Truth Groups:**
- ['BABE', 'NAPOLEON', 'PIGLET', 'PORKY']
- ['DUCKLING', 'EMPEROR', 'MERMAID', 'PRINCESS']
- ['FAWN', 'FLATTER', 'GUSH', 'PRAISE']
- ['CALF', 'CHALK', 'COLONEL', 'WOULD']

**Predictions Sequence:**
- Incorrect: ['DUCKLING', 'FAWN', 'PIGLET', 'PORKY']
- Incorrect: ['DUCKLING', 'MERMAID', 'PIGLET', 'PORKY']
- Incorrect: ['FAWN', 'MERMAID', 'PIGLET', 'PORKY']
- One away: ['BABE', 'DUCKLING', 'PIGLET', 'PORKY']

**Matched Groups: 0/4**

### Game 47
**Ground Truth Groups:**
- ['COMPARE', 'CONNECTICUT', 'PARSNIP', 'WALLOP']
- ['ANKLE', 'COMPRESSION', 'CREW', 'DRESS']
- ['BASE', 'BOTTOM', 'FOOT', 'FOUNDATION']
- ['BUZZ', 'KNOCK', 'RING', 'SHOUT']

**Predictions Sequence:**
- Incorrect: ['BOTTOM', 'COMPARE', 'FOOT', 'KNOCK']
- One away: ['BUZZ', 'COMPARE', 'KNOCK', 'SHOUT']
- Incorrect: ['ANKLE', 'BOTTOM', 'FOOT', 'KNOCK']
- Incorrect: ['BOTTOM', 'COMPARE', 'KNOCK', 'SHOUT']

**Matched Groups: 0/4**

### Game 48
**Ground Truth Groups:**
- ['BIO', 'PLAYER', 'STAT', 'TEAM']
- ['BUNCH', 'GATHER', 'PUCKER', 'RUFFLE']
- ['LATER', 'NOW', 'SOON', 'THEN']
- ['FAST', 'FINGER', 'JUNK', 'SOUL']

**Predictions Sequence:**
- Correct: ['LATER', 'NOW', 'SOON', 'THEN']
- Incorrect: ['BUNCH', 'FAST', 'PLAYER', 'TEAM']
- Incorrect: ['BUNCH', 'GATHER', 'PLAYER', 'TEAM']
- Incorrect: ['BUNCH', 'FAST', 'GATHER', 'TEAM']
- Incorrect: ['FAST', 'GATHER', 'PLAYER', 'TEAM']

**Matched Groups: 1/4**

### Game 49
**Ground Truth Groups:**
- ['EQUIP', 'OUTFIT', 'PREPARE', 'READY']
- ['ABLE', 'CANE', 'EAVE', 'NOAA']
- ['DOWN', 'EAGER', 'GAME', 'WILLING']
- ['N.F.L.', 'NASA', 'PARAMOUNT', 'SUBARU']

**Predictions Sequence:**
- Incorrect: ['ABLE', 'EAGER', 'READY', 'WILLING']
- Incorrect: ['ABLE', 'PREPARE', 'READY', 'WILLING']
- Incorrect: ['ABLE', 'EAGER', 'PREPARE', 'READY']
- Incorrect: ['EAGER', 'PREPARE', 'READY', 'WILLING']

**Matched Groups: 0/4**

### Game 50
**Ground Truth Groups:**
- ['DROVE', 'HOST', 'LOAD', 'SCORE']
- ['BELL', 'DIESEL', 'SINGER', 'WATT']
- ['BOOT', 'POINTS', 'TICKET', 'TOW']
- ['BALL', 'BONE', 'FRISBEE', 'STICK']

**Predictions Sequence:**
- Incorrect: ['BALL', 'DROVE', 'SCORE', 'STICK']
- One away: ['BALL', 'DROVE', 'LOAD', 'SCORE']
- Incorrect: ['BALL', 'DROVE', 'LOAD', 'STICK']
- Incorrect: ['BALL', 'BOOT', 'DROVE', 'STICK']

**Matched Groups: 0/4**

### Game 51
**Ground Truth Groups:**
- ['BI', 'BUY', 'BY', 'BYE']
- ['AB', 'PEC', 'QUAD', 'TRI']
- ['DUB', 'EMO', 'POP', 'TRAP']
- ['HI', 'LO', 'MED', 'OFF']

**Predictions Sequence:**
- Incorrect: ['BUY', 'BY', 'OFF', 'TRAP']
- Incorrect: ['BY', 'OFF', 'POP', 'TRAP']
- Incorrect: ['BUY', 'BY', 'OFF', 'POP']
- Incorrect: ['BUY', 'OFF', 'POP', 'TRAP']

**Matched Groups: 0/4**

### Game 52
**Ground Truth Groups:**
- ['BANK', 'POOL', 'RESERVE', 'STORE']
- ['BRAKE', 'PARK', 'SIGNAL', 'TURN']
- ['COACH', 'GUIDE', 'SCHOOL', 'TRAIN']
- ['BOOK', 'EARTH', 'GLOW', 'INCH']

**Predictions Sequence:**
- Incorrect: ['BOOK', 'EARTH', 'GUIDE', 'TURN']
- Incorrect: ['BOOK', 'GUIDE', 'STORE', 'TURN']
- Incorrect: ['BOOK', 'GUIDE', 'TRAIN', 'TURN']
- Incorrect: ['BOOK', 'GUIDE', 'SIGNAL', 'TURN']

**Matched Groups: 0/4**

### Game 53
**Ground Truth Groups:**
- ['HENRY', 'JENNIFER', 'KATE', 'ROCK']
- ['GLASS', 'METAL', 'PAPER', 'PLASTIC']
- ['CENTRAL', 'CRITICAL', 'KEY', 'VITAL']
- ['ASSIGNMENT', 'DEFEAT', 'TEMPO', 'TIRED']

**Predictions Sequence:**
- Correct: ['CENTRAL', 'CRITICAL', 'KEY', 'VITAL']
- Correct: ['GLASS', 'METAL', 'PAPER', 'PLASTIC']
- Incorrect: ['ASSIGNMENT', 'JENNIFER', 'KATE', 'TIRED']
- One away: ['HENRY', 'JENNIFER', 'KATE', 'TIRED']
- One away: ['ASSIGNMENT', 'HENRY', 'JENNIFER', 'KATE']
- Incorrect: ['DEFEAT', 'JENNIFER', 'KATE', 'TIRED']

**Matched Groups: 2/4**

### Game 54
**Ground Truth Groups:**
- ['BALLOON', 'BANNER', 'CONFETTI', 'GARLAND']
- ['CREATIVE', 'FRESH', 'NOVEL', 'ORIGINAL']
- ['AMBASSADOR', 'INFLUENCER', 'MODEL', 'STREAMER']
- ['BOOK', 'CHARTER', 'RESERVE', 'SECURE']

**Predictions Sequence:**
- One away: ['BOOK', 'CREATIVE', 'NOVEL', 'ORIGINAL']
- Incorrect: ['BOOK', 'MODEL', 'NOVEL', 'ORIGINAL']
- Incorrect: ['BOOK', 'GARLAND', 'NOVEL', 'ORIGINAL']
- Incorrect: ['BOOK', 'NOVEL', 'ORIGINAL', 'SECURE']

**Matched Groups: 0/4**

### Game 55
**Ground Truth Groups:**
- ['ATLAS', 'ECHO', 'HELEN', 'PAN']
- ['CANDLE', 'JOHNSON', 'LOOFAH', 'TOILETRIES']
- ['DIRECT', 'GUIDE', 'LEAD', 'SHEPHERD']
- ['COMPASS', 'PINE TREE', 'SEWING KIT', 'TURNTABLE']

**Predictions Sequence:**
- Incorrect: ['HELEN', 'JOHNSON', 'SEWING KIT', 'SHEPHERD']
- Incorrect: ['GUIDE', 'JOHNSON', 'SEWING KIT', 'SHEPHERD']
- Incorrect: ['GUIDE', 'HELEN', 'JOHNSON', 'SHEPHERD']
- Incorrect: ['ECHO', 'HELEN', 'JOHNSON', 'SHEPHERD']

**Matched Groups: 0/4**

### Game 56
**Ground Truth Groups:**
- ['DASH', 'DROP', 'PINCH', 'SPLASH']
- ['BOAT', 'CAR', 'PLANE', 'TRAIN']
- ['DOWN', 'GAME', 'IN', 'ON BOARD']
- ['BLUE', 'GOOSE', 'RASP', 'STRAW']

**Predictions Sequence:**
- One away: ['DOWN', 'DROP', 'IN', 'ON BOARD']
- Correct: ['DOWN', 'GAME', 'IN', 'ON BOARD']
- Correct: ['BOAT', 'CAR', 'PLANE', 'TRAIN']
- One away: ['DASH', 'DROP', 'RASP', 'SPLASH']
- One away: ['BLUE', 'DASH', 'DROP', 'SPLASH']
- Correct: ['DASH', 'DROP', 'PINCH', 'SPLASH']
- Correct: ['BLUE', 'GOOSE', 'RASP', 'STRAW']

**Matched Groups: 4/4**

### Game 57
**Ground Truth Groups:**
- ['BED', 'PARCEL', 'PATCH', 'PLOT']
- ['CHIP', 'ROCKY', 'SCRATCHY', 'STITCH']
- ['BUMPY', 'ROUGH', 'RUGGED', 'UNEVEN']
- ['AMUSEMENT', 'NATIONAL', 'PARALLEL', 'SOUTH']

**Predictions Sequence:**
- One away: ['BUMPY', 'PATCH', 'ROUGH', 'UNEVEN']
- Incorrect: ['BUMPY', 'PATCH', 'ROCKY', 'ROUGH']
- Incorrect: ['PATCH', 'ROCKY', 'ROUGH', 'RUGGED']
- One away: ['BUMPY', 'PATCH', 'ROUGH', 'RUGGED']

**Matched Groups: 0/4**

### Game 58
**Ground Truth Groups:**
- ['DRUM', 'MARK', 'WAX', 'WIG']
- ['EXAMPLE', 'IDEAL', 'MODEL', 'SYMBOL']
- ['CAR', 'CONDUCTOR', 'STATION', 'TRACK']
- ['CYMBAL', 'SCIMITAR', 'SIMMER', 'SYMPHONY']

**Predictions Sequence:**
- Correct: ['EXAMPLE', 'IDEAL', 'MODEL', 'SYMBOL']
- Incorrect: ['CAR', 'DRUM', 'MARK', 'TRACK']
- One away: ['CAR', 'MARK', 'STATION', 'TRACK']
- One away: ['CAR', 'DRUM', 'STATION', 'TRACK']
- Incorrect: ['CONDUCTOR', 'DRUM', 'SYMPHONY', 'TRACK']

**Matched Groups: 1/4**

### Game 59
**Ground Truth Groups:**
- ['CHEF', 'GARDEN', 'GREEK', 'WEDGE']
- ['ARTS', 'BUSINESS', 'COMICS', 'SPORTS']
- ['EAGLE', 'MONTICELLO', 'SHIELD', 'TORCH']
- ['BARK', 'CROWN', 'RINGS', 'ROOTS']

**Predictions Sequence:**
- One away: ['ARTS', 'BUSINESS', 'GARDEN', 'SPORTS']
- One away: ['ARTS', 'BUSINESS', 'ROOTS', 'SPORTS']
- Incorrect: ['ARTS', 'BUSINESS', 'GARDEN', 'ROOTS']
- Incorrect: ['BUSINESS', 'GARDEN', 'ROOTS', 'SPORTS']

**Matched Groups: 0/4**

### Game 60
**Ground Truth Groups:**
- ['BALL', 'CURL', 'DOODLE', 'PUFF']
- ['CRAWL', 'CREEP', 'DRAG', 'INCH']
- ['BOTTLE', 'CAN', 'DRAFT', 'TAP']
- ['BUTTERFLY', 'DOMINO', 'HALO', 'PLACEBO']

**Predictions Sequence:**
- One away: ['CAN', 'CRAWL', 'CREEP', 'DRAG']
- Incorrect: ['CAN', 'CREEP', 'DRAG', 'TAP']
- One away: ['CRAWL', 'CREEP', 'CURL', 'DRAG']
- Incorrect: ['CAN', 'CRAWL', 'DRAG', 'TAP']

**Matched Groups: 0/4**

### Game 61
**Ground Truth Groups:**
- ['DAISY', 'ROSE', 'TULIP', 'VIOLET']
- ['ASTER', 'CARPENTER', 'CRAVEN', 'WAN']
- ['BARN', 'CHICKEN', 'FARMER', 'TRACTOR']
- ['DUST', 'LIFE', 'SPORTS', 'YELLOW']

**Predictions Sequence:**
- One away: ['BARN', 'CARPENTER', 'FARMER', 'TRACTOR']
- Incorrect: ['BARN', 'CARPENTER', 'FARMER', 'LIFE']
- Incorrect: ['CARPENTER', 'FARMER', 'LIFE', 'TRACTOR']
- Incorrect: ['CARPENTER', 'FARMER', 'LIFE', 'VIOLET']

**Matched Groups: 0/4**

### Game 62
**Ground Truth Groups:**
- ['BURN', 'KINDLE', 'LIGHT', 'TORCH']
- ['DATA', 'INFO', 'INTEL', 'NEWS']
- ['DELL', 'GLEN', 'HOLLOW', 'VALLEY']
- ['APPLE', 'COMPUTER', 'PLANET', 'REACTOR']

**Predictions Sequence:**
- Incorrect: ['APPLE', 'COMPUTER', 'DELL', 'INTEL']
- Incorrect: ['APPLE', 'COMPUTER', 'DATA', 'INTEL']
- Incorrect: ['COMPUTER', 'DATA', 'DELL', 'INTEL']
- Incorrect: ['APPLE', 'COMPUTER', 'INFO', 'INTEL']

**Matched Groups: 0/4**

### Game 63
**Ground Truth Groups:**
- ['ARROW', 'BOW', 'QUIVER', 'TARGET']
- ['BRIDGE', 'GIN', 'SPIT', 'WAR']
- ['ANCHOR', 'DRAGON', 'HEART', 'ROSE']
- ['BELT', 'CLOCK', 'DECK', 'SLUG']

**Predictions Sequence:**
- One away: ['ARROW', 'BOW', 'DRAGON', 'QUIVER']
- One away: ['ARROW', 'BOW', 'QUIVER', 'SPIT']
- One away: ['ARROW', 'BOW', 'HEART', 'QUIVER']
- One away: ['ARROW', 'BOW', 'DECK', 'QUIVER']

**Matched Groups: 0/4**

### Game 64
**Ground Truth Groups:**
- ['BUNK', 'CROCK', 'HOGWASH', 'HORSEFEATHERS']
- ['GOATEE', 'HORNS', 'PITCHFORK', 'TAIL']
- ['BATON', 'HAMMER', 'HURDLE', 'POLE']
- ['BEND', 'BOWLINE', 'HITCH', 'SHEEPSHANK']

**Predictions Sequence:**
- One away: ['CROCK', 'HOGWASH', 'HORSEFEATHERS', 'SHEEPSHANK']
- Incorrect: ['HITCH', 'HOGWASH', 'HORSEFEATHERS', 'SHEEPSHANK']
- One away: ['BUNK', 'HOGWASH', 'HORSEFEATHERS', 'SHEEPSHANK']
- Incorrect: ['HAMMER', 'HOGWASH', 'HORSEFEATHERS', 'SHEEPSHANK']

**Matched Groups: 0/4**

### Game 65
**Ground Truth Groups:**
- ['CHARM', 'ENTRANCE', 'RIVET', 'THRILL']
- ['CLIP', 'EXTRACT', 'PASSAGE', 'QUOTE']
- ['BANGLE', 'CHICK', 'GO-GO', 'SUPREME']
- ['BOARD', 'FISH', 'GATE', 'STRUCK']

**Predictions Sequence:**
- Incorrect: ['ENTRANCE', 'GATE', 'PASSAGE', 'STRUCK']
- Incorrect: ['ENTRANCE', 'GATE', 'PASSAGE', 'THRILL']
- Incorrect: ['BOARD', 'ENTRANCE', 'GATE', 'PASSAGE']
- Incorrect: ['ENTRANCE', 'GATE', 'PASSAGE', 'QUOTE']

**Matched Groups: 0/4**

### Game 66
**Ground Truth Groups:**
- ['DAY', 'FLOYD', 'SABBATH', 'STRIPES']
- ['CHECKERS', 'OUTBACK', 'SONIC', 'SUBWAY']
- ['BOAT', 'CAR', 'PLANE', 'TRAIN']
- ['KING', 'MAGIC', 'SUN', 'THUNDER']

**Predictions Sequence:**
- One away: ['BOAT', 'CAR', 'DAY', 'TRAIN']
- Correct: ['BOAT', 'CAR', 'PLANE', 'TRAIN']
- Incorrect: ['DAY', 'MAGIC', 'SONIC', 'THUNDER']
- Incorrect: ['DAY', 'FLOYD', 'MAGIC', 'THUNDER']
- One away: ['DAY', 'KING', 'MAGIC', 'THUNDER']

**Matched Groups: 1/4**

### Game 67
**Ground Truth Groups:**
- ['BLOW', 'BUMBLE', 'FLUFF', 'SPOIL']
- ['FRAME', 'HANDLE', 'HINGE', 'LOCK']
- ['KINDLING', 'LOG', 'MATCH', 'TINDER']
- ['DRILL', 'GRINDER', 'ROUTER', 'SAW']

**Predictions Sequence:**
- Incorrect: ['FRAME', 'HANDLE', 'LOG', 'SAW']
- Incorrect: ['BLOW', 'FRAME', 'HANDLE', 'SAW']
- Incorrect: ['BLOW', 'HANDLE', 'SAW', 'SPOIL']
- Incorrect: ['BLOW', 'HANDLE', 'HINGE', 'SAW']

**Matched Groups: 0/4**

### Game 68
**Ground Truth Groups:**
- ['BOW', 'FELLOW', 'HORN', 'LEGS']
- ['AFFILIATE', 'ASSOCIATE', 'EQUATE', 'RELATE']
- ['LIBERTY', 'STORM', 'SUN', 'WINGS']
- ['BAIT', 'CHUM', 'FLY', 'LURE']

**Predictions Sequence:**
- Incorrect: ['BAIT', 'EQUATE', 'LURE', 'RELATE']
- One away: ['ASSOCIATE', 'EQUATE', 'FELLOW', 'RELATE']
- Incorrect: ['EQUATE', 'LIBERTY', 'LURE', 'RELATE']
- Incorrect: ['EQUATE', 'FELLOW', 'LURE', 'RELATE']

**Matched Groups: 0/4**

### Game 69
**Ground Truth Groups:**
- ['BLOW', 'LOSE', 'MISS', 'WASTE']
- ['CAMO', 'DRAB', 'FATIGUE', 'KHAKI']
- ['HUB', 'RIM', 'SPOKE', 'TIRE']
- ['DUCT', 'EXHAUST', 'HOSE', 'PIPE']

**Predictions Sequence:**
- One away: ['BLOW', 'LOSE', 'MISS', 'SPOKE']
- Correct: ['DUCT', 'EXHAUST', 'HOSE', 'PIPE']
- One away: ['BLOW', 'LOSE', 'MISS', 'TIRE']
- Incorrect: ['BLOW', 'FATIGUE', 'LOSE', 'SPOKE']
- One away: ['BLOW', 'HUB', 'LOSE', 'MISS']

**Matched Groups: 1/4**

### Game 70
**Ground Truth Groups:**
- ['COSMOPOLITAN', 'GREYHOUND', 'SCREWDRIVER', 'SEA BREEZE']
- ['CLOWNFISH', 'GRUMPY OLD MAN', 'RACECAR', 'TALKING DOLL']
- ['BOLT CUTTER', 'CHOW MEIN', 'SCARF RING', 'WOLF EEL']
- ['AWARENESS RIBBON', 'BOUTONNIÈRE', 'LAPEL PIN', 'LAVALIER']

**Predictions Sequence:**
- Incorrect: ['BOUTONNIÈRE', 'GRUMPY OLD MAN', 'LAVALIER', 'TALKING DOLL']
- Incorrect: ['BOUTONNIÈRE', 'LAVALIER', 'SCARF RING', 'TALKING DOLL']
- Incorrect: ['BOUTONNIÈRE', 'GRUMPY OLD MAN', 'LAVALIER', 'SCARF RING']
- Incorrect: ['BOLT CUTTER', 'BOUTONNIÈRE', 'GRUMPY OLD MAN', 'LAVALIER']

**Matched Groups: 0/4**

### Game 71
**Ground Truth Groups:**
- ['ABOUT-FACE', 'BACKPEDAL', 'FLIP-FLOP', 'RENEGE']
- ['JAM', 'SHOEHORN', 'STUFF', 'WEDGE']
- ['KANGAROO', 'NEIGHBORHOOD', 'RAINBOW', 'STREET']
- ['FLAG', 'PINWHEEL', 'VANE', 'WIND CHIME']

**Predictions Sequence:**
- Incorrect: ['ABOUT-FACE', 'BACKPEDAL', 'PINWHEEL', 'SHOEHORN']
- One away: ['ABOUT-FACE', 'BACKPEDAL', 'RENEGE', 'SHOEHORN']
- Incorrect: ['ABOUT-FACE', 'BACKPEDAL', 'PINWHEEL', 'WIND CHIME']
- Incorrect: ['ABOUT-FACE', 'BACKPEDAL', 'SHOEHORN', 'WIND CHIME']

**Matched Groups: 0/4**

### Game 72
**Ground Truth Groups:**
- ['CROC', 'LOAFER', 'MOCCASIN', 'SLIPPER']
- ['BASIC', 'JAVA', 'PYTHON', 'RUBY']
- ['BOA', 'HEADDRESS', 'PILLOW', 'SHUTTLECOCK']
- ['COBRA', 'INSPIRATION', 'LIGHTNING', 'UNION']

**Predictions Sequence:**
- One away: ['CROC', 'LOAFER', 'MOCCASIN', 'SHUTTLECOCK']
- Incorrect: ['BOA', 'CROC', 'LOAFER', 'SHUTTLECOCK']
- Incorrect: ['BOA', 'LOAFER', 'MOCCASIN', 'SHUTTLECOCK']
- One away: ['BOA', 'CROC', 'LOAFER', 'MOCCASIN']

**Matched Groups: 0/4**

### Game 73
**Ground Truth Groups:**
- ['🍞', '🥓', '🥬', '🧀']
- ['👽', '😱', '🧛', '🪚']
- ['🐑', '🐝', '👁️', '🫖']
- ['✈️', '🌧️', '🚂', '🧠']

**Predictions Sequence:**
- Incorrect: ['🍞', '🧠', '🪚', '🫖']
- Incorrect: ['🍞', '😱', '🧠', '🫖']
- Incorrect: ['🍞', '🐑', '🧠', '🫖']
- Incorrect: ['🍞', '🥬', '🧠', '🫖']

**Matched Groups: 0/4**

### Game 74
**Ground Truth Groups:**
- ['CORAL', 'GARTER', 'KING', 'RATTLE']
- ['CLEFT', 'RENT', 'SPLIT', 'TORN']
- ['BOTTLE', 'CARAFE', 'DECANTER', 'GLASS']
- ['BOOTY', 'CLOSE', 'COLD', 'CURTAIN']

**Predictions Sequence:**
- Correct: ['BOTTLE', 'CARAFE', 'DECANTER', 'GLASS']
- Incorrect: ['CLOSE', 'COLD', 'SPLIT', 'TORN']
- Incorrect: ['CLOSE', 'CURTAIN', 'SPLIT', 'TORN']
- One away: ['CLOSE', 'RENT', 'SPLIT', 'TORN']
- One away: ['CLOSE', 'COLD', 'CURTAIN', 'SPLIT']

**Matched Groups: 1/4**

### Game 75
**Ground Truth Groups:**
- ['BROOD', 'MUSE', 'PONDER', 'REFLECT']
- ['INTERRUPT', 'LITTER', 'POINT', 'STARE']
- ['CRAFT', 'LINE', 'TRADE', 'WORK']
- ['CABBAGE', 'NICOTINE', 'ROUGH', 'SOUL']

**Predictions Sequence:**
- Incorrect: ['LINE', 'POINT', 'REFLECT', 'WORK']
- Incorrect: ['POINT', 'PONDER', 'REFLECT', 'WORK']
- Incorrect: ['POINT', 'REFLECT', 'ROUGH', 'WORK']
- Incorrect: ['LINE', 'POINT', 'ROUGH', 'WORK']

**Matched Groups: 0/4**

### Game 76
**Ground Truth Groups:**
- ['DOROTHY GALE', 'LEPRECHAUN', 'PRIDE', 'UNICORN']
- ['CONSIDER', 'COUNT', 'FACTOR', 'INCLUDE']
- ['FIEVEL', 'FOREHEAD', 'TOUCAN', 'WONDER']
- ['CLOWNFISH', 'HONEYBEE', 'TIGER', 'ZEBRA']

**Predictions Sequence:**
- One away: ['CONSIDER', 'FACTOR', 'INCLUDE', 'WONDER']
- Incorrect: ['CONSIDER', 'DOROTHY GALE', 'INCLUDE', 'WONDER']
- Incorrect: ['CONSIDER', 'INCLUDE', 'PRIDE', 'WONDER']
- One away: ['CONSIDER', 'COUNT', 'INCLUDE', 'WONDER']

**Matched Groups: 0/4**

### Game 77
**Ground Truth Groups:**
- ['ACE', 'CRACKERJACK', 'EXPERT', 'HOTSHOT']
- ['GLADIATOR', 'HER', 'JOKER', 'SIGNS']
- ['CHOCK', 'JACK', 'TIRE', 'WRENCH']
- ['POPCORN', 'RAPTURE', 'ROCKETRY', 'SOULMATE']

**Predictions Sequence:**
- Incorrect: ['CRACKERJACK', 'HOTSHOT', 'RAPTURE', 'SOULMATE']
- Incorrect: ['CHOCK', 'CRACKERJACK', 'HOTSHOT', 'SOULMATE']
- Incorrect: ['CRACKERJACK', 'GLADIATOR', 'HOTSHOT', 'SOULMATE']
- Incorrect: ['CHOCK', 'CRACKERJACK', 'RAPTURE', 'SOULMATE']

**Matched Groups: 0/4**

### Game 78
**Ground Truth Groups:**
- ['BUG', 'CHILL', 'COLD', 'COUGH']
- ['BREACH', 'DIVE', 'SING', 'SPOUT']
- ['MAGIC', 'MOTOR', 'SIN', 'WINDY']
- ['EASY', 'ENOUGH', 'QUIET', 'RELAX']

**Predictions Sequence:**
- Correct: ['EASY', 'ENOUGH', 'QUIET', 'RELAX']
- Incorrect: ['CHILL', 'COLD', 'SPOUT', 'WINDY']
- One away: ['CHILL', 'COLD', 'COUGH', 'WINDY']
- Incorrect: ['CHILL', 'COLD', 'DIVE', 'WINDY']
- One away: ['BUG', 'CHILL', 'COLD', 'WINDY']

**Matched Groups: 1/4**

### Game 79
**Ground Truth Groups:**
- ['COMPACT', 'DENSE', 'THICK', 'TIGHT']
- ['BRUSH', 'MIRROR', 'NAIL FILE', 'TWEEZERS']
- ['DELIBERATE', 'MUSE', 'NOODLE', 'REFLECT']
- ['CRANIUM', 'CRONUT', 'LUNAR', 'PUFNSTUF']

**Predictions Sequence:**
- Incorrect: ['CRONUT', 'MIRROR', 'NAIL FILE', 'REFLECT']
- Incorrect: ['CRANIUM', 'CRONUT', 'NAIL FILE', 'REFLECT']
- One away: ['CRANIUM', 'CRONUT', 'NAIL FILE', 'PUFNSTUF']
- Incorrect: ['CRONUT', 'DELIBERATE', 'NAIL FILE', 'REFLECT']

**Matched Groups: 0/4**

### Game 80
**Ground Truth Groups:**
- ['BRAVE', 'FLOW', 'FROZEN', 'UP']
- ['DUE', 'FAIR', 'JUST', 'RIGHT']
- ['COME', 'DOWN', 'HEEL', 'STAY']
- ['LEFT', 'PLACED', 'PUT', 'SET']

**Predictions Sequence:**
- Incorrect: ['COME', 'JUST', 'PUT', 'UP']
- Incorrect: ['DOWN', 'JUST', 'PUT', 'UP']
- Incorrect: ['COME', 'DOWN', 'JUST', 'UP']
- Incorrect: ['COME', 'DOWN', 'PUT', 'UP']

**Matched Groups: 0/4**

### Game 81
**Ground Truth Groups:**
- ['AMERICAN', 'BLUE', 'JACK', 'SWISS']
- ['DATE', 'DUTCH', 'JEOPARDY', 'SPACE']
- ['CUBAN', 'KITTEN', 'STILETTO', 'WEDGE']
- ['HAMMER', 'HURDLE', 'JAVELIN', 'POLE']

**Predictions Sequence:**
- Incorrect: ['DATE', 'HAMMER', 'JACK', 'JEOPARDY']
- Incorrect: ['AMERICAN', 'DATE', 'HAMMER', 'JACK']
- Incorrect: ['AMERICAN', 'DATE', 'JACK', 'JEOPARDY']
- Incorrect: ['DATE', 'HAMMER', 'HURDLE', 'JEOPARDY']

**Matched Groups: 0/4**

### Game 82
**Ground Truth Groups:**
- ['BUCKLE', 'BUTTON', 'SNAP', 'ZIP']
- ['CURL', 'LUNGE', 'PLANK', 'PRESS']
- ['JACK', 'NADA', 'NOTHING', 'SQUAT']
- ['CRUNCH', 'KANGAROO', 'OBVIOUS', 'PLANET']

**Predictions Sequence:**
- Incorrect: ['JACK', 'NOTHING', 'OBVIOUS', 'PRESS']
- Incorrect: ['BUCKLE', 'JACK', 'NOTHING', 'OBVIOUS']
- Incorrect: ['JACK', 'NOTHING', 'OBVIOUS', 'SNAP']
- Incorrect: ['BUCKLE', 'NOTHING', 'OBVIOUS', 'SQUAT']

**Matched Groups: 0/4**

### Game 83
**Ground Truth Groups:**
- ['BROOM', 'MOP', 'RAG', 'SPONGE']
- ['ACID', 'FAT', 'HEAT', 'SALT']
- ['DUST', 'PEPPER', 'POLLEN', 'SMOKE']
- ['MAGNUM', 'MONK', 'SHAFT', 'TRACY']

**Predictions Sequence:**
- One away: ['DUST', 'MOP', 'RAG', 'SPONGE']
- Correct: ['BROOM', 'MOP', 'RAG', 'SPONGE']
- Incorrect: ['DUST', 'HEAT', 'SALT', 'SMOKE']
- One away: ['DUST', 'HEAT', 'POLLEN', 'SMOKE']
- One away: ['DUST', 'PEPPER', 'SALT', 'SMOKE']

**Matched Groups: 1/4**

### Game 84
**Ground Truth Groups:**
- ['BISHOP', 'BURNS', 'LORDE', 'POPE']
- ['DEBUT', 'INTRODUCTION', 'LAUNCH', 'PREMIERE']
- ['MOTHER', 'PRESIDENT', 'SAINT PATRICK', 'SAINT VALENTINE']
- ['BIRD', 'CLERGY MEMBER', 'M.L.B. PLAYER', 'N.F.L. PLAYER']

**Predictions Sequence:**
- Incorrect: ['BISHOP', 'CLERGY MEMBER', 'SAINT PATRICK', 'SAINT VALENTINE']
- Incorrect: ['CLERGY MEMBER', 'N.F.L. PLAYER', 'SAINT PATRICK', 'SAINT VALENTINE']
- Incorrect: ['CLERGY MEMBER', 'INTRODUCTION', 'SAINT PATRICK', 'SAINT VALENTINE']
- One away: ['CLERGY MEMBER', 'MOTHER', 'SAINT PATRICK', 'SAINT VALENTINE']

**Matched Groups: 0/4**

### Game 85
**Ground Truth Groups:**
- ['AARDVARK', 'EERIE', 'LLAMA', 'OOZE']
- ['BUDDY', 'CHAMP', 'SPORT', 'TIGER']
- ['BLUE', 'BRINY', 'DEEP', 'DRINK']
- ['AMENABLE', 'DOWN', 'GAME', 'WILLING']

**Predictions Sequence:**
- One away: ['AMENABLE', 'DOWN', 'DRINK', 'WILLING']
- One away: ['AMENABLE', 'DEEP', 'DOWN', 'WILLING']
- One away: ['AMENABLE', 'BUDDY', 'DOWN', 'WILLING']
- One away: ['AMENABLE', 'DOWN', 'OOZE', 'WILLING']

**Matched Groups: 0/4**

### Game 86
**Ground Truth Groups:**
- ['CIDER', 'PORT', 'SAKE', 'STOUT']
- ['ESSENCE', 'HEART', 'SOUL', 'SPIRIT']
- ['BRAVE', 'CARS', 'COCO', 'UP']
- ['DEMO', 'RED', 'SCOTCH', 'TICKER']

**Predictions Sequence:**
- One away: ['ESSENCE', 'SAKE', 'SOUL', 'SPIRIT']
- One away: ['BRAVE', 'ESSENCE', 'SOUL', 'SPIRIT']
- Incorrect: ['BRAVE', 'ESSENCE', 'SAKE', 'SPIRIT']
- Correct: ['ESSENCE', 'HEART', 'SOUL', 'SPIRIT']
- Incorrect: ['BRAVE', 'RED', 'SAKE', 'UP']

**Matched Groups: 1/4**

### Game 87
**Ground Truth Groups:**
- ['CROCK', 'POT', 'SKILLET', 'WOK']
- ['BUD', 'CHUM', 'MATE', 'PAL']
- ['CLOG', 'PUMP', 'SLIDE', 'WEDGE']
- ['GRASS', 'HERB', 'MARY JANE', 'WEED']

**Predictions Sequence:**
- Incorrect: ['MARY JANE', 'PAL', 'PUMP', 'SLIDE']
- Correct: ['GRASS', 'HERB', 'MARY JANE', 'WEED']
- One away: ['CLOG', 'PAL', 'PUMP', 'SLIDE']
- Correct: ['CROCK', 'POT', 'SKILLET', 'WOK']
- Incorrect: ['BUD', 'PAL', 'PUMP', 'SLIDE']
- Correct: ['CLOG', 'PUMP', 'SLIDE', 'WEDGE']
- Correct: ['BUD', 'CHUM', 'MATE', 'PAL']

**Matched Groups: 4/4**

### Game 88
**Ground Truth Groups:**
- ['COLLAR', 'HALTER', 'HARNESS', 'LEAD']
- ['BRASS', 'CHEEK', 'GALL', 'NERVE']
- ['FERRY', 'MERCURY', 'NICKS', 'PLANT']
- ['CAROUSEL', 'GLOBE', 'RECORD', 'TOP']

**Predictions Sequence:**
- Incorrect: ['BRASS', 'LEAD', 'RECORD', 'TOP']
- One away: ['GLOBE', 'LEAD', 'RECORD', 'TOP']
- Incorrect: ['HARNESS', 'LEAD', 'RECORD', 'TOP']
- Incorrect: ['LEAD', 'MERCURY', 'RECORD', 'TOP']

**Matched Groups: 0/4**

### Game 89
**Ground Truth Groups:**
- ['DOWN', 'FUR', 'SCALES', 'SHELL']
- ['CLOCK', 'MAIL', 'MAPS', 'NOTES']
- ['BOW', 'KNEEL', 'SALUTE', 'STAND']
- ['ARROW', 'DOG', 'FINGER', 'HINT']

**Predictions Sequence:**
- Incorrect: ['DOWN', 'HINT', 'NOTES', 'STAND']
- Incorrect: ['DOWN', 'FINGER', 'HINT', 'STAND']
- Incorrect: ['BOW', 'DOWN', 'HINT', 'STAND']
- Incorrect: ['DOG', 'DOWN', 'HINT', 'STAND']

**Matched Groups: 0/4**

### Game 90
**Ground Truth Groups:**
- ['FICTION', 'HUMOR', 'POETRY', 'TRAVEL']
- ['BIG', 'GIANT', 'GREAT', 'HUGE']
- ['CHEERS', 'EUPHORIA', 'FELICITY', 'GLEE']
- ['CONSTRUCTION', 'FRASIER', 'PAPER', 'WHOOPING']

**Predictions Sequence:**
- Correct: ['BIG', 'GIANT', 'GREAT', 'HUGE']
- One away: ['FICTION', 'HUMOR', 'PAPER', 'POETRY']
- One away: ['FICTION', 'PAPER', 'POETRY', 'TRAVEL']
- Correct: ['FICTION', 'HUMOR', 'POETRY', 'TRAVEL']
- One away: ['CHEERS', 'FELICITY', 'FRASIER', 'GLEE']
- One away: ['CHEERS', 'EUPHORIA', 'FRASIER', 'GLEE']

**Matched Groups: 2/4**

### Game 91
**Ground Truth Groups:**
- ['INFRARED', 'MARIGOLD', 'STINGRAY', 'ULTRAVIOLET']
- ['AIR CONDITIONER', 'CUP HOLDER', 'RADIO', 'SHIFTER']
- ['BIN', 'CARRY-ON', 'METAL DETECTOR', 'X-RAY']
- ['BLENDER', 'MICROWAVE', 'RICE COOKER', 'TOASTER']

**Predictions Sequence:**
- Incorrect: ['INFRARED', 'METAL DETECTOR', 'ULTRAVIOLET', 'X-RAY']
- Incorrect: ['INFRARED', 'MICROWAVE', 'ULTRAVIOLET', 'X-RAY']
- Incorrect: ['INFRARED', 'METAL DETECTOR', 'MICROWAVE', 'ULTRAVIOLET']
- Incorrect: ['AIR CONDITIONER', 'INFRARED', 'METAL DETECTOR', 'ULTRAVIOLET']

**Matched Groups: 0/4**

### Game 92
**Ground Truth Groups:**
- ['ACHE', 'BURN', 'SMART', 'STING']
- ['GUARD', 'MIND', 'TEND', 'WATCH']
- ['ANSWER', 'TWO', 'WRIST', 'WRONG']
- ['BRAIN', 'COURAGE', 'HEART', 'HOME']

**Predictions Sequence:**
- Incorrect: ['ANSWER', 'MIND', 'SMART', 'WRONG']
- Incorrect: ['ANSWER', 'MIND', 'WATCH', 'WRONG']
- Incorrect: ['ANSWER', 'HEART', 'MIND', 'WRONG']
- Incorrect: ['ANSWER', 'MIND', 'TEND', 'WRONG']

**Matched Groups: 0/4**

### Game 93
**Ground Truth Groups:**
- ['HAPPY HOUR', 'KARAOKE', 'LIVE MUSIC', 'TRIVIA NIGHT']
- ['AMERICAN HUSTLE', 'COMIC CON', 'GOLDEN FLEECE', 'LUCKY STIFF']
- ['CAN OF WORMS', 'HORNET’S NEST', 'MINEFIELD', 'PANDORA’S BOX']
- ['ABOUT TIME', 'FINALLY', 'GOOD RIDDANCE', 'SAYONARA']

**Predictions Sequence:**
- Incorrect: ['ABOUT TIME', 'CAN OF WORMS', 'FINALLY', 'HAPPY HOUR']
- Incorrect: ['ABOUT TIME', 'CAN OF WORMS', 'HAPPY HOUR', 'TRIVIA NIGHT']
- Incorrect: ['ABOUT TIME', 'CAN OF WORMS', 'HAPPY HOUR', 'LUCKY STIFF']
- Incorrect: ['ABOUT TIME', 'CAN OF WORMS', 'HAPPY HOUR', 'LIVE MUSIC']

**Matched Groups: 0/4**

### Game 94
**Ground Truth Groups:**
- ['BALLPARK', 'BROAD', 'GENERAL', 'ROUGH']
- ['BROW', 'DADA', 'MOMA', 'SISI']
- ['FRICK', 'FUDGE', 'SHOOT', 'SUGAR']
- ['ANGEL', 'MET', 'RAY', 'ROYAL']

**Predictions Sequence:**
- Incorrect: ['BROAD', 'MET', 'ROUGH', 'SHOOT']
- Incorrect: ['ANGEL', 'MET', 'ROUGH', 'SHOOT']
- Incorrect: ['BALLPARK', 'MET', 'ROUGH', 'SHOOT']
- Incorrect: ['MET', 'RAY', 'ROUGH', 'SHOOT']

**Matched Groups: 0/4**

### Game 95
**Ground Truth Groups:**
- ['COUPLE', 'TIE', 'UNITE', 'WED']
- ['TO', 'TOO', 'TUE', 'TWO']
- ['LAID', 'PLACED', 'PUT', 'SAT']
- ['MAY', 'SUN', 'WALL', 'WILD']

**Predictions Sequence:**
- Incorrect: ['MAY', 'PUT', 'TO', 'TOO']
- Incorrect: ['MAY', 'PLACED', 'PUT', 'TO']
- Incorrect: ['COUPLE', 'MAY', 'PUT', 'TOO']
- Incorrect: ['MAY', 'PUT', 'TO', 'TWO']

**Matched Groups: 0/4**

### Game 96
**Ground Truth Groups:**
- ['LARGE', 'LEGEND', 'PROOF', 'ROOM']
- ['CHANNEL', 'MEANS', 'MEDIUM', 'VEHICLE']
- ['GRANDE', 'MARS', 'STYLES', 'SWIFT']
- ['OUTSIDE', 'REMOTE', 'SLIM', 'SMALL']

**Predictions Sequence:**
- Incorrect: ['LARGE', 'MEANS', 'OUTSIDE', 'SMALL']
- Incorrect: ['LARGE', 'OUTSIDE', 'ROOM', 'SMALL']
- Incorrect: ['LARGE', 'MEANS', 'ROOM', 'SMALL']
- Incorrect: ['LARGE', 'MEANS', 'PROOF', 'SMALL']

**Matched Groups: 0/4**

### Game 97
**Ground Truth Groups:**
- ['DEPEND', 'HINGE', 'RELY', 'REST']
- ['CAR', 'DOOR', 'GOAT', 'HOST']
- ['BITTERS', 'ORANGE', 'RYE', 'SUGAR']
- ['CHILL', 'EASY', 'ENOUGH', 'RELAX']

**Predictions Sequence:**
- One away: ['DEPEND', 'ENOUGH', 'RELY', 'REST']
- Incorrect: ['EASY', 'ENOUGH', 'RELY', 'REST']
- One away: ['DEPEND', 'EASY', 'RELY', 'REST']
- Incorrect: ['DEPEND', 'EASY', 'ENOUGH', 'RELY']

**Matched Groups: 0/4**

### Game 98
**Ground Truth Groups:**
- ['DRESS', 'LOOK', 'MANNER', 'STYLE']
- ['SIGHT', 'SMELL', 'TASTE', 'TOUCH']
- ['BLUE', 'HARVEST', 'NEW', 'SAILOR']
- ['DITTO', 'LIKEWISE', 'SAME', 'SECOND']

**Predictions Sequence:**
- Incorrect: ['LIKEWISE', 'LOOK', 'SAME', 'TOUCH']
- Incorrect: ['LIKEWISE', 'LOOK', 'MANNER', 'SAME']
- Incorrect: ['LIKEWISE', 'LOOK', 'SAME', 'SIGHT']
- One away: ['LIKEWISE', 'LOOK', 'SAME', 'SECOND']

**Matched Groups: 0/4**

### Game 99
**Ground Truth Groups:**
- ['CHOCOLATE', 'PEACE', 'PIGEON', 'SOAP']
- ['BLACK', 'EVEN', 'ODD', 'RED']
- ['AUDITORIUM', 'GYM', 'LAB', 'LIBRARY']
- ['GOLDEN', 'GREY', 'MOTHER', 'SILLY']

**Predictions Sequence:**
- Incorrect: ['EVEN', 'MOTHER', 'ODD', 'SILLY']
- One away: ['BLACK', 'EVEN', 'GREY', 'RED']
- One away: ['BLACK', 'EVEN', 'ODD', 'SILLY']
- Incorrect: ['EVEN', 'LAB', 'ODD', 'SILLY']

**Matched Groups: 0/4**

### Game 100
**Ground Truth Groups:**
- ['FOX', 'IBEX', 'LYNX', 'ORYX']
- ['BOOK', 'BOUNCE', 'RUN', 'SPLIT']
- ['EBONY', 'JET', 'ONYX', 'RAVEN']
- ['ASH', 'BLACK', 'CYBER', 'FAT']

**Predictions Sequence:**
- One away: ['BLACK', 'EBONY', 'ONYX', 'RAVEN']
- Incorrect: ['BLACK', 'BOOK', 'ONYX', 'RAVEN']
- One away: ['BOOK', 'EBONY', 'ONYX', 'RAVEN']
- One away: ['BLACK', 'BOOK', 'RUN', 'SPLIT']

**Matched Groups: 0/4**

## Overall Metrics

- **Number of games completely solved**: 3 / 100 (3%)
- **Total number of groups completely solved**: 39 / 400 (9.75%)
- **Number of groups in which 3 words correct and 1 wrong**: 134 / 400 (33.5%)
- **Number of games with exactly 2 solved groups and 2 one-away groups**: 1 / 100 (1%)

## Why This Method Still Struggles

1. **Static Embeddings**: GloVe cannot distinguish between different senses of a word (polysemy), making it vulnerable to ambiguous Connections categories.
2. **Semantic Mismatch**: The model relies on co-occurrence data, which often misses the clever, lateral, or phonetic relationships that characterize NYT puzzles.
