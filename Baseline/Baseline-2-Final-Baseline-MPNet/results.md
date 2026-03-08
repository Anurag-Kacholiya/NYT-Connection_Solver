# MPNet Baseline Evaluation Results

## How to Run this Evaluation
To reproduce these results, navigate to the `Baseline-2-Final-Baseline-MPNet` directory and execute:

```bash
python evaluate_mpnet_custom.py
```

## Model Architecture
Detailed system architecture: [architecture.html](architecture.html)

## Methodology
This baseline utilizes a state-of-the-art **Neural-Greedy Simulation** approach:

1. **Embedding Layer**: We use `all-mpnet-base-v2` to extract 768-dimensional contextual word vectors.
2. **Feature Fusion**: A similarity matrix is computed by weighting Semantic Similarity (80%) and Lexical n-gram Similarity (20%).
3. **Double Centering Transformer**: The matrix is normalized by subtracting mean row and column similarities, emphasizing relative board-specific connections.
4. **Greedy Human-AI Simulation**: The model mimics human gameplay by picking the top-scoring 4-word clusters iteratively. It accounts for the game's life system (4 lives) and pivots its search based on ground-truth feedback.

---

Evaluated on 100 random games from the Connections dataset.

## Game-by-Game Breakdown

### Game 1
**Ground Truth Groups:**
- ['DOORKNOB', 'KNIFE', 'TOPKNOT', 'UNKNOWN']
- ['FORMULA', 'RECIPE', 'SECRET', 'TICKET']
- ['DISH', 'SPILL', 'TALK', 'TELL']
- ['BRANCH', 'FORK', 'PART', 'SPLIT']

**Predictions Sequence:**
- Incorrect: ['BRANCH', 'DOORKNOB', 'FORK', 'KNIFE']
- Correct: ['BRANCH', 'FORK', 'PART', 'SPLIT']
- One away: ['FORMULA', 'RECIPE', 'SECRET', 'UNKNOWN']
- One away: ['SECRET', 'SPILL', 'TALK', 'TELL']
- One away: ['FORMULA', 'RECIPE', 'SECRET', 'TELL']

**Matched Groups: 1/4**

### Game 2
**Ground Truth Groups:**
- ['INFINITY', 'KIDDIE', 'KIDNEY', 'OLYMPIC']
- ['BRAVO', 'DELTA', 'GOLF', 'LIMA']
- ['ELEPHANT', 'GREAT', 'NAVY', 'VACUUM']
- ['BRONCO', 'FIESTA', 'MUSTANG', 'PINTO']

**Predictions Sequence:**
- Correct: ['BRONCO', 'FIESTA', 'MUSTANG', 'PINTO']
- Incorrect: ['BRAVO', 'GREAT', 'KIDDIE', 'KIDNEY']
- Incorrect: ['BRAVO', 'DELTA', 'GREAT', 'NAVY']
- Incorrect: ['GOLF', 'INFINITY', 'NAVY', 'OLYMPIC']
- Incorrect: ['GOLF', 'INFINITY', 'OLYMPIC', 'VACUUM']

**Matched Groups: 1/4**

### Game 3
**Ground Truth Groups:**
- ['LIT', 'SCENTED', 'WAXY', 'WICKED']
- ['AMAZING', 'BEGINNER', 'GENIUS', 'SOLID']
- ['EVIL', 'LIVE', 'VEIL', 'VILE']
- ['EASY', 'FLEXIBLE', 'OPEN', 'RECEPTIVE']

**Predictions Sequence:**
- Incorrect: ['EVIL', 'GENIUS', 'VILE', 'WICKED']
- One away: ['EVIL', 'VEIL', 'VILE', 'WICKED']
- Incorrect: ['EVIL', 'VILE', 'WAXY', 'WICKED']
- One away: ['AMAZING', 'GENIUS', 'LIT', 'SOLID']

**Matched Groups: 0/4**

### Game 4
**Ground Truth Groups:**
- ['ARROW', 'BULLET', 'CHECKBOX', 'HYPHEN']
- ['ANGRY', 'BETRAY', 'CRUCIFIX', 'HYBRID']
- ['BAROQUE', 'COMPLEX', 'ELABORATE', 'INVOLVED']
- ['CRUX', 'ESSENCE', 'HEART', 'SUBSTANCE']

**Predictions Sequence:**
- One away: ['ANGRY', 'COMPLEX', 'ELABORATE', 'INVOLVED']
- Incorrect: ['ARROW', 'BULLET', 'CRUCIFIX', 'CRUX']
- One away: ['CRUCIFIX', 'CRUX', 'ESSENCE', 'SUBSTANCE']
- Incorrect: ['COMPLEX', 'ELABORATE', 'ESSENCE', 'SUBSTANCE']

**Matched Groups: 0/4**

### Game 5
**Ground Truth Groups:**
- ['BALL', 'MOVIE', 'SCHOOL', 'VITAMIN']
- ['COIL', 'SPOOL', 'WIND', 'WRAP']
- ['LASER', 'PLUCK', 'THREAD', 'WAX']
- ['HONEYCOMB', 'ORGANISM', 'SOLAR PANEL', 'SPREADSHEET']

**Predictions Sequence:**
- Incorrect: ['HONEYCOMB', 'PLUCK', 'WAX', 'WRAP']
- One away: ['BALL', 'MOVIE', 'SCHOOL', 'THREAD']
- One away: ['MOVIE', 'ORGANISM', 'SCHOOL', 'VITAMIN']
- One away: ['BALL', 'MOVIE', 'ORGANISM', 'VITAMIN']

**Matched Groups: 0/4**

### Game 6
**Ground Truth Groups:**
- ['BILLBOARD', 'MOJO', 'PITCHFORK', 'SPIN']
- ['BOILERPLATE', 'BUTTERCUP', 'JACKKNIFE', 'WITHERSPOON']
- ['BUBBLES', 'FIZZ', 'FOAM', 'FROTH']
- ['BLOSSOM', 'DEVELOP', 'MATURE', 'PROGRESS']

**Predictions Sequence:**
- Correct: ['BUBBLES', 'FIZZ', 'FOAM', 'FROTH']
- One away: ['DEVELOP', 'MATURE', 'PROGRESS', 'SPIN']
- One away: ['DEVELOP', 'MATURE', 'PITCHFORK', 'PROGRESS']
- One away: ['BOILERPLATE', 'DEVELOP', 'MATURE', 'PROGRESS']
- Correct: ['BLOSSOM', 'DEVELOP', 'MATURE', 'PROGRESS']
- One away: ['BUTTERCUP', 'JACKKNIFE', 'MOJO', 'WITHERSPOON']

**Matched Groups: 2/4**

### Game 7
**Ground Truth Groups:**
- ['CHEF', 'FISH', 'MASSAGE', 'MEATBALL']
- ['EQUALS', 'MINUS', 'PERCENT', 'TIMES']
- ['COMBO', 'GOLDFISH', 'RUFFLE', 'TAKI']
- ['LOCK', 'SHOCK', 'THATCH', 'TUFT']

**Predictions Sequence:**
- One away: ['CHEF', 'FISH', 'GOLDFISH', 'MEATBALL']
- Incorrect: ['FISH', 'GOLDFISH', 'MEATBALL', 'TAKI']
- Incorrect: ['CHEF', 'FISH', 'GOLDFISH', 'TAKI']
- Incorrect: ['FISH', 'GOLDFISH', 'MEATBALL', 'SHOCK']

**Matched Groups: 0/4**

### Game 8
**Ground Truth Groups:**
- ['EMOJI', 'GINKGO', 'KARAOKE', 'TYCOON']
- ['LIFT', 'LODGE', 'MOGUL', 'SLOPE']
- ['ANGLE', 'CYCLE', 'DENT', 'POD']
- ['CLASS', 'HOMEROOM', 'LUNCH', 'RECESS']

**Predictions Sequence:**
- Correct: ['CLASS', 'HOMEROOM', 'LUNCH', 'RECESS']
- Incorrect: ['ANGLE', 'CYCLE', 'LIFT', 'SLOPE']
- Incorrect: ['GINKGO', 'MOGUL', 'POD', 'TYCOON']
- Incorrect: ['ANGLE', 'EMOJI', 'LIFT', 'SLOPE']
- Incorrect: ['KARAOKE', 'MOGUL', 'POD', 'TYCOON']

**Matched Groups: 1/4**

### Game 9
**Ground Truth Groups:**
- ['FISH', 'FUMBLE', 'ROOT', 'RUMMAGE']
- ['CALL', 'DUB', 'LABEL', 'NAME']
- ['INLAND', 'OLIVIA', 'RAN', 'WANDA']
- ['CRUISE', 'TAKEOFF', 'TAXI', 'TOUCHDOWN']

**Predictions Sequence:**
- Incorrect: ['FUMBLE', 'RAN', 'TAKEOFF', 'TOUCHDOWN']
- Incorrect: ['FUMBLE', 'RAN', 'RUMMAGE', 'TOUCHDOWN']
- Incorrect: ['DUB', 'FUMBLE', 'TAKEOFF', 'TOUCHDOWN']
- Incorrect: ['FUMBLE', 'RUMMAGE', 'TAKEOFF', 'TOUCHDOWN']

**Matched Groups: 0/4**

### Game 10
**Ground Truth Groups:**
- ['MAC', 'PAD', 'PHONE', 'POD']
- ['JOB', 'LIMA', 'MOBILE', 'POLISH']
- ['CONE', 'CUBE', 'PYRAMID', 'SPHERE']
- ['CUCUMBER', 'EGGPLANT', 'PEPPER', 'TOMATO']

**Predictions Sequence:**
- Correct: ['CUCUMBER', 'EGGPLANT', 'PEPPER', 'TOMATO']
- Correct: ['CONE', 'CUBE', 'PYRAMID', 'SPHERE']
- Incorrect: ['JOB', 'MAC', 'MOBILE', 'PHONE']
- One away: ['MAC', 'MOBILE', 'PAD', 'PHONE']
- Incorrect: ['LIMA', 'MAC', 'MOBILE', 'PHONE']
- One away: ['MAC', 'MOBILE', 'PHONE', 'POD']

**Matched Groups: 2/4**

### Game 11
**Ground Truth Groups:**
- ['BETTER', 'ECLIPSE', 'SURPASS', 'TOP']
- ['BASKET', 'BLANKET', 'COOLER', 'UTENSILS']
- ['BOMBER', 'FEDORA', 'SATCHEL', 'WHIP']
- ['PAGAN', 'SMARTER', 'VIXEN', 'WIDEN']

**Predictions Sequence:**
- Incorrect: ['BETTER', 'BOMBER', 'COOLER', 'SMARTER']
- Incorrect: ['BETTER', 'COOLER', 'SMARTER', 'TOP']
- Incorrect: ['BASKET', 'BETTER', 'COOLER', 'SMARTER']
- Incorrect: ['BETTER', 'COOLER', 'SMARTER', 'WIDEN']

**Matched Groups: 0/4**

### Game 12
**Ground Truth Groups:**
- ['RAM', 'RESOLUTION', 'SPEED', 'STORAGE']
- ['BLINDFOLD', 'ROBE', 'SCALES', 'SWORD']
- ['CLOVE', 'FLORET', 'SPEAR', 'STALK']
- ['BELLYACHE', 'CARP', 'CRAB', 'GRUMBLE']

**Predictions Sequence:**
- Correct: ['RAM', 'RESOLUTION', 'SPEED', 'STORAGE']
- One away: ['BLINDFOLD', 'ROBE', 'SPEAR', 'SWORD']
- Incorrect: ['CLOVE', 'ROBE', 'SPEAR', 'SWORD']
- Incorrect: ['CARP', 'ROBE', 'SPEAR', 'SWORD']
- Incorrect: ['FLORET', 'ROBE', 'SPEAR', 'SWORD']

**Matched Groups: 1/4**

### Game 13
**Ground Truth Groups:**
- ['HOLD', 'PUNT', 'STALL', 'TABLE']
- ['BADGE', 'INVITE', 'PASS', 'TICKET']
- ['CENTER', 'END', 'SAFETY', 'TACKLE']
- ['CHAIR', 'DIRECT', 'LEAD', 'RUN']

**Predictions Sequence:**
- Incorrect: ['HOLD', 'PUNT', 'RUN', 'TACKLE']
- Incorrect: ['CENTER', 'HOLD', 'PUNT', 'TACKLE']
- Incorrect: ['HOLD', 'PASS', 'PUNT', 'TACKLE']
- Correct: ['BADGE', 'INVITE', 'PASS', 'TICKET']
- One away: ['HOLD', 'PUNT', 'STALL', 'TACKLE']

**Matched Groups: 1/4**

### Game 14
**Ground Truth Groups:**
- ['CATCH', 'NOTICE', 'OBSERVE', 'SEE']
- ['FIDO', 'LUCKY', 'ROVER', 'SPOT']
- ['BONE', 'REX', 'SHIRT', 'STORM']
- ['BAIT', 'CHUM', 'FLY', 'SINKER']

**Predictions Sequence:**
- One away: ['FLY', 'NOTICE', 'OBSERVE', 'SEE']
- One away: ['NOTICE', 'OBSERVE', 'SEE', 'SPOT']
- Correct: ['CATCH', 'NOTICE', 'OBSERVE', 'SEE']
- Incorrect: ['BONE', 'FIDO', 'REX', 'ROVER']
- Incorrect: ['BONE', 'CHUM', 'FIDO', 'REX']

**Matched Groups: 1/4**

### Game 15
**Ground Truth Groups:**
- ['MINI', 'PENCIL', 'POODLE', 'WRAP']
- ['CILANTRO', 'LIME', 'ONION', 'SALSA']
- ['CAPITOL', 'FAITH', 'FOOT', 'MOLE']
- ['KILO', 'MEGA', 'MICRO', 'PICO']

**Predictions Sequence:**
- One away: ['KILO', 'MEGA', 'MICRO', 'MINI']
- Incorrect: ['CAPITOL', 'MEGA', 'MICRO', 'MINI']
- Incorrect: ['MEGA', 'MICRO', 'MINI', 'MOLE']
- One away: ['MEGA', 'MICRO', 'MINI', 'PICO']

**Matched Groups: 0/4**

### Game 16
**Ground Truth Groups:**
- ['AVIATOR', 'CAT EYE', 'WAYFARER', 'WRAPAROUND']
- ['BASSOON', 'CLARINET', 'FLUTE', 'OBOE']
- ['HARBOR', 'HARP', 'HOODED', 'MONK']
- ['CAP', 'COVER', 'LID', 'TOP']

**Predictions Sequence:**
- Correct: ['BASSOON', 'CLARINET', 'FLUTE', 'OBOE']
- Incorrect: ['COVER', 'HOODED', 'LID', 'WRAPAROUND']
- Correct: ['CAP', 'COVER', 'LID', 'TOP']
- One away: ['AVIATOR', 'HOODED', 'WAYFARER', 'WRAPAROUND']
- One away: ['AVIATOR', 'CAT EYE', 'HOODED', 'WAYFARER']
- Correct: ['AVIATOR', 'CAT EYE', 'WAYFARER', 'WRAPAROUND']
- Correct: ['HARBOR', 'HARP', 'HOODED', 'MONK']

**Matched Groups: 4/4**

### Game 17
**Ground Truth Groups:**
- ['DOG', 'DRIFT', 'HOLLY', 'SANDAL']
- ['BREAK', 'HOLIDAY', 'LEAVE', 'RECESS']
- ['BUCKLE', 'HOLE', 'LOOP', 'STRAP']
- ['HOLEY', 'HOLI', 'HOLY', 'WHOLLY']

**Predictions Sequence:**
- Incorrect: ['HOLI', 'HOLIDAY', 'HOLLY', 'HOLY']
- Incorrect: ['HOLE', 'HOLEY', 'HOLI', 'HOLIDAY']
- Incorrect: ['HOLI', 'HOLIDAY', 'HOLLY', 'WHOLLY']
- Incorrect: ['BREAK', 'HOLI', 'HOLIDAY', 'HOLLY']

**Matched Groups: 0/4**

### Game 18
**Ground Truth Groups:**
- ['ALARM', 'DATE', 'LIGHT', 'STOPWATCH']
- ['ACCORD', 'AWARD', 'GRANT', 'VEST']
- ['FLEECE', 'HOSE', 'HUSTLE', 'SHAFT']
- ['CAPOTE', 'LINCOLN', 'MILK', 'RAY']

**Predictions Sequence:**
- One away: ['CAPOTE', 'GRANT', 'LINCOLN', 'RAY']
- One away: ['ACCORD', 'ALARM', 'DATE', 'STOPWATCH']
- One away: ['ALARM', 'CAPOTE', 'DATE', 'STOPWATCH']
- Correct: ['ALARM', 'DATE', 'LIGHT', 'STOPWATCH']
- One away: ['FLEECE', 'HOSE', 'SHAFT', 'VEST']

**Matched Groups: 1/4**

### Game 19
**Ground Truth Groups:**
- ['BUDDY', 'CHAMP', 'SPORT', 'TIGER']
- ['BLUE', 'BRINY', 'DEEP', 'DRINK']
- ['AARDVARK', 'EERIE', 'LLAMA', 'OOZE']
- ['AMENABLE', 'DOWN', 'GAME', 'WILLING']

**Predictions Sequence:**
- One away: ['CHAMP', 'GAME', 'SPORT', 'TIGER']
- Incorrect: ['CHAMP', 'DRINK', 'GAME', 'SPORT']
- Correct: ['BUDDY', 'CHAMP', 'SPORT', 'TIGER']
- One away: ['AMENABLE', 'BLUE', 'DOWN', 'WILLING']
- Incorrect: ['AMENABLE', 'EERIE', 'LLAMA', 'WILLING']

**Matched Groups: 1/4**

### Game 20
**Ground Truth Groups:**
- ['DOUBT', 'MOVIE', 'SHADOW', 'VOTE']
- ['GEAR', 'HANDLEBAR', 'PEDAL', 'WHEEL']
- ['DOG', 'FOLLOW', 'TAIL', 'TRACK']
- ['BEARD', 'GOATEE', 'MUSTACHE', 'STUBBLE']

**Predictions Sequence:**
- Correct: ['BEARD', 'GOATEE', 'MUSTACHE', 'STUBBLE']
- One away: ['GEAR', 'PEDAL', 'TRACK', 'WHEEL']
- Correct: ['GEAR', 'HANDLEBAR', 'PEDAL', 'WHEEL']
- One away: ['DOG', 'DOUBT', 'MOVIE', 'SHADOW']
- Correct: ['DOG', 'FOLLOW', 'TAIL', 'TRACK']
- Correct: ['DOUBT', 'MOVIE', 'SHADOW', 'VOTE']

**Matched Groups: 4/4**

### Game 21
**Ground Truth Groups:**
- ['MULHOLLAND', 'RODEO', 'SUNSET', 'VINE']
- ['COWBOY', 'MAVERICK', 'STAR', 'WING']
- ['EGG', 'PUSH', 'SPUR', 'URGE']
- ['MINK', 'PARS', 'ROE', 'SOFA']

**Predictions Sequence:**
- Incorrect: ['COWBOY', 'MAVERICK', 'RODEO', 'SUNSET']
- Incorrect: ['COWBOY', 'MAVERICK', 'RODEO', 'SPUR']
- Incorrect: ['COWBOY', 'MAVERICK', 'MINK', 'RODEO']
- One away: ['COWBOY', 'MAVERICK', 'RODEO', 'STAR']

**Matched Groups: 0/4**

### Game 22
**Ground Truth Groups:**
- ['HUM', 'SCAT', 'SING', 'WHISTLE']
- ['LIME', 'MINT', 'RUM', 'SODA']
- ['COUNT', 'GROSS', 'SUM', 'TOTAL']
- ['GLUE', 'GUM', 'STICK', 'TAPE']

**Predictions Sequence:**
- Correct: ['COUNT', 'GROSS', 'SUM', 'TOTAL']
- One away: ['GUM', 'LIME', 'RUM', 'SODA']
- One away: ['GUM', 'MINT', 'RUM', 'SODA']
- One away: ['GUM', 'LIME', 'MINT', 'SODA']
- Correct: ['LIME', 'MINT', 'RUM', 'SODA']
- Correct: ['GLUE', 'GUM', 'STICK', 'TAPE']
- Correct: ['HUM', 'SCAT', 'SING', 'WHISTLE']

**Matched Groups: 4/4**

### Game 23
**Ground Truth Groups:**
- ['ALE', 'EEK', 'HIVE', 'QUASH']
- ['PINCH', 'SCROLL', 'SWIPE', 'TAP']
- ['CHILL', 'LOAF', 'LOUNGE', 'VEG']
- ['BAR', 'BOOTH', 'COUNTER', 'TABLE']

**Predictions Sequence:**
- Correct: ['PINCH', 'SCROLL', 'SWIPE', 'TAP']
- One away: ['BAR', 'BOOTH', 'LOUNGE', 'TABLE']
- Incorrect: ['ALE', 'BAR', 'LOUNGE', 'TABLE']
- Incorrect: ['ALE', 'BAR', 'LOAF', 'LOUNGE']
- One away: ['BAR', 'COUNTER', 'LOUNGE', 'TABLE']

**Matched Groups: 1/4**

### Game 24
**Ground Truth Groups:**
- ['CHIME', 'CUCKOO', 'TICK', 'TOCK']
- ['DENT', 'DING', 'NICK', 'SCRATCH']
- ['BOARDWALK', 'DOCK', 'LIGHTHOUSE', 'WHARF']
- ['COCK', 'MOCK', 'PIG', 'PONY']

**Predictions Sequence:**
- Correct: ['BOARDWALK', 'DOCK', 'LIGHTHOUSE', 'WHARF']
- One away: ['CHIME', 'DING', 'TICK', 'TOCK']
- Incorrect: ['DING', 'SCRATCH', 'TICK', 'TOCK']
- Incorrect: ['COCK', 'MOCK', 'TICK', 'TOCK']
- One away: ['COCK', 'CUCKOO', 'PIG', 'PONY']

**Matched Groups: 1/4**

### Game 25
**Ground Truth Groups:**
- ['CRUSH', 'MUG', 'SPRITE', 'SQUIRT']
- ['ARIZONA', 'COLORADO', 'NEVADA', 'UTAH']
- ['GENESIS', 'KANSAS', 'RUSH', 'YES']
- ['HAWK', 'MONTANA', 'SOPRANO', 'STARK']

**Predictions Sequence:**
- Incorrect: ['ARIZONA', 'COLORADO', 'KANSAS', 'MONTANA']
- One away: ['ARIZONA', 'COLORADO', 'MONTANA', 'NEVADA']
- Incorrect: ['COLORADO', 'KANSAS', 'MONTANA', 'NEVADA']
- Incorrect: ['COLORADO', 'KANSAS', 'MONTANA', 'UTAH']

**Matched Groups: 0/4**

### Game 26
**Ground Truth Groups:**
- ['CUP', 'GALLON', 'PINT', 'QUART']
- ['RAND', 'REAL', 'STERLING', 'WON']
- ['CARROT', 'COFFEE', 'POUND', 'SPONGE']
- ['AWFUL', 'QUITE', 'SUPER', 'VERY']

**Predictions Sequence:**
- One away: ['GALLON', 'PINT', 'POUND', 'QUART']
- Correct: ['CUP', 'GALLON', 'PINT', 'QUART']
- One away: ['AWFUL', 'QUITE', 'REAL', 'VERY']
- Correct: ['AWFUL', 'QUITE', 'SUPER', 'VERY']
- One away: ['CARROT', 'COFFEE', 'RAND', 'SPONGE']
- Incorrect: ['COFFEE', 'RAND', 'REAL', 'SPONGE']

**Matched Groups: 2/4**

### Game 27
**Ground Truth Groups:**
- ['HYPER', 'SUPER', 'UBER', 'ULTRA']
- ['BALLOON', 'MUSHROOM', 'SNOWBALL', 'SWELL']
- ['DOMINO', 'JACK', 'MARBLE', 'STICK']
- ['CLEAN', 'NEAT', 'TIDY', 'TRIM']

**Predictions Sequence:**
- Correct: ['CLEAN', 'NEAT', 'TIDY', 'TRIM']
- Correct: ['HYPER', 'SUPER', 'UBER', 'ULTRA']
- One away: ['BALLOON', 'MARBLE', 'MUSHROOM', 'SNOWBALL']
- Correct: ['BALLOON', 'MUSHROOM', 'SNOWBALL', 'SWELL']
- Correct: ['DOMINO', 'JACK', 'MARBLE', 'STICK']

**Matched Groups: 4/4**

### Game 28
**Ground Truth Groups:**
- ['ROGER', 'TRIX', 'VELVETEEN', 'WHITE']
- ['PEN', 'PRINTER', 'SQUID', 'TATTOO MACHINE']
- ['DONUT', 'JUROR', 'MONTH', 'ROSE']
- ['FOLLOW', 'MONITOR', 'TRACK', 'WATCH']

**Predictions Sequence:**
- Correct: ['FOLLOW', 'MONITOR', 'TRACK', 'WATCH']
- Correct: ['PEN', 'PRINTER', 'SQUID', 'TATTOO MACHINE']
- Incorrect: ['JUROR', 'ROGER', 'ROSE', 'VELVETEEN']
- One away: ['ROGER', 'ROSE', 'TRIX', 'VELVETEEN']
- Incorrect: ['DONUT', 'ROSE', 'VELVETEEN', 'WHITE']
- Incorrect: ['DONUT', 'ROGER', 'ROSE', 'WHITE']

**Matched Groups: 2/4**

### Game 29
**Ground Truth Groups:**
- ['MYRRH', 'NYMPH', 'RHYTHM', 'SPHYNX']
- ['ESTEEM', 'PRIZE', 'TREASURE', 'VALUE']
- ['FRANKINCENSE', 'JACKPOT', 'MARKDOWN', 'NICKNAME']
- ['JOURNEY', 'ODYSSEY', 'QUEST', 'VOYAGE']

**Predictions Sequence:**
- Correct: ['JOURNEY', 'ODYSSEY', 'QUEST', 'VOYAGE']
- One away: ['JACKPOT', 'PRIZE', 'TREASURE', 'VALUE']
- One away: ['MYRRH', 'NICKNAME', 'NYMPH', 'SPHYNX']
- One away: ['MARKDOWN', 'MYRRH', 'NYMPH', 'SPHYNX']
- Correct: ['MYRRH', 'NYMPH', 'RHYTHM', 'SPHYNX']
- Correct: ['ESTEEM', 'PRIZE', 'TREASURE', 'VALUE']
- Correct: ['FRANKINCENSE', 'JACKPOT', 'MARKDOWN', 'NICKNAME']

**Matched Groups: 4/4**

### Game 30
**Ground Truth Groups:**
- ['BEAT', 'POUND', 'PULSE', 'THUMP']
- ['BALL', 'HILLS', 'LOCKER', 'PRINT']
- ['HAMMER', 'HANGER', 'LEVEL', 'NAIL']
- ['CAPE', 'CRATER', 'GORGE', 'RIDGE']

**Predictions Sequence:**
- One away: ['CRATER', 'GORGE', 'HILLS', 'RIDGE']
- One away: ['CAPE', 'GORGE', 'HILLS', 'RIDGE']
- Incorrect: ['BEAT', 'HAMMER', 'NAIL', 'THUMP']
- One away: ['BEAT', 'HAMMER', 'PULSE', 'THUMP']

**Matched Groups: 0/4**

### Game 31
**Ground Truth Groups:**
- ['BODY', 'LOVE', 'ROMANCE', 'SIGN']
- ['BEAKER', 'DROPPER', 'GOGGLES', 'MICROSCOPE']
- ['CELL', 'FORMULA', 'SHEET', 'SORT']
- ['CENTER', 'HEART', 'HUB', 'NUCLEUS']

**Predictions Sequence:**
- One away: ['BODY', 'HEART', 'LOVE', 'ROMANCE']
- One away: ['HEART', 'LOVE', 'ROMANCE', 'SIGN']
- Incorrect: ['BEAKER', 'CELL', 'MICROSCOPE', 'NUCLEUS']
- Incorrect: ['HEART', 'HUB', 'LOVE', 'ROMANCE']

**Matched Groups: 0/4**

### Game 32
**Ground Truth Groups:**
- ['FAN', 'HISTORICAL', 'PULP', 'SCIENCE']
- ['BIOPIC', 'CALCIUM', 'CHEMISE', 'TRIGLYCERIDE']
- ['JUICE', 'PULL', 'SWAY', 'WEIGHT']
- ['CLUSTER', 'COLLECT', 'CONCENTRATE', 'GROUP']

**Predictions Sequence:**
- Incorrect: ['CALCIUM', 'CHEMISE', 'JUICE', 'PULP']
- Incorrect: ['CALCIUM', 'JUICE', 'PULP', 'TRIGLYCERIDE']
- Correct: ['CLUSTER', 'COLLECT', 'CONCENTRATE', 'GROUP']
- One away: ['CALCIUM', 'CHEMISE', 'PULP', 'TRIGLYCERIDE']
- Incorrect: ['CALCIUM', 'JUICE', 'PULP', 'SCIENCE']

**Matched Groups: 1/4**

### Game 33
**Ground Truth Groups:**
- ['SANDWICH', 'SHOEHORN', 'SQUEEZE', 'WEDGE']
- ['CATERPILLAR', 'DOVE', 'GREYHOUND', 'PUMA']
- ['INDY', 'METTLE', 'SEOUL', 'WRAP']
- ['BELT', 'BLOW', 'SLUG', 'SOCK']

**Predictions Sequence:**
- One away: ['CATERPILLAR', 'DOVE', 'GREYHOUND', 'SLUG']
- One away: ['SHOEHORN', 'SOCK', 'SQUEEZE', 'WEDGE']
- Correct: ['CATERPILLAR', 'DOVE', 'GREYHOUND', 'PUMA']
- Incorrect: ['BELT', 'BLOW', 'SQUEEZE', 'WRAP']
- Incorrect: ['BELT', 'BLOW', 'METTLE', 'SQUEEZE']

**Matched Groups: 1/4**

### Game 34
**Ground Truth Groups:**
- ['DODGE', 'HIDE', 'HOP', 'TAG']
- ['BRAND', 'COLLECTION', 'LABEL', 'LINE']
- ['BUNNY', 'DUCK', 'MARTIAN', 'PIG']
- ['GREECE', 'HARE', 'KATZ', 'MAIM']

**Predictions Sequence:**
- One away: ['BUNNY', 'DUCK', 'HARE', 'PIG']
- One away: ['BRAND', 'COLLECTION', 'LABEL', 'TAG']
- Incorrect: ['BUNNY', 'DUCK', 'HARE', 'HOP']
- Incorrect: ['BUNNY', 'HARE', 'HOP', 'PIG']

**Matched Groups: 0/4**

### Game 35
**Ground Truth Groups:**
- ['BAGUETTE', 'EMERALD', 'PRINCESS', 'RADIANT']
- ['CANDY', 'COSTUME', 'DECORATIONS', 'PUMPKIN']
- ['BEAN', 'MELON', 'NOODLE', 'NUT']
- ['ANISE', 'FENNEL', 'LICORICE', 'TARRAGON']

**Predictions Sequence:**
- Incorrect: ['COSTUME', 'DECORATIONS', 'EMERALD', 'RADIANT']
- Incorrect: ['COSTUME', 'DECORATIONS', 'PRINCESS', 'RADIANT']
- One away: ['FENNEL', 'LICORICE', 'MELON', 'TARRAGON']
- One away: ['CANDY', 'COSTUME', 'DECORATIONS', 'PRINCESS']

**Matched Groups: 0/4**

### Game 36
**Ground Truth Groups:**
- ['CAR', 'DEAD', 'LIVER', 'WHIRL']
- ['DRAW', 'GRAB', 'HOOK', 'PULL']
- ['KIDNEY', 'MUNG', 'NAVY', 'PINTO']
- ['DRIVE', 'LOW', 'NEUTRAL', 'REVERSE']

**Predictions Sequence:**
- Correct: ['DRAW', 'GRAB', 'HOOK', 'PULL']
- Incorrect: ['CAR', 'KIDNEY', 'LIVER', 'PINTO']
- Incorrect: ['CAR', 'KIDNEY', 'LIVER', 'NAVY']
- Incorrect: ['CAR', 'DRIVE', 'KIDNEY', 'LIVER']
- One away: ['KIDNEY', 'LIVER', 'NAVY', 'PINTO']

**Matched Groups: 1/4**

### Game 37
**Ground Truth Groups:**
- ['EAR', 'KETTLE', 'OIL', 'STEEL']
- ['EXTENT', 'RANGE', 'REACH', 'SCOPE']
- ['BRUSH', 'COMB', 'DRYER', 'IRON']
- ['ARMS', 'COAT', 'CREST', 'SHIELD']

**Predictions Sequence:**
- Correct: ['EXTENT', 'RANGE', 'REACH', 'SCOPE']
- One away: ['BRUSH', 'COMB', 'CREST', 'DRYER']
- Incorrect: ['BRUSH', 'COAT', 'COMB', 'CREST']
- One away: ['IRON', 'KETTLE', 'OIL', 'STEEL']
- Incorrect: ['DRYER', 'IRON', 'OIL', 'STEEL']

**Matched Groups: 1/4**

### Game 38
**Ground Truth Groups:**
- ['BALL', 'NET', 'PADDLE', 'TABLE']
- ['BUTT', 'FORE', 'OAR', 'SEW']
- ['CLATTER', 'RACKET', 'ROW', 'RUCKUS']
- ['ARTICLE', 'COLUMN', 'FEATURE', 'STORY']

**Predictions Sequence:**
- One away: ['CLATTER', 'OAR', 'RACKET', 'RUCKUS']
- Incorrect: ['COLUMN', 'PADDLE', 'ROW', 'TABLE']
- Incorrect: ['ARTICLE', 'COLUMN', 'ROW', 'TABLE']
- Incorrect: ['BUTT', 'COLUMN', 'ROW', 'TABLE']

**Matched Groups: 0/4**

### Game 39
**Ground Truth Groups:**
- ['DASH', 'HYPHEN', 'MINUS', 'UNDERSCORE']
- ['FORMER', 'LATE', 'PAST', 'PRIOR']
- ['BAGGIE', 'BARGE', 'COMER', 'PISA']
- ['CHARGE', 'COST', 'PRICE', 'RATE']

**Predictions Sequence:**
- Correct: ['CHARGE', 'COST', 'PRICE', 'RATE']
- Correct: ['FORMER', 'LATE', 'PAST', 'PRIOR']
- Correct: ['DASH', 'HYPHEN', 'MINUS', 'UNDERSCORE']
- Correct: ['BAGGIE', 'BARGE', 'COMER', 'PISA']

**Matched Groups: 4/4**

### Game 40
**Ground Truth Groups:**
- ['BOUNTY', 'CRUNCH', 'HEATH', 'MILKY WAY']
- ['DREAMCAST', 'GENESIS', 'SWITCH', 'WII']
- ['ANVIL', 'COCHLEA', 'HAMMER', 'STIRRUP']
- ['BLINK', 'MAROON', 'SUM', 'U']

**Predictions Sequence:**
- One away: ['DREAMCAST', 'SWITCH', 'U', 'WII']
- One away: ['BLINK', 'DREAMCAST', 'SWITCH', 'WII']
- One away: ['DREAMCAST', 'MILKY WAY', 'SWITCH', 'WII']
- One away: ['DREAMCAST', 'MAROON', 'SWITCH', 'WII']

**Matched Groups: 0/4**

### Game 41
**Ground Truth Groups:**
- ['BUDGET', 'ENTERPRISE', 'NATIONAL', 'THRIFTY']
- ['RUMBLING', 'SPECULATION', 'TALK', 'WHISPERS']
- ['CREEP', 'SLIP', 'STEAL', 'TIPTOE']
- ['ENVIRON', 'JARGON', 'LATIN', 'PLEAD']

**Predictions Sequence:**
- One away: ['BUDGET', 'ENTERPRISE', 'ENVIRON', 'THRIFTY']
- One away: ['CREEP', 'RUMBLING', 'SLIP', 'TIPTOE']
- Correct: ['RUMBLING', 'SPECULATION', 'TALK', 'WHISPERS']
- Correct: ['BUDGET', 'ENTERPRISE', 'NATIONAL', 'THRIFTY']
- One away: ['CREEP', 'PLEAD', 'SLIP', 'TIPTOE']
- Correct: ['CREEP', 'SLIP', 'STEAL', 'TIPTOE']
- Correct: ['ENVIRON', 'JARGON', 'LATIN', 'PLEAD']

**Matched Groups: 4/4**

### Game 42
**Ground Truth Groups:**
- ['DIRECT', 'IRREGULAR', 'POSSESSIVE', 'PRESENT']
- ['MIX', 'SAMPLE', 'SCRATCH', 'SPIN']
- ['ECOLOGICAL', 'JEALOUS', 'LUSH', 'NAUSEOUS']
- ['STRAIN', 'TAX', 'TEST', 'TRY']

**Predictions Sequence:**
- Incorrect: ['SAMPLE', 'SPIN', 'TEST', 'TRY']
- Incorrect: ['ECOLOGICAL', 'IRREGULAR', 'LUSH', 'STRAIN']
- Incorrect: ['MIX', 'SAMPLE', 'TEST', 'TRY']
- One away: ['IRREGULAR', 'JEALOUS', 'LUSH', 'NAUSEOUS']

**Matched Groups: 0/4**

### Game 43
**Ground Truth Groups:**
- ['BOOB', 'EGGSHELL', 'GIGGLE', 'HELLO']
- ['CANT', 'LEAN', 'LIST', 'SLOPE']
- ['CON', 'DUPE', 'FOOL', 'TRICK']
- ['DOPE', 'SCOOP', 'SKINNY', 'WORD']

**Predictions Sequence:**
- Incorrect: ['DOPE', 'LEAN', 'SKINNY', 'SLOPE']
- Incorrect: ['LEAN', 'SCOOP', 'SKINNY', 'SLOPE']
- Incorrect: ['DOPE', 'DUPE', 'LEAN', 'SLOPE']
- Incorrect: ['GIGGLE', 'LEAN', 'SKINNY', 'SLOPE']

**Matched Groups: 0/4**

### Game 44
**Ground Truth Groups:**
- ['CANARY', 'LEMON', 'MINION', 'MUSTARD']
- ['ADDITION', 'ANNEX', 'EXTENSION', 'WING']
- ['COLONEL', 'PHARAOH', 'WEDNESDAY', 'WORCESTERSHIRE']
- ['CALCULUS', 'CAVITY', 'PLAQUE', 'TARTAR']

**Predictions Sequence:**
- One away: ['ADDITION', 'ANNEX', 'CALCULUS', 'EXTENSION']
- Incorrect: ['LEMON', 'MUSTARD', 'TARTAR', 'WORCESTERSHIRE']
- Incorrect: ['ADDITION', 'CALCULUS', 'CAVITY', 'EXTENSION']
- One away: ['CANARY', 'LEMON', 'MUSTARD', 'TARTAR']

**Matched Groups: 0/4**

### Game 45
**Ground Truth Groups:**
- ['ALTERNATIVE', 'EMOTIONAL', 'POPULAR', 'PROGRESSIVE']
- ['CHEESE', 'CUP', 'PAN', 'SHORT']
- ['BLAB', 'DISH', 'SPILL', 'TELL']
- ['COMMAND', 'CONTROL', 'OPTION', 'RETURN']

**Predictions Sequence:**
- One away: ['ALTERNATIVE', 'OPTION', 'POPULAR', 'PROGRESSIVE']
- One away: ['ALTERNATIVE', 'COMMAND', 'CONTROL', 'OPTION']
- One away: ['ALTERNATIVE', 'COMMAND', 'OPTION', 'RETURN']
- Incorrect: ['ALTERNATIVE', 'CHEESE', 'OPTION', 'PROGRESSIVE']

**Matched Groups: 0/4**

### Game 46
**Ground Truth Groups:**
- ['CELSIUS', 'COLD', 'ONE HUNDRED', 'SEE']
- ['BUMPER', 'GIANT', 'MONSTER', 'TITANIC']
- ['FINAL', 'QUALIFIER', 'QUARTER', 'SEMI']
- ['DIVE', 'DUMP', 'HOLE', 'JOINT']

**Predictions Sequence:**
- One away: ['CELSIUS', 'COLD', 'ONE HUNDRED', 'QUARTER']
- One away: ['DIVE', 'DUMP', 'HOLE', 'TITANIC']
- One away: ['DIVE', 'DUMP', 'HOLE', 'SEE']
- Correct: ['DIVE', 'DUMP', 'HOLE', 'JOINT']
- Incorrect: ['CELSIUS', 'ONE HUNDRED', 'QUALIFIER', 'QUARTER']

**Matched Groups: 1/4**

### Game 47
**Ground Truth Groups:**
- ['DRE', 'EVIL', 'PEPPER', 'SEUSS']
- ['CERAMICS', 'CHINA', 'POTTERY', 'WARE']
- ['CRYPTO', 'DECAF', 'DISNEY', 'PRUSSIAN']
- ['CRUMB', 'GRAIN', 'MORSEL', 'SHRED']

**Predictions Sequence:**
- One away: ['CERAMICS', 'GRAIN', 'POTTERY', 'WARE']
- Incorrect: ['CERAMICS', 'GRAIN', 'PEPPER', 'POTTERY']
- One away: ['CERAMICS', 'CHINA', 'GRAIN', 'POTTERY']
- Incorrect: ['CERAMICS', 'CRYPTO', 'GRAIN', 'POTTERY']

**Matched Groups: 0/4**

### Game 48
**Ground Truth Groups:**
- ['ANDROID', 'RATTLESNAKE', 'ROLLERBLADE', 'SHAKESPEARE']
- ['DANCE', 'MONOLOGUE', 'SKETCH', 'SONG']
- ['CERTIFICATE', 'DEED', 'RECEIPT', 'TITLE']
- ['DONUT', 'PACIFIER', 'SAXOPHONE', 'SKATEBOARD']

**Predictions Sequence:**
- Incorrect: ['DANCE', 'ROLLERBLADE', 'SAXOPHONE', 'SKATEBOARD']
- Incorrect: ['DANCE', 'ROLLERBLADE', 'SKATEBOARD', 'SKETCH']
- Incorrect: ['ANDROID', 'DANCE', 'ROLLERBLADE', 'SKATEBOARD']
- Correct: ['CERTIFICATE', 'DEED', 'RECEIPT', 'TITLE']
- Incorrect: ['DANCE', 'ROLLERBLADE', 'SKATEBOARD', 'SONG']

**Matched Groups: 1/4**

### Game 49
**Ground Truth Groups:**
- ['MICHAEL JORDAN', 'RODEO', 'TAURUS', 'WALL STREET']
- ['HOLLYWOOD', 'HOURGLASS', 'KUBRICK', 'NEUROPLASTIC']
- ['DISSOLVE', 'END', 'SCRAP', 'SUNSET']
- ['HERB', 'SHRUB', 'TREE', 'VINE']

**Predictions Sequence:**
- Correct: ['HERB', 'SHRUB', 'TREE', 'VINE']
- Incorrect: ['HOLLYWOOD', 'KUBRICK', 'MICHAEL JORDAN', 'WALL STREET']
- One away: ['HOLLYWOOD', 'KUBRICK', 'NEUROPLASTIC', 'WALL STREET']
- One away: ['HOLLYWOOD', 'MICHAEL JORDAN', 'RODEO', 'WALL STREET']
- Incorrect: ['HOLLYWOOD', 'RODEO', 'SUNSET', 'WALL STREET']

**Matched Groups: 1/4**

### Game 50
**Ground Truth Groups:**
- ['BIRD', 'COUNT', 'GROUCH', 'MONSTER']
- ['HALO', 'HARP', 'ROBE', 'WINGS']
- ['CHIC', 'CURRENT', 'FRESH', 'HAPPENING']
- ['BLOOD', 'BLUES', 'CHEMICAL', 'MARX']

**Predictions Sequence:**
- Incorrect: ['BIRD', 'BLUES', 'HARP', 'WINGS']
- One away: ['BIRD', 'HALO', 'HARP', 'WINGS']
- Incorrect: ['BIRD', 'BLOOD', 'BLUES', 'WINGS']
- Incorrect: ['BIRD', 'BLUES', 'MONSTER', 'WINGS']

**Matched Groups: 0/4**

### Game 51
**Ground Truth Groups:**
- ['BEARD', 'HAT', 'ROBE', 'STAFF']
- ['CLEAR', 'EARN', 'MAKE', 'NET']
- ['BRIDGE', 'DECK', 'HATCH', 'KEEL']
- ['CLEF', 'SECTION', 'SPAN', 'SUITE']

**Predictions Sequence:**
- One away: ['BEARD', 'HAT', 'HATCH', 'ROBE']
- One away: ['BEARD', 'EARN', 'HAT', 'ROBE']
- One away: ['BEARD', 'DECK', 'HAT', 'ROBE']
- One away: ['BEARD', 'CLEAR', 'HAT', 'ROBE']

**Matched Groups: 0/4**

### Game 52
**Ground Truth Groups:**
- ['A FEW', 'CERTAIN', 'SOME', 'VARIOUS']
- ['A DEAL', 'A ROSE', 'ENOUGH', 'LOVE']
- ['A CAPPELLA', 'BARBERSHOP', 'DOO-WOP', 'MADRIGAL']
- ['A LIFE', 'A NOVEL', 'ESSAYS', 'PART ONE']

**Predictions Sequence:**
- Correct: ['A FEW', 'CERTAIN', 'SOME', 'VARIOUS']
- Incorrect: ['A LIFE', 'A NOVEL', 'A ROSE', 'LOVE']
- One away: ['A LIFE', 'A NOVEL', 'A ROSE', 'ESSAYS']
- Incorrect: ['A DEAL', 'A LIFE', 'A NOVEL', 'A ROSE']
- Incorrect: ['A CAPPELLA', 'A LIFE', 'A NOVEL', 'A ROSE']

**Matched Groups: 1/4**

### Game 53
**Ground Truth Groups:**
- ['CHOCOLATE', 'GUM', 'LICORICE', 'LOLLIPOP']
- ['ACORN', 'CONE', 'POLLEN', 'SAP']
- ['CHUMP', 'FOOL', 'MARK', 'SUCKER']
- ['FACE', 'HURRICANE', 'NEEDLE', 'POTATO']

**Predictions Sequence:**
- One away: ['CHOCOLATE', 'GUM', 'LICORICE', 'POLLEN']
- One away: ['CHUMP', 'FOOL', 'SAP', 'SUCKER']
- One away: ['CHUMP', 'FACE', 'FOOL', 'SUCKER']
- Correct: ['CHUMP', 'FOOL', 'MARK', 'SUCKER']
- Correct: ['CHOCOLATE', 'GUM', 'LICORICE', 'LOLLIPOP']
- One away: ['CONE', 'HURRICANE', 'NEEDLE', 'POTATO']

**Matched Groups: 2/4**

### Game 54
**Ground Truth Groups:**
- ['CHIP', 'ROCKY', 'SCRATCHY', 'STITCH']
- ['BED', 'PARCEL', 'PATCH', 'PLOT']
- ['BUMPY', 'ROUGH', 'RUGGED', 'UNEVEN']
- ['AMUSEMENT', 'NATIONAL', 'PARALLEL', 'SOUTH']

**Predictions Sequence:**
- Incorrect: ['BUMPY', 'ROCKY', 'RUGGED', 'SCRATCHY']
- One away: ['BUMPY', 'ROCKY', 'ROUGH', 'RUGGED']
- One away: ['BUMPY', 'ROUGH', 'RUGGED', 'SCRATCHY']
- One away: ['BUMPY', 'RUGGED', 'SCRATCHY', 'UNEVEN']

**Matched Groups: 0/4**

### Game 55
**Ground Truth Groups:**
- ['CHEAT', 'CON', 'FAKE', 'QUACK']
- ['BOLT', 'DOWN', 'GOBBLE', 'SCARF']
- ['CREDIT', 'PRAISE', 'RECOGNITION', 'THANKS']
- ['FAT', 'GIVING', 'SUPER', 'TACO']

**Predictions Sequence:**
- One away: ['CREDIT', 'GIVING', 'PRAISE', 'RECOGNITION']
- One away: ['CREDIT', 'GIVING', 'PRAISE', 'THANKS']
- Incorrect: ['BOLT', 'GOBBLE', 'QUACK', 'TACO']
- One away: ['CHEAT', 'CON', 'FAKE', 'FAT']

**Matched Groups: 0/4**

### Game 56
**Ground Truth Groups:**
- ['DIRECT', 'GUIDE', 'LEAD', 'STEER']
- ['DEATH', 'HIDDEN', 'SILICON', 'UNCANNY']
- ['DRYER', 'HAMPER', 'IRON', 'WASHER']
- ['IODINE', 'IOTA', 'MYSELF', 'ONE']

**Predictions Sequence:**
- Incorrect: ['IODINE', 'IOTA', 'IRON', 'SILICON']
- Incorrect: ['IODINE', 'IOTA', 'LEAD', 'SILICON']
- One away: ['DRYER', 'HAMPER', 'STEER', 'WASHER']
- Correct: ['DRYER', 'HAMPER', 'IRON', 'WASHER']
- Incorrect: ['DEATH', 'IODINE', 'IOTA', 'SILICON']

**Matched Groups: 1/4**

### Game 57
**Ground Truth Groups:**
- ['GRAVITY', 'IMPORT', 'SUBSTANCE', 'WEIGHT']
- ['ABYSS', 'FLY', 'MATRIX', 'THING']
- ['BENCH', 'DRAFT', 'START', 'TRADE']
- ['BANK', 'DUNE', 'HILL', 'MOUND']

**Predictions Sequence:**
- Incorrect: ['BENCH', 'HILL', 'MOUND', 'START']
- Incorrect: ['BENCH', 'DRAFT', 'HILL', 'MOUND']
- Incorrect: ['DRAFT', 'HILL', 'MOUND', 'START']
- One away: ['ABYSS', 'DUNE', 'HILL', 'MOUND']

**Matched Groups: 0/4**

### Game 58
**Ground Truth Groups:**
- ['ADHERE', 'GLUE', 'PASTE', 'STICK']
- ['CARAT', 'CLARITY', 'COLOR', 'CUT']
- ['COPY', 'TEXT', 'WORDS', 'WRITING']
- ['LIST', 'OK', 'PLUS', 'ROD']

**Predictions Sequence:**
- One away: ['CLARITY', 'TEXT', 'WORDS', 'WRITING']
- One away: ['COPY', 'PASTE', 'TEXT', 'WRITING']
- One away: ['COLOR', 'TEXT', 'WORDS', 'WRITING']
- One away: ['PASTE', 'TEXT', 'WORDS', 'WRITING']

**Matched Groups: 0/4**

### Game 59
**Ground Truth Groups:**
- ['FIRE', 'HEAT', 'KICK', 'SPICE']
- ['BOOM', 'CRASH', 'ROAR', 'THUNDER']
- ['BRUSH', 'CAPE', 'CLIPPERS', 'GEL']
- ['BASEBALL', 'MAGIC', 'SET', 'TAROT']

**Predictions Sequence:**
- Incorrect: ['FIRE', 'MAGIC', 'SPICE', 'TAROT']
- Incorrect: ['MAGIC', 'ROAR', 'TAROT', 'THUNDER']
- Incorrect: ['GEL', 'MAGIC', 'SPICE', 'TAROT']
- Incorrect: ['BASEBALL', 'CLIPPERS', 'HEAT', 'THUNDER']

**Matched Groups: 0/4**

### Game 60
**Ground Truth Groups:**
- ['CRADLE', 'HAMMOCK', 'ROCKER', 'SWING']
- ['FAZE', 'RATTLE', 'RUFFLE', 'THROW']
- ['CAN', 'ROLLER', 'SILO', 'SPOOL']
- ['CHAIN', 'CHAT', 'CONVERSATION', 'THREAD']

**Predictions Sequence:**
- One away: ['CHAT', 'CONVERSATION', 'CRADLE', 'THREAD']
- One away: ['CHAT', 'CONVERSATION', 'THREAD', 'THROW']
- Correct: ['CHAIN', 'CHAT', 'CONVERSATION', 'THREAD']
- Incorrect: ['RATTLE', 'ROCKER', 'ROLLER', 'RUFFLE']
- Incorrect: ['CRADLE', 'RATTLE', 'ROCKER', 'ROLLER']

**Matched Groups: 1/4**

### Game 61
**Ground Truth Groups:**
- ['BODY', 'BRIDGE', 'FRET', 'NECK']
- ['BROOD', 'CLUTCH', 'HATCH', 'LITTER']
- ['HASH', 'JUMBLE', 'MEDLEY', 'STEW']
- ['CHALLENGE', 'CONFRONT', 'FACE', 'OPPOSE']

**Predictions Sequence:**
- Incorrect: ['BODY', 'FACE', 'HASH', 'NECK']
- One away: ['BODY', 'FACE', 'FRET', 'NECK']
- Incorrect: ['BODY', 'CONFRONT', 'FACE', 'NECK']
- One away: ['BROOD', 'CLUTCH', 'HATCH', 'STEW']

**Matched Groups: 0/4**

### Game 62
**Ground Truth Groups:**
- ['BARN', 'CHICKEN', 'FARMER', 'TRACTOR']
- ['DAISY', 'ROSE', 'TULIP', 'VIOLET']
- ['ASTER', 'CARPENTER', 'CRAVEN', 'WAN']
- ['DUST', 'LIFE', 'SPORTS', 'YELLOW']

**Predictions Sequence:**
- One away: ['BARN', 'CARPENTER', 'FARMER', 'TRACTOR']
- Correct: ['DAISY', 'ROSE', 'TULIP', 'VIOLET']
- Incorrect: ['ASTER', 'CARPENTER', 'FARMER', 'TRACTOR']
- Correct: ['BARN', 'CHICKEN', 'FARMER', 'TRACTOR']
- One away: ['DUST', 'LIFE', 'SPORTS', 'WAN']
- Correct: ['DUST', 'LIFE', 'SPORTS', 'YELLOW']
- Correct: ['ASTER', 'CARPENTER', 'CRAVEN', 'WAN']

**Matched Groups: 4/4**

### Game 63
**Ground Truth Groups:**
- ['JAM', 'PACK', 'SQUEEZE', 'STUFF']
- ['CANARY', 'FINK', 'RAT', 'SNITCH']
- ['BUTTER', 'DRAGON', 'FIRE', 'HORSE']
- ['CAT', 'COW', 'MOUNTAIN', 'TRIANGLE']

**Predictions Sequence:**
- Incorrect: ['CAT', 'COW', 'HORSE', 'RAT']
- Incorrect: ['CANARY', 'CAT', 'HORSE', 'RAT']
- One away: ['CANARY', 'CAT', 'RAT', 'SNITCH']
- Incorrect: ['CANARY', 'CAT', 'DRAGON', 'RAT']

**Matched Groups: 0/4**

### Game 64
**Ground Truth Groups:**
- ['AIM', 'INTEND', 'MEAN', 'PLAN']
- ['FLEECE', 'GIRLS', 'PARACHUTE', 'RULE']
- ['CAL', 'GAL', 'IN', 'OZ']
- ['CURB', 'GRATE', 'GUTTER', 'MANHOLE']

**Predictions Sequence:**
- Correct: ['AIM', 'INTEND', 'MEAN', 'PLAN']
- One away: ['GRATE', 'GUTTER', 'MANHOLE', 'PARACHUTE']
- Incorrect: ['FLEECE', 'GRATE', 'GUTTER', 'PARACHUTE']
- One away: ['CAL', 'GAL', 'GIRLS', 'OZ']
- Incorrect: ['FLEECE', 'GUTTER', 'MANHOLE', 'PARACHUTE']

**Matched Groups: 1/4**

### Game 65
**Ground Truth Groups:**
- ['BITE', 'BLOW', 'STINK', 'SUCK']
- ['CAGE', 'ENO', 'GLASS', 'REICH']
- ['EMO', 'FUNK', 'METAL', 'POP']
- ['EAT', 'ROCK', 'RULE', 'SLAY']

**Predictions Sequence:**
- One away: ['BLOW', 'EAT', 'STINK', 'SUCK']
- One away: ['BLOW', 'RULE', 'STINK', 'SUCK']
- Correct: ['BITE', 'BLOW', 'STINK', 'SUCK']
- Incorrect: ['EMO', 'ENO', 'GLASS', 'METAL']
- Incorrect: ['GLASS', 'METAL', 'POP', 'ROCK']

**Matched Groups: 1/4**

### Game 66
**Ground Truth Groups:**
- ['BOAT', 'CAR', 'PLANE', 'TRAIN']
- ['KING', 'MAGIC', 'SUN', 'THUNDER']
- ['DAY', 'FLOYD', 'SABBATH', 'STRIPES']
- ['CHECKERS', 'OUTBACK', 'SONIC', 'SUBWAY']

**Predictions Sequence:**
- Correct: ['BOAT', 'CAR', 'PLANE', 'TRAIN']
- Incorrect: ['DAY', 'SABBATH', 'SUN', 'THUNDER']
- One away: ['MAGIC', 'SABBATH', 'SUN', 'THUNDER']
- Incorrect: ['DAY', 'MAGIC', 'SABBATH', 'SUN']
- Incorrect: ['DAY', 'SABBATH', 'SUBWAY', 'SUN']

**Matched Groups: 1/4**

### Game 67
**Ground Truth Groups:**
- ['DIRTY', 'HONEY', 'PLATINUM', 'STRAWBERRY']
- ['GAS', 'STINKER', 'TOOT', 'WIND']
- ['BRIGHT', 'FLASHY', 'GARISH', 'LOUD']
- ['HUG', 'OF', 'OXYGEN', 'ZERO']

**Predictions Sequence:**
- Correct: ['BRIGHT', 'FLASHY', 'GARISH', 'LOUD']
- Incorrect: ['GAS', 'OXYGEN', 'WIND', 'ZERO']
- Incorrect: ['GAS', 'HUG', 'OXYGEN', 'WIND']
- Incorrect: ['GAS', 'OXYGEN', 'STRAWBERRY', 'WIND']
- One away: ['GAS', 'OXYGEN', 'STINKER', 'WIND']

**Matched Groups: 1/4**

### Game 68
**Ground Truth Groups:**
- ['AMAZON', 'APPLE', 'INTUIT', 'ORACLE']
- ['BRIM', 'LIME', 'RHINE', 'YELLOW']
- ['FACE', 'FRONT', 'OUTSIDE', 'SURFACE']
- ['FLAIR', 'GIFT', 'INSTINCT', 'NOSE']

**Predictions Sequence:**
- Correct: ['AMAZON', 'APPLE', 'INTUIT', 'ORACLE']
- Incorrect: ['BRIM', 'FACE', 'FRONT', 'NOSE']
- Incorrect: ['BRIM', 'FACE', 'NOSE', 'SURFACE']
- Incorrect: ['BRIM', 'FACE', 'FLAIR', 'NOSE']
- Incorrect: ['BRIM', 'FACE', 'NOSE', 'RHINE']

**Matched Groups: 1/4**

### Game 69
**Ground Truth Groups:**
- ['EXPIRATION', 'NAME', 'NUMBER', 'ZIP']
- ['BUBBLY', 'CAB', 'CHARD', 'ZIN']
- ['GLOSSY', 'MAG', 'RAG', 'ZINE']
- ['BURN', 'KNOCK', 'SLIGHT', 'ZING']

**Predictions Sequence:**
- Incorrect: ['ZIN', 'ZINE', 'ZING', 'ZIP']
- Incorrect: ['MAG', 'ZIN', 'ZINE', 'ZING']
- Incorrect: ['EXPIRATION', 'ZIN', 'ZINE', 'ZING']
- Incorrect: ['NAME', 'ZIN', 'ZINE', 'ZING']

**Matched Groups: 0/4**

### Game 70
**Ground Truth Groups:**
- ['BUFFER', 'COMFORT', 'TIME', 'TWILIGHT']
- ['FLAVOR', 'SALT', 'SEASON', 'SPICE']
- ['CHARACTER', 'FILM', 'METHOD', 'STAGE']
- ['COLON', 'DASH', 'PERIOD', 'SLASH']

**Predictions Sequence:**
- One away: ['COLON', 'FLAVOR', 'SALT', 'SPICE']
- One away: ['FLAVOR', 'METHOD', 'SALT', 'SPICE']
- Correct: ['FLAVOR', 'SALT', 'SEASON', 'SPICE']
- Incorrect: ['BUFFER', 'DASH', 'SLASH', 'STAGE']
- Incorrect: ['DASH', 'FILM', 'SLASH', 'TWILIGHT']

**Matched Groups: 1/4**

### Game 71
**Ground Truth Groups:**
- ['CHEMISTRY', 'CONNECTION', 'FIREWORKS', 'SPARK']
- ['BANSHEE', 'HEARTBREAKER', 'PIP', 'WAILER']
- ['CARPET', 'LAMINATE', 'TILE', 'WOOD']
- ['BANK', 'COAST', 'SHORE', 'STRAND']

**Predictions Sequence:**
- Correct: ['CARPET', 'LAMINATE', 'TILE', 'WOOD']
- Incorrect: ['COAST', 'HEARTBREAKER', 'SHORE', 'WAILER']
- Correct: ['BANK', 'COAST', 'SHORE', 'STRAND']
- Correct: ['CHEMISTRY', 'CONNECTION', 'FIREWORKS', 'SPARK']
- Correct: ['BANSHEE', 'HEARTBREAKER', 'PIP', 'WAILER']

**Matched Groups: 4/4**

### Game 72
**Ground Truth Groups:**
- ['EON', 'ETHER', 'NET', 'TOW']
- ['BATH', 'CARD', 'CURTAIN', 'PICTURE']
- ['AIR', 'RUN', 'SCREEN', 'SHOW']
- ['REWIND', 'SHUFFLE', 'SKIP', 'STOP']

**Predictions Sequence:**
- One away: ['RUN', 'SHUFFLE', 'SKIP', 'STOP']
- Incorrect: ['BATH', 'CURTAIN', 'NET', 'SCREEN']
- One away: ['REWIND', 'RUN', 'SHUFFLE', 'SKIP']
- Incorrect: ['AIR', 'CARD', 'EON', 'ETHER']

**Matched Groups: 0/4**

### Game 73
**Ground Truth Groups:**
- ['BALANCED', 'DRY', 'FULL', 'SWEET']
- ['BEAR', 'HARBOR', 'HOLD', 'MAINTAIN']
- ['GOLDILOCKS', 'IRONIC', 'LEADERSHIP', 'TINDER']
- ['BED', 'CHAIR', 'COUCH', 'TABLE']

**Predictions Sequence:**
- Correct: ['BED', 'CHAIR', 'COUCH', 'TABLE']
- One away: ['DRY', 'FULL', 'HOLD', 'SWEET']
- One away: ['BEAR', 'GOLDILOCKS', 'LEADERSHIP', 'TINDER']
- Incorrect: ['DRY', 'FULL', 'HARBOR', 'HOLD']
- Incorrect: ['BALANCED', 'HOLD', 'IRONIC', 'SWEET']

**Matched Groups: 1/4**

### Game 74
**Ground Truth Groups:**
- ['BOTTLE', 'CARAFE', 'DECANTER', 'GLASS']
- ['CORAL', 'GARTER', 'KING', 'RATTLE']
- ['CLEFT', 'RENT', 'SPLIT', 'TORN']
- ['BOOTY', 'CLOSE', 'COLD', 'CURTAIN']

**Predictions Sequence:**
- Correct: ['BOTTLE', 'CARAFE', 'DECANTER', 'GLASS']
- One away: ['KING', 'RENT', 'SPLIT', 'TORN']
- One away: ['CURTAIN', 'RENT', 'SPLIT', 'TORN']
- One away: ['COLD', 'RENT', 'SPLIT', 'TORN']
- Incorrect: ['COLD', 'KING', 'SPLIT', 'TORN']

**Matched Groups: 1/4**

### Game 75
**Ground Truth Groups:**
- ['BOXER', 'JUDGE', 'MONK', 'WIZARD']
- ['DITCH', 'HOLE', 'PIT', 'TRENCH']
- ['BATTER', 'BOTTOMS', 'CHIN', 'LAWYER']
- ['DISCUSSION', 'LAB', 'LECTURE', 'SEMINAR']

**Predictions Sequence:**
- Correct: ['DITCH', 'HOLE', 'PIT', 'TRENCH']
- Correct: ['DISCUSSION', 'LAB', 'LECTURE', 'SEMINAR']
- One away: ['BOXER', 'JUDGE', 'LAWYER', 'MONK']
- One away: ['JUDGE', 'LAWYER', 'MONK', 'WIZARD']
- One away: ['BOXER', 'JUDGE', 'LAWYER', 'WIZARD']
- One away: ['BOXER', 'LAWYER', 'MONK', 'WIZARD']

**Matched Groups: 2/4**

### Game 76
**Ground Truth Groups:**
- ['BELT', 'GULP', 'SHOT', 'SLUG']
- ['EGG', 'HARD TACO', 'M&M', 'SNAIL']
- ['COMPLETE', 'DONE', 'OVER', 'THROUGH']
- ['BULL', 'LAP', 'UNDER', 'WATCH']

**Predictions Sequence:**
- One away: ['EGG', 'M&M', 'SLUG', 'SNAIL']
- Incorrect: ['EGG', 'GULP', 'SLUG', 'SNAIL']
- Correct: ['COMPLETE', 'DONE', 'OVER', 'THROUGH']
- One away: ['EGG', 'HARD TACO', 'SLUG', 'SNAIL']
- Incorrect: ['EGG', 'SHOT', 'SLUG', 'SNAIL']

**Matched Groups: 1/4**

### Game 77
**Ground Truth Groups:**
- ['AIOLI', 'BARBECUE', 'MARINARA', 'RANCH']
- ['BILL', 'INVOICE', 'RECEIPT', 'STATEMENT']
- ['ANCESTRY', 'BROTH', 'CATTLE', 'MERCHANDISE']
- ['EXAMINE', 'GRILL', 'PROBE', 'QUESTION']

**Predictions Sequence:**
- One away: ['BARBECUE', 'GRILL', 'MARINARA', 'RANCH']
- Incorrect: ['BARBECUE', 'CATTLE', 'GRILL', 'RANCH']
- One away: ['BILL', 'INVOICE', 'MERCHANDISE', 'RECEIPT']
- Incorrect: ['BARBECUE', 'BROTH', 'GRILL', 'MARINARA']

**Matched Groups: 0/4**

### Game 78
**Ground Truth Groups:**
- ['BRAT', 'DOG', 'FRANK', 'LINK']
- ['CHAIN', 'SERIES', 'STRING', 'TRAIN']
- ['DRIVER', 'IRON', 'WEDGE', 'WOOD']
- ['BIRDS', 'FOG', 'OMEN', 'RING']

**Predictions Sequence:**
- One away: ['CHAIN', 'RING', 'SERIES', 'STRING']
- Incorrect: ['CHAIN', 'RING', 'STRING', 'WOOD']
- Incorrect: ['CHAIN', 'LINK', 'RING', 'STRING']
- One away: ['CHAIN', 'RING', 'STRING', 'TRAIN']

**Matched Groups: 0/4**

### Game 79
**Ground Truth Groups:**
- ['DODGE', 'HAND', 'PICKLE', 'RACQUET']
- ['CELL', 'HOCK', 'PEDAL', 'TRAYED']
- ['BREEZE', 'COAST', 'FLOAT', 'SKATE']
- ['BACON', 'CAGE', 'CRUISE', 'FORD']

**Predictions Sequence:**
- Incorrect: ['DODGE', 'PEDAL', 'RACQUET', 'SKATE']
- One away: ['BREEZE', 'COAST', 'CRUISE', 'FLOAT']
- Incorrect: ['HAND', 'PEDAL', 'RACQUET', 'SKATE']
- Incorrect: ['BACON', 'CAGE', 'CELL', 'PICKLE']

**Matched Groups: 0/4**

### Game 80
**Ground Truth Groups:**
- ['FORTUNE', 'OATMEAL', 'RAINBOW', 'SUGAR']
- ['ARTS', 'RATS', 'STAR', 'TSAR']
- ['DARN', 'MEND', 'PATCH', 'STITCH']
- ['BLAST', 'BLOW', 'GALE', 'GUST']

**Predictions Sequence:**
- Correct: ['BLAST', 'BLOW', 'GALE', 'GUST']
- Correct: ['DARN', 'MEND', 'PATCH', 'STITCH']
- One away: ['OATMEAL', 'RAINBOW', 'RATS', 'SUGAR']
- Incorrect: ['OATMEAL', 'RATS', 'SUGAR', 'TSAR']
- One away: ['FORTUNE', 'OATMEAL', 'RATS', 'SUGAR']
- Incorrect: ['ARTS', 'OATMEAL', 'RATS', 'SUGAR']

**Matched Groups: 2/4**

### Game 81
**Ground Truth Groups:**
- ['FRAME', 'HANDLE', 'HINGE', 'LOCK']
- ['BLOW', 'BUMBLE', 'FLUFF', 'SPOIL']
- ['DRILL', 'GRINDER', 'ROUTER', 'SAW']
- ['KINDLING', 'LOG', 'MATCH', 'TINDER']

**Predictions Sequence:**
- One away: ['DRILL', 'GRINDER', 'ROUTER', 'TINDER']
- Incorrect: ['BUMBLE', 'GRINDER', 'ROUTER', 'TINDER']
- Correct: ['DRILL', 'GRINDER', 'ROUTER', 'SAW']
- Incorrect: ['BUMBLE', 'FLUFF', 'KINDLING', 'TINDER']
- One away: ['FRAME', 'HINGE', 'LOCK', 'MATCH']

**Matched Groups: 1/4**

### Game 82
**Ground Truth Groups:**
- ['BARBIE', 'BUSH', 'CRIKEY', 'MATE']
- ['EINSTEIN', 'FEYNMAN', 'HAWKING', 'OPPENHEIMER']
- ['MATCH', 'MOOT', 'SELLING', 'WEST']
- ['FACULTY', 'FLAIR', 'GENIUS', 'GIFT']

**Predictions Sequence:**
- Correct: ['EINSTEIN', 'FEYNMAN', 'HAWKING', 'OPPENHEIMER']
- Incorrect: ['CRIKEY', 'MATCH', 'MATE', 'MOOT']
- Incorrect: ['CRIKEY', 'FLAIR', 'MATCH', 'MATE']
- One away: ['BARBIE', 'CRIKEY', 'MATCH', 'MATE']
- Incorrect: ['CRIKEY', 'GENIUS', 'MATCH', 'MATE']

**Matched Groups: 1/4**

### Game 83
**Ground Truth Groups:**
- ['CIRCLE', 'HORSESHOE', 'PITCHFORK', 'TRIANGLE']
- ['BASKETBALL', 'CARROT', 'GOLDFISH', 'PUMPKIN']
- ['POLE', 'ROD', 'STAFF', 'STICK']
- ['CART', 'CLUB', 'HOLE', 'TEE']

**Predictions Sequence:**
- One away: ['CIRCLE', 'HOLE', 'HORSESHOE', 'TRIANGLE']
- Incorrect: ['CARROT', 'CART', 'PITCHFORK', 'PUMPKIN']
- One away: ['CARROT', 'GOLDFISH', 'PITCHFORK', 'PUMPKIN']
- Incorrect: ['CARROT', 'PITCHFORK', 'POLE', 'PUMPKIN']

**Matched Groups: 0/4**

### Game 84
**Ground Truth Groups:**
- ['CAMO', 'DRAB', 'FATIGUE', 'KHAKI']
- ['BLOW', 'LOSE', 'MISS', 'WASTE']
- ['HUB', 'RIM', 'SPOKE', 'TIRE']
- ['DUCT', 'EXHAUST', 'HOSE', 'PIPE']

**Predictions Sequence:**
- Correct: ['DUCT', 'EXHAUST', 'HOSE', 'PIPE']
- Correct: ['CAMO', 'DRAB', 'FATIGUE', 'KHAKI']
- Correct: ['HUB', 'RIM', 'SPOKE', 'TIRE']
- Correct: ['BLOW', 'LOSE', 'MISS', 'WASTE']

**Matched Groups: 4/4**

### Game 85
**Ground Truth Groups:**
- ['BEAR', 'GENERATE', 'PRODUCE', 'YIELD']
- ['ASIDE', 'DETOUR', 'DIGRESSION', 'TANGENT']
- ['ARE', 'RADIUS', 'REVERSE', 'RIGHT']
- ['BABE', 'FOX', 'SNACK', 'TEN']

**Predictions Sequence:**
- One away: ['GENERATE', 'PRODUCE', 'RADIUS', 'YIELD']
- Correct: ['BEAR', 'GENERATE', 'PRODUCE', 'YIELD']
- One away: ['ASIDE', 'DETOUR', 'DIGRESSION', 'REVERSE']
- Correct: ['ASIDE', 'DETOUR', 'DIGRESSION', 'TANGENT']
- Incorrect: ['ARE', 'FOX', 'RIGHT', 'TEN']
- Correct: ['BABE', 'FOX', 'SNACK', 'TEN']
- Correct: ['ARE', 'RADIUS', 'REVERSE', 'RIGHT']

**Matched Groups: 4/4**

### Game 86
**Ground Truth Groups:**
- ['BUD', 'LEAF', 'PETAL', 'STALK']
- ['DRAGON', 'HORSE', 'RABBIT', 'TIGER']
- ['GNOME', 'GOBLIN', 'OGRE', 'TROLL']
- ['AGENT', 'MOLE', 'PLANT', 'SPY']

**Predictions Sequence:**
- One away: ['DRAGON', 'GNOME', 'GOBLIN', 'OGRE']
- One away: ['GNOME', 'GOBLIN', 'MOLE', 'OGRE']
- Incorrect: ['DRAGON', 'GOBLIN', 'HORSE', 'OGRE']
- Correct: ['DRAGON', 'HORSE', 'RABBIT', 'TIGER']
- One away: ['BUD', 'LEAF', 'PETAL', 'PLANT']

**Matched Groups: 1/4**

### Game 87
**Ground Truth Groups:**
- ['DISTORT', 'DOCTOR', 'FUDGE', 'TWIST']
- ['DANCE', 'MEAL', 'ONE', 'ROOT']
- ['CAKE', 'COBBLER', 'PIE', 'TART']
- ['FISHER', 'MASON', 'MILLER', 'SMITH']

**Predictions Sequence:**
- Correct: ['FISHER', 'MASON', 'MILLER', 'SMITH']
- Incorrect: ['CAKE', 'FUDGE', 'MEAL', 'PIE']
- One away: ['CAKE', 'FUDGE', 'PIE', 'TART']
- Incorrect: ['CAKE', 'FUDGE', 'MEAL', 'TART']
- Incorrect: ['DISTORT', 'ROOT', 'TART', 'TWIST']

**Matched Groups: 1/4**

### Game 88
**Ground Truth Groups:**
- ['BOB', 'CLAP', 'SNAP', 'TAP']
- ['BUN', 'HERO', 'ROLL', 'WRAP']
- ['CHOPS', 'KISSER', 'TRAP', 'YAP']
- ['APPLAUSE', 'DRINKS', 'FUNDING', 'GOLF']

**Predictions Sequence:**
- One away: ['APPLAUSE', 'CLAP', 'SNAP', 'TAP']
- Incorrect: ['APPLAUSE', 'CLAP', 'ROLL', 'TAP']
- Incorrect: ['APPLAUSE', 'CLAP', 'SNAP', 'WRAP']
- Incorrect: ['APPLAUSE', 'CLAP', 'SNAP', 'YAP']

**Matched Groups: 0/4**

### Game 89
**Ground Truth Groups:**
- ['LOW', 'SHORT', 'SHY', 'WANTING']
- ['CLUB', 'DIAMOND', 'HEART', 'SPADE']
- ['ACHE', 'LONG', 'PINE', 'THIRST']
- ['IRON', 'PUTTER', 'WEDGE', 'WOOD']

**Predictions Sequence:**
- One away: ['PINE', 'PUTTER', 'WEDGE', 'WOOD']
- Incorrect: ['DIAMOND', 'IRON', 'PINE', 'WOOD']
- One away: ['IRON', 'PINE', 'PUTTER', 'WOOD']
- Incorrect: ['ACHE', 'HEART', 'THIRST', 'WANTING']

**Matched Groups: 0/4**

### Game 90
**Ground Truth Groups:**
- ['COUPLE', 'HITCH', 'LINK', 'TIE']
- ['ACOUSTIC', 'AUDITORY', 'HEARD', 'SONIC']
- ['AMP', 'FIRE', 'HYPE', 'PUMP']
- ['CABLE', 'ELECTRIC', 'GAS', 'WATER']

**Predictions Sequence:**
- Incorrect: ['ACOUSTIC', 'AMP', 'AUDITORY', 'ELECTRIC']
- Incorrect: ['FIRE', 'GAS', 'PUMP', 'WATER']
- One away: ['ELECTRIC', 'FIRE', 'GAS', 'WATER']
- One away: ['ACOUSTIC', 'AUDITORY', 'ELECTRIC', 'SONIC']

**Matched Groups: 0/4**

### Game 91
**Ground Truth Groups:**
- ['CAMEROON', 'COCKATIEL', 'UNREAD', 'WHISTLEBLEW']
- ['BUST', 'LEMON', 'MISS', 'TURKEY']
- ['HOOK', 'LUFFY', 'SILVER', 'SPARROW']
- ['BEAR', 'STAND', 'STOMACH', 'SWALLOW']

**Predictions Sequence:**
- Incorrect: ['CAMEROON', 'LEMON', 'SILVER', 'TURKEY']
- Incorrect: ['CAMEROON', 'COCKATIEL', 'SPARROW', 'TURKEY']
- Incorrect: ['BEAR', 'CAMEROON', 'LEMON', 'TURKEY']
- Incorrect: ['CAMEROON', 'COCKATIEL', 'LEMON', 'TURKEY']

**Matched Groups: 0/4**

### Game 92
**Ground Truth Groups:**
- ['HANDSOME', 'HIPPO', 'LEGEND', 'LIPID']
- ['BIG', 'HOT', 'IN', 'POPULAR']
- ['ASSESS', 'CHARGE', 'FINE', 'LEVY']
- ['CHARACTER', 'GLYPH', 'ICON', 'SYMBOL']

**Predictions Sequence:**
- Correct: ['CHARACTER', 'GLYPH', 'ICON', 'SYMBOL']
- One away: ['BIG', 'HANDSOME', 'HOT', 'POPULAR']
- Incorrect: ['BIG', 'FINE', 'HANDSOME', 'HOT']
- Incorrect: ['FINE', 'HANDSOME', 'HOT', 'POPULAR']
- Incorrect: ['FINE', 'HANDSOME', 'HOT', 'IN']

**Matched Groups: 1/4**

### Game 93
**Ground Truth Groups:**
- ['BEGINS', 'INSTITUTES', 'LAUNCHES', 'PIONEERS']
- ['KITE', 'ROBIN', 'SWALLOW', 'SWIFT']
- ['DAYS', 'ENDING', 'RETURNS', 'TRAILS']
- ['ETERNAL', 'EVERLASTING', 'FOREVER', 'PERMANENT']

**Predictions Sequence:**
- Correct: ['ETERNAL', 'EVERLASTING', 'FOREVER', 'PERMANENT']
- One away: ['KITE', 'PIONEERS', 'ROBIN', 'SWIFT']
- Correct: ['KITE', 'ROBIN', 'SWALLOW', 'SWIFT']
- One away: ['INSTITUTES', 'LAUNCHES', 'PIONEERS', 'TRAILS']
- One away: ['BEGINS', 'LAUNCHES', 'PIONEERS', 'TRAILS']
- Incorrect: ['BEGINS', 'ENDING', 'LAUNCHES', 'RETURNS']

**Matched Groups: 2/4**

### Game 94
**Ground Truth Groups:**
- ['BUST', 'HIT', 'SPLIT', 'STAND']
- ['BENCH', 'PIGEON', 'STATUE', 'TREE']
- ['CORE', 'MIDSECTION', 'TORSO', 'TRUNK']
- ['BACKUP', 'COVER', 'RELIEF', 'SUBSTITUTE']

**Predictions Sequence:**
- Correct: ['CORE', 'MIDSECTION', 'TORSO', 'TRUNK']
- Incorrect: ['BUST', 'HIT', 'PIGEON', 'STATUE']
- One away: ['PIGEON', 'STAND', 'STATUE', 'TREE']
- Correct: ['BACKUP', 'COVER', 'RELIEF', 'SUBSTITUTE']
- Incorrect: ['BENCH', 'BUST', 'STAND', 'STATUE']
- One away: ['BENCH', 'STAND', 'STATUE', 'TREE']

**Matched Groups: 2/4**

### Game 95
**Ground Truth Groups:**
- ['BOXER', 'BRIEF', 'DRAWER', 'SHORT']
- ['CUE', 'PROMPT', 'SIGNAL', 'WORD']
- ['CLUTCH', 'MESSENGER', 'SATCHEL', 'TOTE']
- ['FREEDOM', 'LATITUDE', 'LICENSE', 'SLACK']

**Predictions Sequence:**
- Incorrect: ['BRIEF', 'CUE', 'PROMPT', 'SHORT']
- Incorrect: ['BOXER', 'DRAWER', 'SATCHEL', 'TOTE']
- Incorrect: ['BRIEF', 'CUE', 'SHORT', 'WORD']
- Incorrect: ['BRIEF', 'PROMPT', 'SHORT', 'WORD']

**Matched Groups: 0/4**

### Game 96
**Ground Truth Groups:**
- ['AMERICAN', 'BLUE', 'JACK', 'SWISS']
- ['HAMMER', 'HURDLE', 'JAVELIN', 'POLE']
- ['CUBAN', 'KITTEN', 'STILETTO', 'WEDGE']
- ['DATE', 'DUTCH', 'JEOPARDY', 'SPACE']

**Predictions Sequence:**
- Incorrect: ['AMERICAN', 'CUBAN', 'DUTCH', 'SWISS']
- One away: ['AMERICAN', 'BLUE', 'DUTCH', 'SWISS']
- Incorrect: ['AMERICAN', 'DATE', 'DUTCH', 'SWISS']
- Incorrect: ['AMERICAN', 'DUTCH', 'JEOPARDY', 'SWISS']

**Matched Groups: 0/4**

### Game 97
**Ground Truth Groups:**
- ['FROM', 'SEA', 'SHINING', 'TO']
- ['FINANCE', 'IT', 'LEGAL', 'SALES']
- ['BOOTH', 'STALL', 'STAND', 'TABLE']
- ['COME', 'HEEL', 'SIT', 'STAY']

**Predictions Sequence:**
- Incorrect: ['SIT', 'STALL', 'STAND', 'STAY']
- One away: ['COME', 'SIT', 'STAND', 'STAY']
- Incorrect: ['COME', 'STALL', 'STAND', 'STAY']
- One away: ['SIT', 'STALL', 'STAND', 'TABLE']

**Matched Groups: 0/4**

### Game 98
**Ground Truth Groups:**
- ['ACES', 'KEEN', 'NEATO', 'NIFTY']
- ['BUBBLE', 'GLOBE', 'MARBLE', 'PEARL']
- ['FOAM', 'FROTH', 'HEAD', 'LATHER']
- ['BUILD', 'GROW', 'MOUNT', 'SWELL']

**Predictions Sequence:**
- One away: ['FOAM', 'FROTH', 'LATHER', 'SWELL']
- Incorrect: ['BUBBLE', 'FOAM', 'MARBLE', 'SWELL']
- One away: ['FOAM', 'FROTH', 'LATHER', 'MARBLE']
- One away: ['BUBBLE', 'FOAM', 'FROTH', 'LATHER']

**Matched Groups: 0/4**

### Game 99
**Ground Truth Groups:**
- ['CHEER', 'FRIEND', 'SCRUB', 'WEED']
- ['FUND', 'HOG', 'MAZE', 'TRIMMER']
- ['JOURNAL', 'LOG', 'RECORD', 'REGISTER']
- ['MANE', 'MOP', 'SHOCK', 'TANGLE']

**Predictions Sequence:**
- Correct: ['JOURNAL', 'LOG', 'RECORD', 'REGISTER']
- Incorrect: ['HOG', 'MOP', 'SCRUB', 'TRIMMER']
- Incorrect: ['HOG', 'MANE', 'MAZE', 'TANGLE']
- Incorrect: ['MOP', 'SCRUB', 'TRIMMER', 'WEED']
- Incorrect: ['CHEER', 'MOP', 'SCRUB', 'TRIMMER']

**Matched Groups: 1/4**

### Game 100
**Ground Truth Groups:**
- ['CYMBAL', 'KICK', 'SNARE', 'TOM']
- ['INFANT', 'PUB', 'SWIMMING', 'TICKER']
- ['DRIVER', 'EAGLE', 'HOLE', 'STROKE']
- ['BREAST', 'DRUMSTICK', 'TENDER', 'WING']

**Predictions Sequence:**
- One away: ['CYMBAL', 'DRUMSTICK', 'KICK', 'SNARE']
- Incorrect: ['CYMBAL', 'DRUMSTICK', 'SNARE', 'TICKER']
- One away: ['CYMBAL', 'DRUMSTICK', 'SNARE', 'TOM']
- Incorrect: ['CYMBAL', 'DRIVER', 'DRUMSTICK', 'SNARE']

**Matched Groups: 0/4**

## Overall Metrics

- **Number of games completely solved**: 11 / 100 (11%)
- **Total number of groups completely solved**: 97 / 400 (24.25%)
- **Number of groups in which 3 words correct and 1 wrong**: 173 / 400 (43.25%)
- **Number of games with exactly 2 solved groups and 2 one-away groups**: 3 / 100 (3%)

## Why This Method Still Struggles

Despite being an improvement over GloVe, several failure modes persist:

- **Ambiguity**: Without external knowledge, words with multiple meanings (polysemy) create blurred vector positions.
- **Abstract Logic**: The model cannot 'reason' about categories that aren't based on semantic usage (e.g., phonetic patterns).
- **Cascade Effect**: A single high-similarity 'Red Herring' choice removes words from the board that belong to other groups, making the global solution unsolvable.
