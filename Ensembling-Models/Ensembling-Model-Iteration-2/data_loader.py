import pandas as pd
from typing import List, Dict, Any

def load_games(csv_path: str) -> List[Dict[str, Any]]:
    """
    Loads Connections data from a CSV file.
    Expected columns: Game ID, Puzzle Date, Word, Group Name, Group Level, Starting Row, Starting Column
    
    Returns a list of dictionaries, one for each game:
    {
        "game_id": int,
        "date": str,
        "words": list of 16 str,
        "groups": {
            0: list of 4 str (Group Level 0 words),
            1: list of 4 str,
            2: list of 4 str,
            3: list of 4 str
        },
        "group_names": {
            0: str,
            1: str,
            2: str,
            3: str
        }
    }
    """
    # Keep 'NA' as a string instead of NaN
    df = pd.read_csv(csv_path, keep_default_na=False)
    games = []
    
    # Group by Game ID
    for game_id, group_df in df.groupby("Game ID"):
        if len(group_df) != 16:
            # Skip games that don't have exactly 16 words (if any)
            continue
            
        date = group_df["Puzzle Date"].iloc[0]
        words = group_df["Word"].tolist()
        
        groups = {}
        group_names = {}
        
        for _, row in group_df.iterrows():
            level = row["Group Level"]
            word = row["Word"]
            name = row["Group Name"]
            
            if level not in groups:
                groups[level] = []
                group_names[level] = name
            groups[level].append(word)
            
        games.append({
            "game_id": int(game_id),
            "date": str(date),
            "words": words,
            "groups": groups,
            "group_names": group_names
        })
        
    return games

if __name__ == "__main__":
    games = load_games("Connections_Data.csv")
    print(f"Loaded {len(games)} games.")
    if games:
        print("Sample Game 1:")
        print(games[0])
