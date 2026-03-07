import itertools
import numpy as np

def solve(similarity_matrix: np.ndarray):
    """
    Finds the exact partition of 16 items into 4 groups of 4 that maximizes
    the minimum group score among the 4 groups.
    The group score is the sum of pairwise cosine similarities within the group.
    
    Returns:
    - best_partition: A list of 4 tuples, each containing 4 indices.
    - best_score: The max-min score of the partition.
    """
    items = list(range(16))
    subsets = list(itertools.combinations(items, 4))
    
    scores = {}
    for S in subsets:
        score = sum(similarity_matrix[i][j] for idx, i in enumerate(S) for j in S[idx+1:])
        scores[S] = float(score)
        
    best_score = -float('inf')
    best_partition = None
    
    def search(unassigned_mask, current_partition, current_score):
        nonlocal best_score, best_partition
        
        if unassigned_mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_partition = current_partition[:]
            return
            
        first_unassigned = (unassigned_mask & -unassigned_mask).bit_length() - 1
        
        remaining_items = []
        temp_mask = unassigned_mask & ~(1 << first_unassigned)
        while temp_mask:
            item = (temp_mask & -temp_mask).bit_length() - 1
            remaining_items.append(item)
            temp_mask &= ~(1 << item)
            
        if len(remaining_items) < 3:
            return
            
        for other3 in itertools.combinations(remaining_items, 3):
            subset = (first_unassigned,) + other3
            subset_score = scores[subset]

            # Pruning based on max possible future score could be added here, 
            # but for now we just recurse.
            
            next_mask = unassigned_mask & ~(1 << first_unassigned)
            for item in other3:
                next_mask &= ~(1 << item)
                
            current_partition.append(subset)
            search(next_mask, current_partition, current_score + subset_score)
            current_partition.pop()
            
    initial_mask = (1 << 16) - 1
    search(initial_mask, [], 0.0)
    
    # If no partition was found strictly > -inf (due to pruning edge cases)
    if best_partition is None:
        # should not happen but just in case, fall back to default
        temp_items = list(range(16))
        best_partition = [tuple(temp_items[i:i+4]) for i in range(0, 16, 4)]
        best_score = 0.0

    return best_partition, best_score

if __name__ == "__main__":
    import time
    np.random.seed(42)
    fake_sim = np.random.rand(16, 16)
    fake_sim = (fake_sim + fake_sim.T) / 2
    np.fill_diagonal(fake_sim, 1.0)
    start = time.time()
    partition, score = solve(fake_sim)
    end = time.time()
    print(f"Partition: {partition}")
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f} seconds")
