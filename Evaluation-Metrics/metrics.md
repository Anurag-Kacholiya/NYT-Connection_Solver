## Evaluation Metric: Search Efficiency (Attempts per Puzzle)

### Definition
In addition to measuring simple accuracy, we also evaluate **Search Efficiency**, which refers to the **number of guesses (attempts) required by the model to solve a puzzle**.  
In the Connections game, the total search space for all possible groupings exceeds **2.6 million combinations**. A naive agent would approach this problem through exhaustive or random guessing. In contrast, an intelligent agent should be able to **prioritize semantically meaningful clusters and converge on the correct solution with fewer attempts**.

### Significance
This metric helps evaluate the model’s **decision-making ability during the search process**. Instead of only measuring whether the puzzle was solved, it measures **how efficiently the model arrived at the solution**.

The number of attempts also reflects the model’s ability to:
- Rank high-probability semantic groups
- Handle misleading word associations
- Recover from incorrect guesses using feedback

To interpret the results, we divide puzzle-solving behavior into two main performance tiers:

- **Optimal Convergence (1–4 attempts):**  
  The model identifies the correct semantic groups quickly, suggesting that the embeddings capture the relationships between words effectively.

- **Iterative Recovery (5+ attempts):**  
  The model initially falls into **semantic traps or distractor groupings**, but is able to adjust its reasoning and eventually reach the correct solution.

### Analysis Goal
By analyzing the **distribution of attempts per puzzle**, we can distinguish between:
- **Confident solves**, where the model finds the correct groups almost immediately
- **Iterative solves**, where the model gradually improves its guesses after receiving feedback

This analysis allows us to demonstrate that our **Early Fusion architecture (MPNet + Numberbatch)** is not only accurate but also **efficient in navigating the large combinatorial search space**. Ideally, an effective model should minimize unnecessary guesses and converge to the correct grouping with fewer attempts.

---

### Guess Distribution Table

| Attempts Required | Interpretation |
|------------------|----------------|
| **1 – 4** | Perfect or Near-Perfect Efficiency |
| **5 – 10** | High Recovery Efficiency |
| **10+** | Distractor-Induced Search |

This table helps summarize how efficiently the model solves puzzles and provides a clearer picture of its search behavior beyond simple accuracy.