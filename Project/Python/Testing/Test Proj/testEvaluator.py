# test_evaluator.py

from evaluator import Evaluator
from enums import Pattern, Color
from typing import List, Tuple

def generate_patterns(score: int) -> List[Pattern]:
    # Simplistic pattern generation based on score for testing
    patterns = []
    remaining = score
    if remaining >= 100000:
        patterns.append(Pattern.F5)
        remaining -= 100000
    if remaining >= 10000:
        patterns.append(Pattern.F4)
        remaining -= 10000
    if remaining >= 5000:
        patterns.append(Pattern.B4)
        remaining -= 5000
    if remaining >= 1000:
        patterns.append(Pattern.F3S)
        remaining -= 1000
    if remaining >= 500:
        patterns.append(Pattern.B3)
        remaining -= 500
    if remaining >= 100:
        patterns.append(Pattern.F3)
        remaining -= 100
    if remaining >= 50:
        patterns.append(Pattern.B2)
        remaining -= 50
    if remaining >= 10:
        patterns.append(Pattern.F2B)
        remaining -= 10
    if remaining >= 5:
        patterns.append(Pattern.F2A)
        remaining -= 5
    if remaining >= 2:
        patterns.append(Pattern.F2)
        remaining -= 2
    return patterns

def main():
    max_score = 100000
    evaluator = Evaluator(max_score=max_score, desired_win_rate=0.99, scaling_factor=20000)
    
    test_scores = [0, 500, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
    
    for score in test_scores:
        patterns = generate_patterns(score)
        win_rate = evaluator.evaluate(patterns, Color.BLACK)
        print(f"Score: {score}, Win Rate: {win_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
