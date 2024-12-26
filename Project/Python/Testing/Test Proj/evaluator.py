# evaluator.py

from typing import List, Tuple
from enums import Pattern, Color


class Evaluator:
    """
    Evaluates the board based on detected patterns.
    """
    def __init__(self):
        """
        Initializes the Evaluator.
        
        You can initialize any scoring parameters or weights here.
        """
        # Define weights for different patterns
        self.pattern_weights = {
            Pattern.F5: 100000,
            Pattern.OL: -1000, 
            Pattern.OLR: -100000,
            Pattern.OLF: 100000,
            Pattern.F4: 10000,
            Pattern.B4: 5000,
            Pattern.F3S: 1000,
            Pattern.F3: 1000,
            Pattern.F2B: 500,
            Pattern.F2A: 500,
            Pattern.F2: 200,
            Pattern.B3: 500,
            Pattern.B2: 200,
            Pattern.B1: 100,
            Pattern.F1: 50,
            Pattern.DEAD: 0,
            # Add more patterns and their corresponding weights as needed
        }

    def evaluate(self, patterns: List[Tuple[Pattern, Color]], ai_color: Color) -> float:
        """
        Evaluates the board based on detected patterns for both AI and opponent.

        Args:
            patterns (List[Tuple[Pattern, Color]]): A list of tuples containing patterns and their corresponding player colors.
            ai_color (Color): The AI's color.

        Returns:
            float: The evaluation score (AI's advantage minus opponent's advantage).
        """
        score = 0.0
        for pattern, color in patterns:
            if pattern in self.pattern_weights:
                if color == ai_color:
                    score += self.pattern_weights[pattern]
                else:
                    score -= self.pattern_weights[pattern]
        return score
    
    def static_evaluator(self, patterns: List[Tuple[Pattern, Color]], ai_color: Color) -> float:
        score = 0.0
        for pattern in patterns:
            if pattern in self.pattern_weights:
                score += self.pattern_weights[pattern]
        return score

