# We use colorama for some debug/analysis outputs
from colorama import Style
import numpy as np

# Stores all information about features of moves
class MoveAnalysis:
    # Here is a list of all features and attributes we record for each move when analysing a game
    def __init__(self, move_num, move, white, best_eval, time_spent, levels, level_amp, book=False,
                 mate_in=None, imbalance=0, shape=0, shape_proba=[], eval_drop=0, partial_credit=0,
                 line_length=0, best_move=None, best_levels=None, best_amp=None, best_shape=None,
                 best_shape_proba=None, next_best_evals=None, changes=0, risk=None):
        # Basic information, such as ply, which colour played it, and the move itself (e.g. Nf3)
        self.move_num = move_num
        self.move = move
        self.white = white

        # The evaluation of the best move in the position - in theory every move that isn't the
        # best move should have an evaluation lower than this number. In practice, this is not the
        # case, which is a phenomenon we observe and account for
        self.best_eval = best_eval

        # Engine evaluation on various depths, and other information about the 'time series' data
        self.levels = levels
        self.level_amplitude = level_amp

        # Details about whether the move was played as prepared theory and how long was spent on it
        self.time_spent = time_spent
        self.book = book

        # Check if mate counter decreases by exactly 1 each turn
        # If mate counter is high, this indicates extreme precision (e.g. M9 -> M8 -> ... -> M1)
        self.mate_in = mate_in

        # Can check if a sacrifice has been made
        self.imbalance = imbalance

        # 0 = u-shaped and ascending
        # 1 = n-shaped and descending
        # 2 = noisy or flat
        self.shape = shape
        # Probability or confidence in the classification
        self.shape_proba = shape_proba

        # The evaluation difference (or drop) between the best move and the played move
        self.eval_drop = eval_drop
        self.partial_credit = partial_credit

        # If the move was part of a tactic, this records how many moves there are in the tactic
        self.line_length = line_length

        # We also store information about the best move and analysis we have performed on it
        self.best_move = best_move
        self.best_levels = best_levels
        self.best_amp = best_amp
        self.best_shape = best_shape
        self.best_shape_proba = best_shape_proba
        self.next_best_evals = next_best_evals

        # This is the number of times the engine changed its mind about what the best move is
        self.changes = changes
        
        # The risk of a move, when considering if we are in a checkmate position
        self.risk = risk
    
    # Display features about the move in the terminal - used for debugging/analysis
    def __str__(self):
        if self.mate_in is None:
            # If the evaluation isn't checkmate then print the evaluation levels and other data
            out = f"{Style.BRIGHT if self.white else Style.DIM}{self.move_num:>2}{'w' if self.white else 'b'}. {self.move} {self.best_eval:>4.0f} {self.eval_drop:>4.0f} ({self.imbalance:>2}) | "
            for l in self.levels:
                out += f"{l:>5.2f} "
            out += f"| {self.level_amplitude:.2f} : {self.partial_credit:.2f}\n{self.shape} {self.shape_proba}" if not self.book else "(book)"
        else:
            # Otherwise format it to display the checkmate evaluation
            out = f"{Style.BRIGHT if self.white else Style.DIM}{self.move_num:>2}{'w' if self.white else 'b'}: {self.move} ({self.imbalance}) | "
            if self.mate_in < 0:
                out += "-"
            out += f"M{abs(self.mate_in)}"
        if self.best_move is not None:
            out += "\n" + str(self.best_move)
        return out
    
    # Used when generating training data for SVC model
    def data(self):
        combined = [self.shape]
        combined.extend(self.levels)
        return combined
    
    # Generates metrics for the main machine learning model
    def metrics(self, fast=False):
        # The amplitude of the evaluation levels vector
        x0 = self.level_amplitude

        # Fit a cubic polynomial and record the coefficients (an estimate of the seldepth 30 shape)
        coefs, x5, _, _, _ = np.polyfit(np.arange(len(self.levels)), self.levels, 3, full=True)
        x1, x2, x3, x4 = coefs
        x5 = x5[0]
        
        # SVC multi depth analysis trend
        x6 = self.shape

        # Measure of on average where the evaluation hovered compared to the best evaluation
        x7 = np.sum(self.levels)/len(self.levels)
        
        # Number of times the best move changed during analysis
        x8 = self.changes

        # Evaluation of the current position (important to give context to the move)
        good = 1 if self.white else -1
        x9 = self.best_eval*good/300

        # Evaluations of next best moves (4 values)
        x10 = abs(self.next_best_evals[0] - self.next_best_evals[1:])/300

        # Similarity of the move played to the best move
        x11 = 0
        for c in range(min(len(self.move), len(self.best_move))):
            x11 += abs(ord(self.move[c])-ord(self.best_move[c]))

        # Is white or black moving?
        x12 = 1 if self.white else 0

        # Quality of the move
        x13 = self.eval_drop*good/300
        if not fast:
            return np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, *x10, x11, x12, x13])
        else:
            return np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9,       x11, x12, x13])

    # Calculate how noisy the evaluation levels vector was, used for analysis/debugging
    def noise(self):
        total = 0
        for i in range(1,len(self.best_levels)-1):
            total += abs(self.best_levels[i] - self.best_levels[i-1])
        return total * self.level_amplitude