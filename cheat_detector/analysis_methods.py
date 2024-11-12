import chess
from math import exp
import numpy as np
from scipy.signal import filtfilt

# Make all evaluations numerical, so M5 -> 10000-5, M6 -> 10000-6 and so on, Mx -> 10000-x
def score(result):
    return result['score'].white().score(mate_score=10000)

# Calculate the risk associated with a given move (in the context of checkmating)
def mate_complexity(sf:chess.engine.SimpleEngine, best_eval:float, board:chess.Board,
                    move:int, played_move:chess.Move):
    # Assign a value to each piece on the board
    piece_values = {
        "K": 0, "k": 0,
        "Q": 9, "q":-9,
        "R": 5, "r":-5,
        "B": 3, "b":-3,
        "N": 3, "n":-3,
        "P": 1, "p":-1,
    }

    # Calculate the current imbalance (sum of the values of all pieces on the board)
    current_pieces = board.piece_map()
    current_imbalance = sum(piece_values[piece.symbol()] for piece in current_pieces.values())

    # Work out whether white or black played the move based on the ply, then decide whether
    # having a positive imbalance is good or bad
    good = 1 if move % 2 else -1

    if best_eval*good*-1 > 9000:
        # Count moves to mate
        mate_in = 10000 - abs(best_eval)
    else:
        # Otherwise the position is not near checkmate or the current colour is not winning
        return None, current_imbalance, None

    # Have a (quick) look at viable moves in the position
    current_imbalance *= good
    all_moves = sf.analyse(board, chess.engine.Limit(time=0.25), multipv=10)

    imbalances = []
    # Look at all (top 10) available moves
    for mv in all_moves:
        future_board = board.copy()

        # If the move is part of a sequence, push all following moves to reach the end
        if len(mv['pv']) >= 2:
            for pv_move in mv['pv']:
                future_board.push(pv_move)
        else:
            imbalances.append(current_imbalance)

        # Look at the imbalance at the end of the line
        future_pieces = board.piece_map()
        future_imbalance = sum(piece_values[piece.symbol()] for piece in future_pieces.values())
        imbalances.append(future_imbalance)
    
    # Use the softmax formula to weight the moves by risk
    exp_total = sum([exp(i) for i in imbalances])
    risks = [exp(i)/exp_total for i in imbalances]

    # We keep track of the lowest risk move we encounter and what risk the move played was
    # The tuple is (risk, evaluation, move)
    lowest_risk = (1, 0, 0)
    played_risk = None

    # Find the move with the lowest risk
    for m,mv in enumerate(all_moves):
        # Check if this move is the lowest risk move that still forces checkmate
        if risks[m] < lowest_risk[0] and score(mv) > 9000:
            lowest_risk = (risks[m], score(mv), mv['pv'][0].uci())

        # If we encounter the move that was played, record its risk too
        if mv['pv'][0].uci() == played_move.uci():
            played_risk = (risks[m], score(mv))
    
    # If the played move was bad enough not to be in the top 10 engine lines
    if played_risk is None:
        return mate_in, current_imbalance, None
    # If the played move was more risky than the lowest risk option
    elif played_risk[0] > lowest_risk[0] and played_risk[1] > lowest_risk[1]:
        return mate_in, current_imbalance, played_risk[0]-lowest_risk[0]
    # Otherwise the move was the lowest risk option
    else:
        return mate_in, current_imbalance, None

# Calculate the length of a tactical sequence composed of only forced moves
def getLength(sf:chess.engine.SimpleEngine, board:chess.Board):
    # Check if the game is over
    if board.is_game_over():
        return 0

    # Analyse the top two lines briefly
    result = sf.analyse(board, chess.engine.Limit(time=0.5), multipv=2)

    # If there are two or more moves available, check the difference in quality between them
    if len(result) >= 2:
        diff = score(result[0]) - score(result[1])

    # Check if the difference between the best and next best moves is large
    if len(result) < 2 or abs(diff) > 30:
        # Push the move to the stack and recursively get the length of the rest of the tactic
        board.push(result[0]['pv'][0])
        val = 1 + getLength(sf, board)
        # Pop the move from the stack and return the length
        board.pop()
        return val
    else:
        # If there isn't a clear best move then it isn't a forced sequence
        return 0

# Smooth and normalise an evaluation levels vector 'levels' based on a reference value
def normalise(levels:list[float], ref_eval:float, move:int):
    # Center around the reference evaluation
    levels = levels - ref_eval

    # Negate evaluation for black because negative is good
    if not move % 2:
        levels *= -1
    
    # Smooth the evaluation curve to mitigate the horizon effect noise
    filtered = filtfilt([1/3 for _ in range(3)], 3, levels)

    # Normalise the data such that the highest/lowest peak in the graph has an amplitude of 1
    max_amp = np.amax(np.abs(filtered))
    # Don't divide by 0
    if max_amp != 0:
        filtered = [l/max_amp for l in filtered]
    filtered = np.array(filtered)

    return filtered, max_amp