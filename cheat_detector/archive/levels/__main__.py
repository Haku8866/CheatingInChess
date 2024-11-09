import chess.pgn
import numpy as np
import chess.engine
from math import exp
import chess.polyglot
import matplotlib.pyplot as plt
from scipy.signal import lfilter, filtfilt

class MoveAnalysis:
    def __init__(self, move_num, move, white, best_eval, time_spent, levels, level_amp, book=False,
                 mate_in=None, shape=0, imbalance=0):
        self.move_num = move_num
        self.move = move
        self.white = white
        self.best_eval = best_eval
        self.levels = levels
        self.level_amplitude = level_amp
        self.time_spent = time_spent
        self.book = book
        # Check if mate counter decreases by exactly 1 each turn
        # If mate counter is high, this indicates extreme precision (e.g. M9 -> M8 -> ... -> M1)
        self.mate_in = mate_in
        # 0 = u-shaped and ascending
        # 1 = n-shaped and descending
        # 2 = noisy
        # 3 = flat
        self.shape = shape
        # Can check if a sacrifice has been made
        self.imbalance = imbalance
    
    def __str__(self) -> str:
        if self.mate_in is None:
            out = f"[ {self.move_num:>2}{'w' if self.white else 'b'}: {self.move} {self.time_spent:>5.1f}s {self.best_eval:>5.0f} ({self.imbalance}) | "
            for l in self.levels:
                out += f"{l:>5.2f} "
            out += f"| {self.level_amplitude:.2f}]" if not self.book else "(book) ]"
        else:
            out = f"[ {self.move_num:>2}{'w' if self.white else 'b'}: {self.move} {self.time_spent:>5.1f}s ({self.imbalance}) | "
            if self.mate_in < 0:
                out += "-"
            out += f"M{abs(self.mate_in)}"
        return out
    
    def data(self):
        combined = [self.shape]
        combined.extend(self.levels)
        return combined

def score(result):
    return result['score'].white().score(mate_score=10000)

def mate_complexity(sf, sf_depth, board):
    resultsf = sf.analyse(board, chess.engine.Limit(depth=sf_depth))
    score = resultsf['score'].white().score(mate_score=10000)
    
    if abs(score) > 9000:
        # Count moves to mate
        if score > 0:
            mate_in = 10000 - score
        else:
            mate_in = score + 10000
    else:
        mate_in = None

    piece_values = {
        "K": 0, "k": 0,
        "Q": 9, "q":-9,
        "R": 5, "r":-5,
        "B": 3, "b":-3,
        "N": 3, "n":-3,
        "P": 1, "p":-1,
    }

    # Check if material is unequal
    imbalance = 0
    pieces = board.piece_map()
    for square in pieces:
        imbalance += piece_values[pieces[square].symbol()]

    # An optimal checkmate may be easier to find if one side has many more pieces
    return mate_in, imbalance

def main():
    # Start stockfish
    sf = chess.engine.SimpleEngine.popen_uci("engine\stockfish.exe")

    # Open the openings book
    book = chess.polyglot.open_reader(r"openings\baron30.bin")

    # Load the game in question

    game_name = "me_ege"
    db = open(f"db\{game_name}.pgn")
    game = chess.pgn.read_game(db)

    # Start off in book on move 0
    inbook = True
    move = 0
    results = []

    # We are analysing to depth 20
    sf_depth = 20

    # fig = plt.figure(figsize=(200,200))

    # movehighlight = 33

    # Scroll through every move in the game
    while game.next() is not None:
        board = game.board()

        # Record the time spent thinking about the move, this will be used later
        clock = game.next().clock()
        if clock is None:
            clock = 1

        levels = []
        next_move = game.next().move
        played = game.next().board()
        score = 0
        pos_eval = 0
        # sf = chess.engine.SimpleEngine.popen_uci("engine\stockfish.exe")
        for j in range(sf_depth):
            # if (move+2)//2 != movehighlight:
            #     levels = [0]*sf_depth
            #     break
            # Analyse the position up to depth j
            # resultsf = sf.analyse(played, chess.engine.Limit(depth=j), multipv=10)
            # pos_eval = resultsf[0]['score'].white().score(mate_score=10000)
            # score = 0
            # # Let score = the evaluation of the move played
            # for i in range(len(resultsf)):
            #     score = resultsf[i]['score'].white().score(mate_score=10000)
            #     if resultsf[i]['pv'][0].uci() == next_move.uci():
            #         break
            # Record this evaluation
            resultsf = sf.analyse(played, chess.engine.Limit(depth=j))
            score = resultsf['score'].white().score(mate_score=10000)
            pos_eval = score
            levels.append(score)

        # Get the evaluation of the current position
        
        # if (move+2)//2 == movehighlight:
        res = sf.analyse(board, chess.engine.Limit(depth=sf_depth))
        # sf.quit()
        best_eval = res['score'].white().score(mate_score=10000)
        # else:
        #     best_eval = 0

        # Take the difference from the best move
        levels = [l - best_eval for l in levels]
        if move%2:
            levels = [l*-1 for l in levels]

        # Increment move counter
        game = game.next()
        move += 1

        mate_in, imbalance = mate_complexity(sf, sf_depth, game.board())

        # Check if the move is a book move
        if inbook and book.get(board=game.next().board()) is None:
            inbook = False

        # We don't analyse book moves, these are common and well-known lines
        if clock is not None and not inbook:
            filtered = filtfilt([1/3 for _ in range(3)], 3, levels)

            # Normalise the data
            max_magf = abs(max(filtered)) if abs(max(filtered)) > abs(min(filtered)) else abs(min(filtered))
            if max_magf:
                filtered = [l/max_magf for l in filtered]

            # Normalise the data
            max_mag = abs(max(levels)) if abs(max(levels)) > abs(min(levels)) else abs(min(levels))
            if max_mag:
                levels = [l/max_mag for l in levels]

            # plt.clf()
            # plt.plot(filtered, "b-")
            # plt.plot(levels, "r--")
            # plt.grid(True)
            # ax = plt.gca()
            # ax.set_ylim([-1,1])
            # ax.set_xlim([0,sf_depth-1])
            # plt.show()
            # shape = int(input("Shape: "))
            # plt.close()
            
            shape = 0

            results.append(MoveAnalysis(
                (move+1)//2,next_move.uci(),move%2, pos_eval,clock,filtered,max_magf,inbook,shape=shape,
                mate_in=mate_in, imbalance=imbalance
            ))
            print(results[-1])

            # if ((move+1)//2) == movehighlight:
            # ax = fig.add_subplot(12, 10, move)
            # ax.plot(filtered, "b-")
            # ax.plot(levels, "r--")
            # ax.grid(True)
            # ax.set_ylim([-1,1])
            # ax.set_xlim([0,sf_depth-1])
            # ax.set_title(f"{(move+1)//2}{'w' if move%2 else 'b'}")
        # else:
        #     results.append(f"[ {next_move.uci()} * book move * ]")
    

    data = np.empty((len(results), sf_depth+1))
    for r,result in enumerate(results):
        data[r] = result.data()

    np.savetxt(f"{game_name}.csv", data, delimiter=",", fmt=r"%3.2f")

    # plt.ylabel('Evaluation')
    # plt.xlabel('Depth')
    # plt.gca().set_aspect('equal')
    # plt.gca().set_ylim([-1,1])
    # plt.grid(True)
    # ax = plt.gca()
    # ax.set_xlim([0, sf_depth-2])
    # ax.set_ylim([-1, 1])
    # plt.title("Torch - Stockfish, move 33: a2f2")
    # plt.show()
    
    # for r in results:
    #     print(r)

    # print(f"White accuracy: {sum(acc)/len(acc)*100:.2f}%")
    # print(f"Black accuracy: {sum(bacc)/len(bacc)*100:.2f}%")
    # Need to shutdown the engine
    sf.quit()

if __name__ == "__main__":
    main()