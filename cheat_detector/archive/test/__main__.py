import chess.pgn
import chess.engine
import chess.polyglot

from math import exp

depth_limit = 5

'''
Features of each move:
    - Evaluation drop between move played and best move
    - Difference in evaluation between lc0 and sf
    - Time spent on the move ( Was it a reflexive move or well thought out? )
    - Ken Regan's partial credit metric

    [https://www.chess.com/computer-chess-championship]

From watching computer games:
    * Was a piece left hanging? -> Humans would not do this
    * Was a free piece available but not taken? -> Humans would take
    * Are there many critical moves ahead in this line? -> Humans would get scared

Metrics of an entire game:
    * How consistent is clock usage? -> Inconsistent timing shows actual thought
    * How one-sided was the game? -> Did one player make 0 mistakes?
    * Overall "accuracy" of a player in relation to average accuracy of games of that ELO
'''

class MoveAnalysis:
    def __init__(self, move, eval_drop, dispute, options, time_spent, 
                 line_length, partial_credit, book=False):
        
        self.move = move
        self.eval_drop = eval_drop # Raw accuracy
        self.dispute = dispute
        self.options = options

        self.time_spent = time_spent
        self.line_length = line_length # 1 = the only good move

        self.partial_credit = partial_credit # Scaled accuracy

        self.book = book # Opening move?
    
    def __str__(self) -> str:
        features = [self.eval_drop,
                    self.dispute,
                    self.options,
                    self.time_spent,
                    self.line_length,
                    self.partial_credit]
        out = f"[ {self.move}"
        for f in features:
            out += f"{f:>10.2f} "
        out += "]" if not self.book else "(book) ]"
        return out

def score(result):
    return result['score'].white().score(mate_score=10000)

def getLength(sf, board):
    result = sf.analyse(board, chess.engine.Limit(depth=depth_limit), multipv=2)
    diff = score(result[0]) - score(result[1])
    # Check if the difference between the best and next best moves is large
    if abs(diff) > 100:
        board.push(result[0]['pv'][0])
        val = 1 + getLength(sf, board)
        board.pop()
        return val
    else:
        return 0

def main():
    # Start stockfish
    sf = chess.engine.SimpleEngine.popen_uci("engine\stockfish.exe")
    # Start lc0
    lc0 = chess.engine.SimpleEngine.popen_uci("engine\lc0.exe")

    # Open the openings book
    book = chess.polyglot.open_reader(r"openings\baron30.bin")

    db = open("db\sample.pgn")
    game = chess.pgn.read_game(db)

    move = 0
    results = []
    inbook = True

    
    acc = []
    bacc = []
    while game.next() is not None:
        board = game.board()
        clock = game.next().clock()

        resultsf = sf.analyse(board, chess.engine.Limit(depth=depth_limit), multipv=99)
        resultlc = lc0.analyse(board, chess.engine.Limit(depth=depth_limit), multipv=99)

        next_move = game.next().move

        pos_eval = resultsf[0]['score'].white().score(mate_score=10000)
        dispute = abs(pos_eval - resultlc[0]['score'].white().score(mate_score=10000))

        line_length = 0
        options = 0

        # Each i is a suggested move for the position
        for i in range(len(resultsf)):
            # The evaluation (score)
            score = resultsf[i]['score'].white().score(mate_score=10000)
            if resultsf[i]['pv'][0].uci() == next_move.uci():
                # If the move was actually played, calculate metrics
                d = abs(pos_eval - score)
                eval_drop = abs(pos_eval - score)
                line_length = getLength(sf, board)
        # Partial credit formula
        s = 100
        c = 2
        partial_credit = exp(-(d/s)**c)

        game = game.next()
        move += 1

        if inbook and book.get(board=game.next().board()) is None:
            inbook = False

        if clock is not None:
            results.append(MoveAnalysis(
                next_move.uci(),eval_drop,dispute,options,clock,line_length,partial_credit,inbook
            ))
            acc.append(partial_credit) if move % 2 else bacc.append(partial_credit)
        else:
            results.append(f"[ {next_move.uci()} * book move * ]")
    
    for r in results:
        print(r)

    print(f"White accuracy: {sum(acc)/len(acc)*100:.2f}%")
    print(f"Black accuracy: {sum(bacc)/len(bacc)*100:.2f}%")

    # Need to shutdown the engines
    sf.quit()
    lc0.quit()

if __name__ == "__main__":
    main()