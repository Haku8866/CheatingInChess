import chess.pgn
import chess.engine

# Start stockfish
sf = chess.engine.SimpleEngine.popen_uci("engine\stockfish.exe")

# Load a game
db = open("db\sample.pgn")
game = chess.pgn.read_game(db)

# Evaluate each position in the game
while game.next() is not None:
    board = game.board()
    resultsf = sf.analyse(board, chess.engine.Limit(depth=5))
    print(f"Evaluation is {resultsf['score']}")
    game = game.next()

# Need to shutdown the engine
sf.quit()

