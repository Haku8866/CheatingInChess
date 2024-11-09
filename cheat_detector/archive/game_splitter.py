import chess
import chess.pgn
from sys import argv

# Some PGN files have many games one after the other, this script splits them into one PGN file
# per game, and filters out any which are too short to be properly analysed (under 30 moves)
def main():
    file = "db/" + argv[1]
    db = open(f"{file}.pgn")
    game = chess.pgn.read_game(db)
    i=1
    while game is not None:
        move = game
        while move.next() is not None:
            move = move.next()
        if move.ply() >= 60:
            with open(f"{file}_game{i}.pgn", "w") as f:
                f.write(str(game))
            i += 1
        game = chess.pgn.read_game(db)

if __name__ == '__main__':
    main()