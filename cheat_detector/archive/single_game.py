from sys import argv
import game_analyser
import pickle
from colorama import init

'''
Experimental method to analyse games, testing if SVC yields any interesting results
'''

def main():
    init(autoreset=True)
    # gamefile = argv[1]

    gamefiles = [
        "db/demo",
        # "db/cc_db/sf_torch",
        # "db/me_cheater",
        # "db/wc_db/game2",
        # "db/wc_db/game3",
        # "db/wc_db/game7",
        # "db/wc_db/game11",
        # "db/wc_db/2014_game1",
        # "db/wc_db/2014_game2",
        # "db/wc_db/2014_game3",
        # "db/wc_db/2014_game4",
        # "db/wc_db/2014_game6",
        # "db/wc_db/2014_game11",
        # "db/wc_db/carlsen_caruana_2018",
        # "db/wc_db/2023game2",
    ]

    for gamefile in gamefiles:
        print(f"Game: {gamefile}")
        try:
            show = int(argv[2])
            if show > 0:
                show = True
        except:
            show = False
        
        try:
            with open(gamefile + ".pkl", "rb") as f:
                game = pickle.load(f)
        except FileNotFoundError:
            analyser = game_analyser.GameAnalyser()
            analyser.analyse(gamefile, show=show)
            game = analyser.analysed_games[gamefile]
            with open(gamefile + ".pkl", "wb") as f:
                pickle.dump(game, f, pickle.HIGHEST_PROTOCOL)
            analyser.quit()

    game.show()

    counts = [0,0,0]

    print(f"{' ' + gamefile + ' report ':=^70}")

    for move in game.moves:
        if move.mate_in is not None:
            continue

        counts[int(move.shape)] += 1

        low = abs(move.best_eval) - move.level_amplitude

        if move.shape == 0 \
        and move.level_amplitude > 5 \
        and move.partial_credit > 0.95:
            print(f"{move.move_num:>2}{'w' if move.white else 'b'} is suspicious")
            print(move)

    print(f"{' end report ':=^70}")
    
if __name__ == "__main__":
    main()