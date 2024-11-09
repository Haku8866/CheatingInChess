import game_analyser
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics as m

from main_model import MainModel

num_games = 138

# Yellow pixels are opening moves (or to fill empty space on the right)
# Yellow pixels are opening moves (or to fill empty space on the right)
# Purple pixels show the middlegame where the evaluation is not decisive
# Orange pixels show the endgame where one player is clearly winning
# Dark blue pixels show the flagged engine moves in the game

# The histograms below show the total engine moves found on each move across all games
# Blue -> white
# Red -> black

def main():
    model = MainModel()
    arr = []
    hist = []
    mvs = 0

    # Other banned players, we don't want to involve them, we want cheater vs legitimate player
    cheaters = ["BedirhanDin","HeroSLitu","Camel-75","777dimonchik","alexandru2007","MartinGiri","f1eetss"]

    for x,game in enumerate([ # 63
        # f"optiver/sample_game{i}" for i in range(1,num_games+1)]+[
        f"ha312/ha312_game{i}" for i in range(1, 29)]+[
        f"vartender/vartender_game{i}" for i in range(1, 34)]+[
        # f"wc_db/2016_wcc/2016_wcc_game{i}" for i in range(1, 13)]+[
        # "cc_db/Stockfish_vs_Lc0_2023.10.26", # 60/81
        # "cc_db/Torch_vs_Stockfish_2024.01.01", # 51/68
        # "cc_db/Stockfish_vs_Torch_2024.01.13", # 64/106
        # "cc_db/Stockfish_vs_Torch_2024.01.15_2",
        # "demo"
        # "master_2",
        # "master_3",
        # "master_4"
        # "wc_db/2013_wcc/2013_wcc_game1",
        # "wc_db/2013_wcc/2013_wcc_game2",
        # "wc_db/2013_wcc/2013_wcc_game3",
        # "wc_db/2013_wcc/2013_wcc_game4",
        # "wc_db/2013_wcc/2013_wcc_game5",
        # "wc_db/2013_wcc/2013_wcc_game6",
        # "wc_db/2013_wcc/2013_wcc_game7",
        # "wc_db/2013_wcc/2013_wcc_game8",
        # "wc_db/2013_wcc/2013_wcc_game9",
        # "wc_db/2013_wcc/2013_wcc_game10",
    ]):
        gamefile = f"db/{game}"
        try:
            with open(gamefile + ".pkl", "rb") as f:
                game = pickle.load(f)
        except FileNotFoundError:
            analyser = game_analyser.GameAnalyser()
            analyser.analyse(gamefile)
            game = analyser.analysed_games[gamefile]
            with open(gamefile + ".pkl", "wb") as f:
                pickle.dump(game, f, pickle.HIGHEST_PROTOCOL)
            analyser.quit()

        if game.pgn.headers["White"] in cheaters or game.pgn.headers["Black"] in cheaters:
            continue

        skip_to = (game.moves[0].move_num)*2-1
        if (not game.moves[0].white):
            skip_to += 1

        # Colour each pixel based on the type of move        
        sequence = [4 for _ in range(skip_to)]

        for i,move in enumerate(game.moves):
            good = 1 if move.white else -1
            if abs(move.best_eval) < 300 and (move.eval_drop*good <= -1 or (move.move == move.best_move)) and model.classify(move) > 0.5:
                sequence.append(0)
                hist.append(i+skip_to)
            else:
                if abs(move.best_eval) < 300:
                    mvs += 1
                    if move.eval_drop*good <= 0:
                        sequence.append(2)
                    else:
                        sequence.append(2)
                else:
                    sequence.append(3)
        
        arr.append(sequence)
        m = max(len(s) for s in arr)
        for s in arr:
            while len(s) < m:
                s.append(4)

    arr = np.array(arr)
    # counts = [sum(r==0)/sum((r==0) | (r==2)) for r in arr]
    # for c in counts:
    #     print(c)
    # counts = np.argsort(counts)
    # arr = arr[counts]

    hist_white = []
    hist_black = []
    for h in hist:
        if h%2:
            hist_black.append(h)
        else:
            hist_white.append(h)
    
    print(len(hist_white)/num_games)
    print(len(hist_black)/num_games)
    
    print(f"Average engine moves: {(len(hist_white)+len(hist_black))/mvs*100:.2f}%")

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_title("Games Visualised")
    ax.imshow(arr, interpolation='nearest', cmap='plasma')
    ax = fig.add_subplot(212)
    ax.hist(np.append(hist_white, hist_black), range=[0, m], bins=m//2, color="g", alpha=0.5)
    ax.set_xlabel("Move")
    ax.set_ylabel("Count")
    # ax.hist(hist_white, range=[0, m], bins=m, color="b", alpha=0.5)
    # ax.hist(hist_black, range=[0, m], bins=m, color="r", alpha=0.5)
    ax.set_xlim(0, m)
    plt.show()

if __name__ == '__main__':
    main()