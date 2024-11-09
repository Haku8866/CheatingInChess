import game_analyser
import pickle
import matplotlib.pyplot as plt
import numpy as np
from main_model import MainModel

num_games = 128

def main():
    model = MainModel()
    arr = []
    hist = []
    mvs = 0

    # Other banned players, we don't want to involve them, we want cheater vs legitimate player
    cheaters = ["BedirhanDin","HeroSLitu","Camel-75","777dimonchik","alexandru2007","MartinGiri","f1eetss"]

    # Across 29 games ha312 has 100% more engine moves flagged than opponents
    # ha312 did not appear to cheat in every single game, this was intermittent cheating
    # For tournament games, 21.95% were engine moves across all games

    cheater_moves = 0
    cheater_opp_moves = 0

    prs = []
    prs2 = []

    for x,game in enumerate([ # 63
        f"optiver/sample_game{i}" for i in range(1,num_games+1)]+[
        # f"ha312/ha312_game{i}" for i in range(1, 29)]+[
        # f"vartender/vartender_game{i}" for i in range(1, 34)]+[
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

        # Drop cheater-cheater games
        if game.pgn.headers["White"] in cheaters or game.pgn.headers["Black"] in cheaters:
            continue
        
        cheater_name = "Ha312"
        cheater_colour = 0
        if (game.pgn.headers["White"] == cheater_name):
            cheater_colour = 1
        elif (game.pgn.headers["Black"] == cheater_name):
            cheater_colour = -1

        # Log positive rates for players
        pr, prw, prb = game.annotatePGN(model)
        if cheater_colour == 0:
            prs.append(pr)
        elif cheater_colour == 1:
            prs.append(pr)
            prs2.append(prw)
        elif cheater_colour == -1:
            prs.append(pr)
            prs2.append(prb)


        # Used for visualisation of games
        skip_to = (game.moves[0].move_num)*2-1
        if (not game.moves[0].white):
            skip_to += 1
        
        sequence = [4 for _ in range(skip_to)]
        for i,move in enumerate(game.moves):
            good = 1 if move.white else -1
            if abs(move.best_eval) < 300 and (move.eval_drop*good <= -1 or (move.move == move.best_move)) and model.classify(move) > 0.5:
                sequence.append(0)
                hist.append(i+skip_to)
                if cheater_colour * good == 1:
                    cheater_moves += 1
                elif cheater_colour != 0:
                    cheater_opp_moves += 1
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

    # fig = plt.figure()
    # fig.add_subplot(211).imshow(arr, interpolation='nearest', cmap='plasma')
    # ax = fig.add_subplot(212)
    # ax.hist(np.append(hist_white, hist_black), range=[0, m], bins=m, color="g", alpha=0.5)
    # ax.hist(hist_white, range=[0, m], bins=m, color="b", alpha=0.5)
    # ax.hist(hist_black, range=[0, m], bins=m, color="r", alpha=0.5)
    # ax.set_xlim(0, m)
    # plt.title("Games with cheater")
    # plt.show()

    prs = np.array(prs)
    prs2 = np.array(prs2)

    plt.title("Positive rate comparison")
    plt.hist(prs, bins=20, range=[0,100], color="b", alpha=0.5, label="Collective Opponents")
    plt.hist(prs2, bins=20, range=[0,100], color="r", alpha=0.5, label="Cheater")
    plt.axvline(x=np.mean(prs), color="b", label="Opponents Mean")
    plt.axvline(x=np.mean(prs2), color="r", label="Cheater Mean")
    plt.ylabel("Count")
    plt.xlabel("PR")
    plt.xlim(0, 100)
    plt.xticks([i*10 for i in range(0,11)])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()