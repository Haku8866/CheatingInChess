import game_analyser
import pickle
import matplotlib.pyplot as plt
import numpy as np
from main_model import MainModel

'''
Used to build the test and training data sets for the main machine learning model.
'''

def main(args):

    missed_moves = []
    played_moves = []

    for game in [
        "cc_db/sf_torch_2",
        "cc_db/sf_torch_blitz",
        "cc_db/sf_torch_blitz2",
        "cc_db/sf_torch_blitz3",
        "cc_db/sf_lc0",
        "cc_db/sf_weiss",
        "cc_db/torch_lc0",
        "cc_db/dragon_berserk",
        "cc_db/igel_berserk",
        "cc_db/torch_akimbo",
        "cc_db/torch_midnight",
        "cc_db/midnight_torch",
        "cc_db/Stockfish_vs_Torch_2024.01.15",
        "cc_db/Torch_vs_Stockfish_2024.01.17",
        "cc_db/Torch_vs_Stockfish_2024.01.13",
        "cc_db/Dragon_vs_Stockfish_2024.02.14",
    ]:
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
        
        # Ignore opening moves
        for move in game.moves[20:]:
            # Must be under 300 eval
            if abs(move.best_eval) < args[1]:
                good = 1 if move.white else -1
                # Must be as good as best move
                if move.eval_drop*good <= args[0]:
                    missed_moves.append(move.metrics())
    
    for game in [
        "wc_db/2014_game1",
        "wc_db/2014_game2",
        "wc_db/2014_game3",
        "wc_db/2014_game4",
        "master_1",
        "wc_db/2014_game6",
        "wc_db/2014_game11",
        "wc_db/carlsen_caruana_2018",
        "wc_db/game2",
        "wc_db/game3",
        "wc_db/game7",
        "wc_db/game11",
        "wc_db/2023game2",
        "wc_db/2013_wcc/2013_wcc_game1",
        "wc_db/2013_wcc/2013_wcc_game2",
        "wc_db/2013_wcc/2013_wcc_game3",
        "wc_db/2013_wcc/2013_wcc_game4",
        "wc_db/2013_wcc/2013_wcc_game5",
        "wc_db/2013_wcc/2013_wcc_game6",
        "wc_db/2013_wcc/2013_wcc_game7",
        "wc_db/2013_wcc/2013_wcc_game8",
        "wc_db/2013_wcc/2013_wcc_game9",
        "wc_db/2013_wcc/2013_wcc_game10",
        # # bad quality human games too
        "me_random",
        "me_ege",
        "me_random2",
        "me_random3",
    ]:
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
        
        for move in game.moves:
            # Same as previous code
            if abs(move.best_eval) < args[2]:
                good = 1 if move.white else -1
                if (move.eval_drop*good <= args[0]):
                    played_moves.append(move.metrics())

    print(len(missed_moves))
    print(len(played_moves))

    X = np.array(played_moves + missed_moves)
    Y = np.array([0]*len(played_moves) + [1]*len(missed_moves))

    db = np.column_stack((Y, X))

    # Save dataset
    np.savetxt(f"prototype/training_data/main_training.csv", db, delimiter=",")

    # Now do a quick test of the model on human and engine games
    model = MainModel()
    cthresh = 0.5

    fp = 0
    tn = 0
    total = 0
    ttotal = 0
    fps = [0, 0]
    hm = 0
    for game in [
        "master_2", # 14/43
        "master_3", # 8/100
        "master_4", # 12/42
        # "wc_db/2023game2",
        # "me_random",
        # "me_ege",
        # "me_random2",
        # "me_random3",
        # "wc_db/2014_game1",
        # "wc_db/2014_game2",
        # "wc_db/2014_game3",
        # "wc_db/2014_game4",
        # "naroditsky_cheater",
        # "demo",
        # "hikaru_sf",
        # "lc0_yaac",
        # "carlsen_caruana_2019",
        # "hikaru_komodo",
        # "hikaru_weak_komodo",
        # "hutasoit/david-hutasoit_vs_mat-sav_2024.02.26",   #    7 to 4 / 27
        # "hutasoit/david-hutasoit_vs_ortznoi57_2024.02.25", #    0 to 4 / 20
        # "hutasoit/Olbu61_vs_david-hutasoit_2024.02.26",    #      0 to 10 / 21
    ]:
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

        # Use model to classify critical moves
        for move in game.moves:
            ttotal += 1
            good = 1 if move.white else -1
            if abs(move.best_eval) < 300:
                if (move.eval_drop*good <= -1) or (move.move == move.best_move):
                    if model.classify(move) > cthresh:
                        fp += 1
                        fps[(good + 1)//2] += 1
                        # print(move.move_num)
                    else:
                        tn += 1
                total += 1
    print(f"Human games score:  {fp}/{total} fps from {tn+fp} excellent moves. ({fp/total:.4f}) ({fp/(tn+fp):.4f})")
    print(f"{fps[1]} from white, {fps[0]} from black")

    human = fp/(tn+fp)

    fp = 0
    tn = 0
    total = 0
    ttotal = 0
    fps = [0, 0]
    hm = 0

    # Repeat the same process for engine games
    for game in [
        "cc_db/Stockfish_vs_Lc0_2023.10.26", # 60/81
        "cc_db/Torch_vs_Stockfish_2024.01.01", # 51/68
        "cc_db/Stockfish_vs_Torch_2024.01.13", # 64/106
        # "cc_db/Stockfish_vs_Torch_2024.01.15_2",
        # "cc_db/dragon_berserk",
        # "cc_db/igel_berserk",
        # "cc_db/torch_midnight",
        # "cc_db/sf_torch",
        # "cc_db/Torch_vs_Stockfish_2024.01.13",
        # "cc_db/Dragon_vs_Torch_2024.02.14",
    ]:
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
        
        for move in game.moves:
            ttotal += 1 
            good = 1 if move.white else -1
            if abs(move.best_eval) < 300:
                total += 1
                if (move.eval_drop*good <= -1) or (move.move == move.best_move):
                # if (move.move == move.best_move):
                    if model.classify(move) > cthresh:
                        fp += 1
                        fps[(good + 1)//2] += 1
                    else:
                        tn += 1
    
    print(f"Engine games score: {fp}/{total} fps from {tn+fp} excellent moves. ({fp/total:.4f}) ({fp/(tn+fp):.4f})")
    print(f"{fps[1]} from white, {fps[0]} from black")

# 300 means abs(eval) should be under 300. -1 means the move should be as good as or better than the
# best move when collecting the data.
if __name__ == '__main__':
    main([-1, 300, 300, -1])