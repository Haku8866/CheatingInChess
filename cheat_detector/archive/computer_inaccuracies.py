import game_analyser
import pickle
from colorama import Fore, init
import numpy as np
import matplotlib.pyplot as plt

'''
This file was used to look for patterns in the inaccuracies and brilliancies in engine games.
It also plotted the amplitude, highest evaluation and lowest evaluation for each move, as well
as measuring how noisy the move was.
'''

init(autoreset=True)

def main():
    computer_moves = []
    other_shapes = [0,0,0]
    comp_shapes = [0,0,0]

    mvlist = []

    for game in [
        "cc_db/sf_torch",
        # "cc_db/dragon_berserk",
        # "cc_db/sf_torch_blitz",
        # "cc_db/sf_torch_blitz2",
        # "cc_db/sf_torch_blitz3",
        # "me_cheater",
        # "cc_db/sf_torch_2",
        # "cc_db/sf_weiss",
        # "cc_db/sf_lc0",
        # "cc_db/torch_lc0",
        # "wc_db/game2",
        # "wc_db/game7",
        # "wc_db/2023game2",
        # "me_random",
        # "wc_db/carlsen_caruana_2018",
        # "me_ege",
        # "me_random2",
        # "me_random3",
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
        
        amps = []
        bamps = []
        noise = []
        pc = []

        plt.ylim(-20, 20)

        # Gather data for move
        for move in game.moves:
            off = (0 if move.white else 0.5)
            amps.append([move.move_num+off, move.level_amplitude*max(move.levels)])
            amps.append([move.move_num+off, move.level_amplitude*min(move.levels)])
            noise.append([move.move_num+off, move.noise()])
            pc.append([move.move_num+off, move.partial_credit*10])
            bamps.append([move.move_num+off, move.best_amp*max(move.best_levels)])
            bamps.append([move.move_num+off, move.best_amp*min(move.best_levels)])
        
        amps = np.array(amps)
        bamps = np.array(bamps)
        noise = np.array(noise)
        pc = np.array(pc)

        # Plot data
        plt.plot(amps[:,0], amps[:,1], "rx")
        plt.plot(bamps[:,0], bamps[:,1], "gx")
        plt.plot(noise[:,0], noise[:,1], "b.")
        plt.plot(pc[:,0], pc[:,1], "r.")

        # Log good and bad moves
        for move in game.moves:
            if move.move == move.best_move or abs(move.best_eval) > 400:
                continue
            if move.move_num > 11 and move.mate_in is None:
                computer_moves.append((move.levels, move.eval_drop, move.level_amplitude))
                good = -1 if move.white else 1
                if (move.eval_drop)*good > 10:
                    print(f"{Fore.GREEN}Brilliancy from {gamefile}:\n{move}")
                    print(f"{Fore.GREEN}Best:\n{move.best_move}")
                    print(f"{Fore.GREEN}Best:\n{move.best_levels} {move.best_amp}")
                    print(f"{move.metrics()}")
                    computer_moves.append((move.levels, move.eval_drop, move.level_amplitude))
                    p = move.move_num + (0 if move.white else 0.5)
                    plt.axvline(p, c="c")
                    for x,prob in enumerate(move.shape_proba):
                        if prob > 0.5 and move.level_amplitude > 2:
                            comp_shapes[x] += 1
                elif (move.eval_drop)*-good > 30:
                    print(f"{Fore.YELLOW}Inaccuracy from {gamefile}:\n{move}")
                    print(f"{Fore.YELLOW}Best:\n{move.best_move}")
                    print(f"{Fore.YELLOW}Best:{move.best_levels} {move.best_amp}")
                    p = move.move_num + (0 if move.white else 0.5)
                    computer_moves.append((move.best_levels, 0, move.best_amp))
                    plt.axvline(p, c="y")
    plt.show()
    print(len(computer_moves))


if __name__ == "__main__":
    main()