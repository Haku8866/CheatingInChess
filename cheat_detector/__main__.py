import sys
import os
from argparse import ArgumentParser
import pickle

# Some pickle files can be very large nested objects
sys.setrecursionlimit(100000)

def main():
    # Parse command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-g", "--game", dest="gamefile", help="Provide the name of a PGN file to analyse")
    arg_parser.add_argument("-i", "--input", dest="gamelist", help="Provide a file containing a list of PGNs to analyse")
    arg_parser.add_argument("-t", "--target", dest="target", help="The name of a player to investigate specifically")
    arg_parser.add_argument("-s", "--show", dest="show", action="store_true", help="Show an animated display")
    arg_parser.add_argument("-f", "--force", dest="force", action="store_true", help="Re-compute and overwrite any existing cached results")
    arg_parser.add_argument("-o", "--optimised", dest="fast", action="store_true", help="Run a faster and reduced analysis")    
    args_list = arg_parser.parse_args()

    # Showing the graphical display with the animation requires cairosvg, which can't be installed with
    # pip, the dll needs to be used from GTK-3 so we have try/except blocks to see if it's installed
    if args_list.show:
        try:
            from cairosvg import svg2png
        except:
            try:
                # The user specifies their cairo dll path in this file
                with open("cairo_path.txt", "r") as f:
                    os.add_dll_directory(f.readline().strip())
                from cairosvg import svg2png
            except:
                print("Graphics not available! Please supply path of cairo DLL. (See README)")
                quit()

    from game_analyser import GameAnalyser
    from main_model import MainModel
    from plot_suspect import inspect_target
    
    # Enter the cheat_detector directory
    os.chdir("cheat_detector")

    # Initialise analyser and model
    model = MainModel(fast=args_list.fast)
    if args_list.show:
        analyser = GameAnalyser(model, fast=args_list.fast, svg2png=svg2png)
    else:
        analyser = GameAnalyser(model, fast=args_list.fast)

    os.chdir("..")
    # Collect the filenames of all PGNs to analyse
    filenames = []
    if args_list.gamelist is not None:
            with open(args_list.gamelist, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() != "":
                        filenames.append(line.strip())
    if args_list.gamefile is not None:
        filenames.append(args_list.gamefile)

    # Once filenames are all collected, scroll through and analyse any non-cached ones
    for gamefile in filenames:
        notfound = False
        if not args_list.force:
            try:
                with open(gamefile + ".pkl", "rb") as f:
                    game = pickle.load(f)
            except FileNotFoundError:
                notfound = True
        
        # Analyse any unseen games OR all games if --force is set
        if notfound or args_list.force:
            analyser.analyse(gamefile, show=args_list.show)
            game = analyser.analysed_games[gamefile]
            with open(gamefile + ".pkl", "wb") as f:
                pickle.dump(game, f, pickle.HIGHEST_PROTOCOL)

        # Annotate the PGN object in memory, then write it to a new file
        game.annotatePGN(model)
        with open(f"{gamefile}_annotated.pgn", "w") as f:
            f.write(str(game.pgn))

    # Shutdown Stockfish
    analyser.quit()
    
    # If there were multiple games given, we can plot a histogram of the average positive rates
    # and highlight the postive rates of the given target compared to them
    if len(filenames) > 1:
        inspect_target(filenames, args_list.target, model)

if __name__ == "__main__":
    main()