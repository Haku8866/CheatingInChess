import pickle
import matplotlib.pyplot as plt
import numpy as np
from main_model import MainModel

# Takes a set of games, and a target name, and compares PR distributions between the target and the
# rest of the cohort. This is plotted as a histogram.
def inspect_target(gamefiles, cheater_name, model):
    prs = [] # "positive rates"
    prs_target = [] # "target's positive rates"

    for gamefile in gamefiles:
        with open(gamefile + ".pkl", "rb") as f:
            game = pickle.load(f)

        cheater_colour = 0
        # Note which colour the target is, if any
        if (game.pgn.headers["White"] == cheater_name):
            cheater_colour = 1
        elif (game.pgn.headers["Black"] == cheater_name):
            cheater_colour = -1

        # Separate the positive rates of each colour, if one is the cheater
        pr, prw, prb = game.annotatePGN(model)
        if cheater_colour == 0:
            prs.append(pr)
        elif cheater_colour == 1:
            prs.append(prb)
            prs_target.append(prw)
        elif cheater_colour == -1:
            prs.append(prw)
            prs_target.append(prb)

    # Convert to numpy arrays and plot the data
    prs = np.array(prs)
    prs_target = np.array(prs_target)

    plt.title("Positive Rate Comparison")
    plt.hist(prs, bins=20, range=[0,100], color="b", alpha=0.5, label="Collective Opponents")
    # Plot cheater histograms if a name was specified
    if cheater_name is not None:
        plt.hist(prs_target, bins=20, range=[0,100], color="r", alpha=0.5, label=f"{cheater_name}")
    plt.axvline(x=np.mean(prs), color="b", label="Opponents Mean")
    if cheater_name is not None:
        plt.axvline(x=np.mean(prs_target), color="r", label=f"{cheater_name} Mean")
    plt.ylabel("Count")
    plt.xlabel("PR")
    plt.xlim(0, 100)
    plt.xticks([i*10 for i in range(0,11)])
    plt.legend()
    plt.show()
