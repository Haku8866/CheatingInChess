import chess
import chess.pgn
import chess.svg

import numpy as np
import chess.engine
from math import exp
import chess.polyglot
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

from multi_depth_analysis import EvalSequenceAnalyser
from data_structures.move_analysis import MoveAnalysis
from data_structures.game_analysis import GameAnalysis
from analysis_methods import *

from main_model import MainModel
from PIL import Image
from io import BytesIO

class GameAnalyser():
    def __init__(self, model:MainModel, fast:bool, svg2png=None):
        # Initialise Stockfish once upon creating an instance of the class
        self.analysed_games = {}
        self.model = EvalSequenceAnalyser()
        self.main_model = model
        self.sf = chess.engine.SimpleEngine.popen_uci(r"engine\stockfish.exe")
        # Enable multithreading for efficiency
        self.sf.configure({"Threads": multiprocessing.cpu_count() // 2})
        self.previous = None
        self.svg2png = svg2png
        self.sf_depth = 20 if fast else 30
        self.fast = fast

        # Open the openings book
        book_list = [
            r'openings\baron30.bin',
            r'openings\Book.bin',
            r'openings\codekiddy.bin',
            r'openings\DCbook_large.bin',
            r'openings\Elo2400.bin',
            r'openings\final-book.bin',
            r'openings\gm2600.bin',
            r'openings\komodo.bin',
            r'openings\KomodoVariety.bin',
            r'openings\Performance.bin',
            r'openings\varied.bin',
        ]
        self.books = [chess.polyglot.open_reader(book) for book in book_list]

    def quit(self):
        # Shutdown Stockfish
        self.sf.quit()

    # Animated display, updates the data plots
    def update_graph(self, lines:list[list[float]], styles:list[str], ylims:list[float]):
        ax = self.ax
        ax.set_autoscalex_on(False)
        ax.set_xticks = np.array([5, 10, 15])
        ax.cla()
        ax.set_xlim(0, self.sf_depth-1)
        ax.set_ylim(*ylims)
        ax.grid(True)

        ax.set_title("Multi-depth Analysis")
        ax.set_ylabel("Evaluation")

        # Plot all lines on the graph
        for i in range(len(lines)):
            ax.plot(lines[i], styles[i])

        # Blit (or update) just the box containing the axes not the whole screen
        self.fig.canvas.blit(ax.bbox)
        self.fig.canvas.flush_events()

    # Analyses the next best few moves
    def multi_analysis(self, played:chess.Board, multipv:int):
        # Briefly analyse the next <multipv> moves, recording the evaluations
        results = self.sf.analyse(played, limit=chess.engine.Limit(time=0.25), multipv=multipv)
        scores = [score(r) for r in results]

        # If there weren't enough moves available, pad with zeroes
        while len(scores) < multipv:
            scores.append(0)
        return np.array(scores)

    # Multi depth analysis function, generates an evaluation levels vector 'raw_levels'
    def iterative_analysis(self, played:chess.Board, move:int, best_levels:list[float]=None,
                           show:bool=False, played_move:chess.Move=None):
        # Stop if the game is drawn or won/lost
        if played.is_game_over():
            return np.ones(self.sf_depth), 0, played, 0
        
        raw_levels = np.zeros(self.sf_depth)
        pos_eval = 0
        pos_move = None
        prev_depth = 0
        changes = -1
        results = self.sf.analysis(played, limit=chess.engine.Limit(time=10))
        
        # Each 'r' is an 'info' line in the UCI output
        for r in results:
            changed = False
            depth = r.get('seldepth')
            # Some entries can be null
            if depth is not None:
                # Cap the seldepth at our defined seldepth limit
                if depth > self.sf_depth:
                    depth = self.sf_depth
            
                pos_eval = score(r)
                if depth > prev_depth:
                    # If the best move changed, increment the change counter
                    if r['pv'][0] != pos_move:
                        changes += 1
                        changed = True
                    pos_move = r['pv'][0]

                    # We may have skipped a depth, which is fine, we fill in the gaps
                    raw_levels[prev_depth:depth] = pos_eval
                    prev_depth = depth
                else:
                    # We may get the value for a skipped depth later
                    raw_levels[depth-1] = pos_eval

                # For animated display
                if show:
                    # Plot previous levels vector, where biggest is use to define the axis limits
                    biggest = np.amax(abs(raw_levels)) + 50
                    good = -1 if move % 2 else 1
                    plot = raw_levels[:max(depth, prev_depth)] * good

                    # If we have data about the best move, plot it too
                    if best_levels is not None:
                        biggest = max(biggest, np.amax(abs(best_levels)) + 50)
                        lastmove = played_move
                    else:
                        lastmove = None
                    
                    # We only update the chess board display if the best move changed
                    if changed:
                        # Create chess board image
                        green_arrow = chess.svg.Arrow(pos_move.from_square, pos_move.to_square,
                                                      color="green")
                        boardsvg = chess.svg.board(played, arrows=[green_arrow], lastmove=lastmove)
                        boardpng = self.svg2png(bytestring=boardsvg)
                        boardimage = Image.open(BytesIO(boardpng))

                        # Display the image
                        self.ax_board.cla()
                        self.ax_board.set_xticks([])
                        self.ax_board.set_yticks([])
                        self.ax_board.set_title("Position")
                        self.ax_board.imshow(boardimage)

                    # Update the graph
                    if best_levels is not None:
                        plot_best = best_levels * good
                        self.update_graph([plot_best, plot], ["w--", "c-"], [-biggest, biggest])
                    else:
                        self.update_graph([plot], ["w--"], [-biggest, biggest])

                # If we have any reason to end analysis early we can do so by breaking
                if depth == self.sf_depth or abs(pos_eval) > 1000:
                    break
        # If evaluation ends early, extrapolate the rest of the data
        if depth != self.sf_depth:
            raw_levels[depth:] = pos_eval

        return raw_levels, pos_eval, pos_move, changes


    # Analyse a single move in a game, returning a MoveAnalysis object
    def analyse_move(self, board:chess.Board, game:chess.pgn.Game, move:int, played:chess.Board,
                     next_move:chess.Move, show:bool):

        # Record the time spent thinking about the move
        clock = game.next().clock()
        if clock is None:
            clock = 1

        # Check if the move is a book move
        game_next = game.next()
        if game_next.next() is not None:
            # We check all books
            for book in self.books:
                inbook = book.get(board=game.next().board())
                if inbook:
                    break
        else:
            return None, True, game_next

        # If it's a book move we skip it
        if inbook:
            return None, True, game_next

        # For animated display, update text to show the position details
        if show:
            self.ax_text.cla()
            self.ax_text.set_xticks([])
            self.ax_text.set_yticks([])
            self.ax_text.set_ylim(0, 10)
            self.ax_text.set_xlim(0, self.sf_depth)
            self.ax_text.text(1, 8,
                f"Move {(move+2)//2} ({'White' if (move+1)%2 else 'Black'}): {next_move.uci()}",
                fontsize=15)
            self.fig.canvas.blit(self.ax_text.bbox)
            self.fig.canvas.flush_events()

        # Get the evaluation of the current position
        if self.previous is None:
            best_levels, best_eval, best_move, _ = \
                self.iterative_analysis(board, move, show=show, played_move=next_move)
        else:
            # Or if it was cached from the previous iteration, even better
            best_levels, best_eval, best_move = self.previous

        # For animated display, update text to show the best move details
        if show:
            self.ax_text.text(1, 6, f"Best move eval: {best_eval} Best move: {best_move.uci()}",
                              fontsize=15)
            self.fig.canvas.blit(self.ax_text.bbox)
            self.fig.canvas.flush_events()
        
        # Get the evaluation of the move that was played
        raw_levels, pos_eval, future_best_move, changes = \
            self.iterative_analysis(played, move, show=show, best_levels=best_levels, played_move=next_move)
        self.previous = (raw_levels, pos_eval, future_best_move)

        # For animated display, update text to show the played move details
        if show:
            self.ax_text.text(1, 4, f"Played move eval: {pos_eval}", fontsize=15)
            self.fig.canvas.blit(self.ax_text.bbox)
            self.fig.canvas.flush_events()

        # Calculate the length of the tactical sequence (if there is one)
        d = best_eval - pos_eval
        eval_drop = d
        if not self.fast:
            line_length = getLength(self.sf, played)
        else:
            line_length = 0

        # Consider partial credit with fixed engine-like sensitivity and consistency values
        s = 100
        c = 2
        partial_credit = exp(-(d/s)**c)

        # For animated display, update text to show the played move's quality
        if show:
            self.ax_text.text(1, 2, f"Move score: {partial_credit*100:.0f}%", fontsize=15)
            self.fig.canvas.blit(self.ax_text.bbox)
            self.fig.canvas.flush_events()

        # Calculate the risk factor for checkmate positions
        if not self.fast:
            mate_in, imbalance, risk_diff = mate_complexity(self.sf, best_eval, board, move, next_move)
        else:
            mate_in, imbalance, risk_diff = (None, 0, None)

        # Increment move counter
        game = game.next()
        move += 1

        # Normalise and filter evaluation levels
        filtered, max_amp = normalise(raw_levels, best_eval, move)
        best_filtered, best_max_amp = normalise(best_levels, best_eval, move)

        # Plot the filtered data
        if show:
            self.update_graph([best_filtered, filtered], ["w--", "c-"], [-1.2, 1.2])
        
        # Classify evaluation levels trend using SVC model, focusing on the early data
        shape, shape_proba = self.model.classify(filtered[:20])
        shape = int(shape)
        best_shape, best_shape_proba = self.model.classify(best_filtered[:20])
        best_shape = int(best_shape)

        # Plot the trend classification
        if show:
            shapes = ["Ascending", "Descending", "No Shape"]
            self.ax.text(0.6, 0.6, shapes[shape], fontsize=20)
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()

        # Get next best evaluations
        if not self.fast:
            scores = self.multi_analysis(played, 5)
        else:
            scores = np.zeros(5)

        # Construct a MoveAnalysis object to be returned
        analysis = MoveAnalysis(
            (move+1)//2, # Convert from ply to move number
            next_move.uci(),
            move % 2, # The colour that played the move
            pos_eval,
            clock,
            filtered,
            max_amp,
            inbook,
            mate_in=mate_in,
            imbalance=imbalance,
            shape=shape,
            shape_proba=shape_proba,
            eval_drop=eval_drop,
            partial_credit=partial_credit,
            line_length=line_length,
            best_move=best_move.uci(),
            best_levels=best_filtered,
            best_amp=best_max_amp,
            best_shape=best_shape,
            best_shape_proba=best_shape_proba,
            changes=changes,
            risk=risk_diff,
            next_best_evals=scores
        )

        # Return data
        return analysis, inbook, game

    # Takes a game path, opens the file, and analyses the game producing a GameAnalysis object
    def analyse(self, gamefile, show=False):
        self.previous = None

        # Open the game file
        try:
            db = open(f"{gamefile}.pgn")
            game = chess.pgn.read_game(db)
        except FileNotFoundError:
            print("Game file not found.")
            exit(1)

        # Start off in book on move 0
        inbook = True
        move = 0
        results = []

        # Set up the graphical display
        if show:
            plt.style.use("dark_background")
            plt.ion()
            plt.rcParams.update({
                "lines.color": "#212121",
                "patch.edgecolor": "black",
                "text.color": "white",
                "axes.facecolor": "#333333",
                "axes.edgecolor": "#333333",
                "axes.labelcolor": "white",
                "xtick.color": "gray",
                "ytick.color": "gray",
                "grid.color": "gray",
                "figure.facecolor": "#212121",
                "figure.edgecolor": "#212121",
                "savefig.facecolor": "#212121",
                "savefig.edgecolor": "#212121"})
            fig = plt.figure(figsize=(12, 15))
            fig.patch.set_facecolor('#212121')
            plt.clf()
            ax = fig.subplot_mosaic("AABB;CCBB")
            self.ax = ax["A"]
            self.ax_board = ax["B"]
            self.ax_text = ax["C"]
            self.fig = fig
            fig.canvas.draw()

        # Set this program as the annotator of the report
        game.headers['Annotator'] = "Cheat Detector"
        m = game
        move_count = 0

        # Count the number of moves in the game (for the progress bar)
        while (m := m.next()) is not None:
            move_count += 1

        # Go back to the start of the linked list
        game = game.game()

        print(f"Analysing {gamefile}...")

        # Scroll through every move in the game (tqdm is a nice visual progress bar)
        for _ in tqdm(range(move_count)):
            # Until we reach the end of the game
            if game.next() is None:
                break

            # Use analyse_move to analyse each move
            board = game.board()
            played = game.next().board()
            next_move = game.next().move
            analysis, inbook, game = self.analyse_move(board, game, move, played, next_move, show)

            move += 1

            # Add the move to our analysis if not a book move
            if not inbook:
                results.append(analysis)

        # Add the fully analysed game to the list of analysed games
        self.analysed_games[gamefile] = GameAnalysis(game.game(), results)
        return
