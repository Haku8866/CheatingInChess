# We use matplotlib to display some graphs in debug/analysis functions
import matplotlib.pyplot as plt
import numpy as np
import chess
import chess.pgn

# Stores all information about a game including the PGN report writing code
class GameAnalysis():
    def __init__(self, game, moves):
        self.moves = moves
        self.pgn = game
    
    # Functions to print game moves for debugging purposes
    def __str__(self):
        return "\n".join([str(move) for move in self.moves])
    
    def show(self):
        for move in self.moves: print(move)

    # A function to plot the amplitudes of moves, this was used for analysis
    def display_amplitudes(self):
        amps = []
        best_move_amps = []

        for move in self.moves:
            # We exclude amplitudes above 50 as these are from checkmates
            if move.level_amplitude <= 50:
                amps.append(move.level_amplitude)
            if move.best_move.level_amplitude<= 50:
                best_move_amps.append(move.best_move.level_amplitude)
            
        plt.plot(amps, "r-")
        plt.plot(best_move_amps, "g-")
        plt.show()

    # Annotates the stored PGN, including adding the game report
    def annotatePGN(self, model):
        # The names of the players
        white_name = self.pgn.headers['White']
        black_name = self.pgn.headers['Black']

        # Skip opening moves, we go to the first move number (in ply, so *2)
        skip_to = (self.moves[0].move_num)*2
        if self.moves[0].white:
            skip_to -= 1
        
        # Clear any existing comments
        for _ in range(skip_to):
            self.pgn.comment = ""
            self.pgn = self.pgn.next()

        # Now we go through non-book moves
        for analysis in self.moves:
            # Clear existing comments
            self.pgn.comment = ""
            self.pgn.starting_comment = ""
            self.pgn.nags.clear()

            # Decide whether negative or postive numbers are good based on the colour
            good = 1 if analysis.white else -1

            # If the move is the best move, consider commenting on it
            if (analysis.move == analysis.best_move):
                # Model confidence of it being an engine move must be above 50%
                # and the move must be in the critical region of the game
                confidence = model.classify(analysis)*100
                if confidence > 50 and abs(analysis.best_eval) < 300:
                    self.pgn.comment = f" This is an engine move, {confidence:.1f}% confidence."
                    # Add a "!?" symbol (also known as a NAG)
                    self.pgn.nags.add(chess.pgn.NAG_SPECULATIVE_MOVE)
                    # Also comment on the tactical sequence length
                    if analysis.line_length > 0:
                        self.pgn.comment += f" This is the first move in a {analysis.line_length+1}-move idea."
                # If not an engine move, comment on its quality if it has a negative
                # evaluation drop (meaning it was better than the best move)
                elif analysis.eval_drop*good <= -10 and abs(analysis.best_eval) < 300:
                    self.pgn.comment = f" Stockfish likes this better than its original best move."
                    self.pgn.nags.add(chess.pgn.NAG_BRILLIANT_MOVE)
                    # Also comment on tactical sequence length
                    if analysis.line_length > 0:
                        self.pgn.comment += f" This is the first move in a {analysis.line_length+1}-move idea."
            try:
                if analysis.risk is not None:
                    self.pgn.comment += f" This was an unnecessary risk ({analysis.risk:.2f} risk score)."
            except:
                pass
            # If there is another move then we continue
            if self.pgn.next() is not None:
                self.pgn = self.pgn.next()
        
        # Skip to the last move of the game
        # while self.pgn.next() is not None:
        #     self.pgn = self.pgn.next()

        # Clear existing comments on the last move and begin writing report
        self.pgn.starting_comment = ""
        self.pgn.nags.clear()
        self.pgn.comment = f"\nTotal non-book moves: {len(self.moves)}\n"

        # Gather statistics before writing, this code counts the number of engine moves, total moves,
        # total moves in critical period, good moves, "better than stockfish" moves, as well as
        # combinations of these figures.
        names = [white_name, black_name]
        total = [0, 0]
        total_under300 = [0, 0]
        engine = [0, 0]
        brilliant = [0, 0]
        excellent = [0, 0]
        excellent_abs_under300 = [0, 0]
        confidence = [[], []]
        prs = [0, 0]
        for move in self.moves:
            colour = 0 if move.white else 1
            good = 1 if move.white else -1
            total[colour] += 1
            # Same criteria for engine, brilliant, and critical moves as defined previously
            classification = model.classify(move)
            if classification > 0.5 and (move.best_move == move.move) and abs(move.best_eval) < 300:
                engine[colour] += 1
                confidence[colour].append(classification)
            if move.eval_drop*good <= -10:
                brilliant[colour] += 1
            if move.move == move.best_move:
                if abs(move.best_eval) < 300:
                    excellent_abs_under300[colour] += 1
                excellent[colour] += 1
            if abs(move.best_eval) < 300:
                total_under300[colour] += 1
                
        # Division defined as a total function where x/0 = 0
        d = lambda x,y: 0 if y==0 else x/y

        # For both colours, insert their half of the report into the comment
        report = ""
        for colour in range(2):
            # Positive rate (pr) is defined as number of engine moves divided by critical moves
            prs[colour] = d(engine[colour],total_under300[colour])*100
            report += f"\nReport for {names[colour]}:\nMoves: {total[colour]}"
            report += f"\nGood moves: {excellent[colour]} "
            report += f"({d(excellent[colour],total[colour])*100:.1f}% of all moves)"
            report += f"\nBrilliant moves: {brilliant[colour]} "
            report += f"({d(brilliant[colour],total[colour])*100:.1f}% of all moves)"
            report += f"\nEngine moves: {engine[colour]} "
            # This is where we state the positive rate
            report += f"({prs[colour]:.1f}% of analysed "
            report += f"({total_under300[colour]}), {d(engine[colour],excellent[colour])*100:.1f}% "
            report += f"of good moves, {d(engine[colour],excellent_abs_under300[colour])*100:.1f}% "
            report += f"of good moves made during critical period)"
            # If there are any engine moves, get the average confidence of them
            if len(confidence[colour]) > 0:
                report += f"\nAverage confidence of engine moves: {np.mean(confidence[colour])*100:.2f}%"
            report += f"\n"

        # Overall positive rate of the game across both players
        pr = d(sum(engine),sum(total_under300))*100
        report += f"Total positive rate: {pr:.2f}%"
        self.pgn.comment += report

        # Traverse back to start of the linked list and return some of the values calculated
        self.pgn = self.pgn.game()
        return pr, prs[0], prs[1]
