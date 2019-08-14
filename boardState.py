from chexers import *
from puct import PUCT

import config
import numpy as np

from copy import deepcopy

class BoardState:

    def __init__(self, counterPositions, colour, convNet):

        # Game stuff
        self.counterPositions = counterPositions
        self.exits = {"red": 0, "green": 0, "blue": 0}
        self.bsNum = 1
        self.positionHistory = set() # This goes across separate boardstate objects to avoid looping during training

        # Neural Net stuff
        self.convNet = convNet
        self.P = None
        self.V = 0
        self.Q = 0
        self.N = 0
        self.prior = 0
        self.vSum = 0

        # Node stuff
        self.parent = None
        self.colour = colour
        self.nextMoves = {}

    def hashableCounterPosition(self, counterPosition, exits):
        return (tuple(counterPosition["red"]), tuple(counterPosition["green"]), tuple(counterPosition["blue"]), exits["red"], exits["green"], exits["blue"])

    def expand(self):
        """
            Initialises all BoardStates that can be reached from the current
            BoardState by a player of colour self.colour.
        """
        winner = self.gameWinner()
        if winner:
            print("winner winner!")
            if winner == self.colour:
                self.V = 0.85 + 0.15*(1 - self.bsNum/config.MAX_MOVES) # Higher reward if win took less moves
            elif winner == "draw":
                self.V = 0
            else:
                self.V = -0.85 - 0.15*(self.bsNum/config.MAX_MOVES) # Higher penalty if loss took less moves
        else:
            # Find a list of moves that we can take and their resultant counter positions
            nextMoves = getNextMoves(self.counterPositions, self.colour)
            nextColour = getNextColour(self.colour)

            # Push our current BoardState through our conv net first
            v, p = self.convNet.evaluate(self.counterPositions, self.colour, self.exits)
            self.V = v
            self.P = adjustNNPriors(p, self.counterPositions, self.colour)
            if not self.parent and nextMoves[0][0] != "PASS":
                self.P = addDirichletNoise(self.P)

            nextPositions = []

            # Encourage exits and disable looping
            modified = False
            for move in nextMoves:
                # if move[0] == "EXIT":
                #     self.P[encodeMove(move)] *= 1.5
                #     modified = True
                position = getPositionFromMove(move, self.counterPositions, self.colour)
                nextPositions.append(position)

                if self.hashableCounterPosition(position, self.exits) in self.positionHistory:
                    self.P[encodeMove(move)] = 0
                    modified = True

            if modified:
                self.P = normalise(self.P)


            # Create new BoardState instances for each move and add to tree
            i = 0
            for move in nextMoves:

                position = nextPositions[i]
                i += 1
                nextState = BoardState(position, nextColour, self.convNet)
                nextState.parent = self

                exits = self.exits.copy()
                nextState.exits = exits
                nextState.bsNum = self.bsNum + 1
                nextState.prior = self.P[encodeMove(move)]

                self.nextMoves[move] = nextState

                if move[0] == "EXIT":
                    nextState.exits[self.colour] += 1
                elif move[0] == "PASS":
                    nextState.prior = 1
                # else:
                #     #if self.hashableCounterPosition(position, exits) not in self.positionHistory:
                #     nextState.prior = self.P[encodeMove(move)]

    def getNextState(self):
        """
            Used for traversal of MCTS tree - chooses the best child to visit,
            ie. the child with the highest PUCT score.
        """

        maxPUCT = -np.inf
        nextMove = None
        nextState = None
        for move in self.nextMoves.keys():
            state = self.nextMoves[move]
            puctScore = PUCT(state)
            if puctScore > maxPUCT:
                maxPUCT = puctScore
                nextState = state
                nextMove = move
        return nextState

    def gameWinner(self):
        """
            Returns draw or the player that won the game, None if the game is
            still going.
        """

        if self.bsNum > config.MAX_MOVES:
            return "draw"

        for player in self.exits.keys():
            if self.exits[player] >= 4:
                return player

        return None

    def asMatrix(self, colour):
        """
            Returns the current board state in matrix form, ready to be put into
            our neural net. This matrix contains only counters of colour 'colour'.
            Other counters are ignored.

            The outputted matrix is 7x7, which doesn't totally reflect the hexagonal
            nature of the board (only 37 hexes total), but this is done to simplify
            things.
        """
        boardMatrix = np.full((7,7),0)
        for cp in self.counterPositions[colour]:
            boardMatrix.itemset((cp[0]+3, cp[1]+3), 1)
        return boardMatrix

    def generateSimData(self):
        """
            Generates simulated game data that matches the format required for
            cnn.train(), except it's adapted for cnn.evaluate()
        """

        t = 0
        tBS = self
        simData = []
        while t < config.INPUT_HISTORY:
            if tBS.parent:
                simData.append((None, tBS.parent))
                tBS = tBS.parent
            else:
                simData.append((None, tBS))
            t += 1

        return simData

    def adjustP(self, validMoves):
        """
            The P values outputted by the neural net will contain many values
            that should be zero but aren't. This alters self.P so that it only
            contains NN-calculated probabilities of valid moves. Invalid moves
            are set to zero, then the vector is renormalised.
        """
        newP = np.zeros(config.P_OUTPUT_DIMS)
        for move in validMoves:
            e = encodeMove(move)
            newP.itemset(e, self.P[e])

        # Normalise the array
        norm = np.linalg.norm(newP, ord=1)
        if norm == 0:
            norm = 1

        self.P = newP/norm
