from cnn import CNN
from mcts import MCTS
from boardState import BoardState
from chexers import *
from copy import copy

import numpy as np
import config
import gc
from git import Repo

import gc

def train(modelOutputFile):
    """
        Top level function for training a fully working Chexers player
    """

    # Initialise a convolutional neural net with random weights
    convNet = CNN(modelOutputFile)
    #repo = Repo("/Users/jeromehan/Documents/GitHub/AlphaChexers")
    #master = repo.remotes.origin

    # Simulate games and train immediately using single game data
    #for i in range(1,2):

        # master.fetch()
        # master.pull()

    gameData, winner, moveNum = simulateGame(convNet, i)
    convNet.train(gameData, winner, moveNum)

        # Export the net's weights and structure for future reuse
    convNet.save()

    gc.collect()

        # repo.git.add(update=True)
        # repo.index.commit("update weights")
        # master.push()


def simulateGame(convNet, gameNum):
    """
        Simulates a full game of chexers, outputs the game data in a trainable
        format
    """

    gameData = []
    boardHistory = []
    moveNum = 0

    currentBS = BoardState({"red": [(-3,0),(-3,1),(-3,2),(-3,3)],
                            "green": [(0,-3),(1,-3),(2,-3),(3,-3)],
                            "blue": [(0,3),(1,2),(2,1),(3,0)]}, "red", convNet)
    currentColour = "red"
    exits = {"red": 0, "green": 0, "blue": 0}
    positionHistory = set()

    while not currentBS.gameWinner() and moveNum <= config.MAX_MOVES:

        searcher = MCTS(currentBS, config.NUM_SIMS, gameNum, moveNum)
        moveProbs = searcher.search()

        positionHistory.add(currentBS.hashableCounterPosition(currentBS.counterPositions, exits))
        boardHistory.append(currentBS.counterPositions)
        printBoard(currentBS.counterPositions)
        print(exits)

        nextColour = getNextColour(currentColour)

        gameData.append((moveProbs, currentBS))

        maxMove = np.argmax(moveProbs)
        options = getNextMoves(currentBS.counterPositions, currentColour)
        move = decodeMove(maxMove, options)

        currentBS = BoardState(currentBS.nextMoves[move].counterPositions, nextColour, convNet)
        currentBS.exits = copy(exits)
        currentBS.bsNum = moveNum + 1
        currentBS.positionHistory = positionHistory

        if move[0]=="EXIT":
            exits[currentColour] += 1

        currentColour = nextColour
        moveNum += 1

    winner = currentBS.gameWinner()
    print("             winner: {}".format(winner))

    return gameData, winner, moveNum

def getRecentBoardHistory(boardHistory, moveNum, backtrack):
    """
        Retrieves the last 'backtrack' moves played in a game with history
        'boardHistory', where 'boardHistory' is a list of counter positions,
        one for each move played in an ongoing game.
    """
    moveNumLower = moveNum - backtrack if moveNum >= backtrack else 0
    return boardHistory[moveNumLower:moveNum+1]


modelOutputFile = "AlphaOneA.h5"
#train(config.NUM_GAMES, modelOutputFile)
for i in range(1, config.NUM_GAMES):
    gc.collect()
    train(modelOutputFile)
