from cnn import CNN
from chexers import *
from mcts import MCTS
from boardState import BoardState
from copy import deepcopy

import numpy as np

convNet = CNN("AlphaOneB.h5")

testBoardState = {"red": [(-3,2),(0,-1),(-2,0),(1,1)],
                    "blue": [(-1,0),(1,-1),(-2,2)],
                    "green": [(1,-3),(0,0),(-3,1),(2,1)]}
testExitState = {"red": [(-1,-1),(-2,0),(-3,3),(-1,1)],
                    "blue": [],
                    "green": [(-2,1),(0,0)]}
testExits = {"red": 1, "blue": 0, "green": 0}
testCurrentPlayer = "red"

if False: # Switch this to test the eval function of the CNN

    v, p = convNet.evaluate(boardState=testExitState, colour=testCurrentPlayer, exits=testExits)
    nextMoves = getNextMoves(testExitState, testCurrentPlayer)
    P = adjustNNPriors(p, testExitState, testCurrentPlayer)

    for move in nextMoves:
        print("{}: {}".format(move, P[encodeMove(move)]))

if False: # Switch this to test v outputs

    full = [(-3,2),(0,-1),(-2,0),(1,1)]
    for i in range(0, len(full)+1):
        bs = deepcopy(testBoardState)
        bs["red"] = full[0:i]

        v, p = convNet.evaluate(boardState=bs, colour=testCurrentPlayer, exits=testExits)

        print("v with {} coordinate for red: {}".format(i+1, v))


if False: # Switch this to test a sample game from a mid-game point

    # Test game data
    currentBS = {"red": [(-3,0),(-3,1),(-3,2),(-3,3)],
                 "green": [(0,-3),(1,-3),(2,-3),(3,-3)],
                 "blue": [(0,3),(1,2),(2,1),(3,0)]}
    currentColour = "red"
    exits = {"red": 0, "green": 0, "blue": 0}

    while (4 not in exits.values()):
        nextMoves = getNextMoves(currentBS, currentColour)

        v, p = convNet.evaluate(currentBS, currentColour, exits)
        priors = adjustNNPriors(p, nextMoves)

        nextMove = decodeMove(np.argmax(priors), nextMoves)

        if nextMove[0] == 'EXIT':
            exits[currentColour] += 1

        printBoard(currentBS)
        print("{} makes move {}".format(currentColour, nextMove))

        currentBS = getPositionFromMove(nextMove, currentBS, currentColour)
        currentColour = getNextColour(currentColour)

if False: # Switch this to test .h5 file importing

    import h5py
    weights = h5py.File('AlphaOneA.h5', 'r')['model_weights']

    layerNames = weights.keys()

    print(weights['policyHead']['policyHead']['kernel:0'][...])

    import pandas as pd

    with pd.read_hdf('./AlphaOneA.h5') as d:
        print(d)

if False: # Switch this to test keras model weight exporting
    print(convNet.model.get_layer(name="conv2d_4").get_weights())

if True: # Switch this to test a single MCTS search

    mcts = MCTS(BoardState(testExitState, testCurrentPlayer, convNet), 10000, 1, 1)
    moveProbs = mcts.search()

    nextMoves = getNextMoves(testExitState, testCurrentPlayer)
    for move in nextMoves:
        print("{}: {}".format(move, moveProbs[encodeMove(move)]))
    print(sum(moveProbs))
    print("")
    for move in nextMoves:
        print("{}: {}".format(move, mcts.rootState.P[encodeMove(move)]))
    print(sum(normalise(mcts.rootState.P)))

if False: # Switch this to test the move encoder
    print(encodeMove(('MOVE', ((0, 3), (0, 2)))))
    print(encodeMove(('MOVE', ((0, 3), (1, 2)))))

if False: # Switch this to test the coordinate encoder
    print(encodeCoordinate((0,3)))
    print(encodeCoordinate((0,2)))
    print(encodeCoordinate((1,2)))
