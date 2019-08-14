"""
    Assorted helper functions for the game of Chexers. These functions should be
    useful for any implentation, be it random, greedy, search-based or ML-based.
"""

import config
import numpy as np
import random

from copy import deepcopy

def getNextMoves(counterPositions, colour):
    """
        Given a dictionary of board positions for each player, returns all moves
        that can be made by a player of colour 'colour'

        Counter positions - dictionary with string for key and list of tuples as
        value. eg. {"red": [(0,0),(0,1)], "blue": [], "green": [(-1,3),(-2,2),(0,3)]}
    """

    moves = []
    allCoordinates = [a for b in counterPositions.values() for a in b]
    for coordinate in counterPositions[colour]:
        moves += getPossibleMovesFromCoordinate(allCoordinates, coordinate, colour)

    if len(moves) == 0:
        moves = [("PASS",None)]

    return moves

def getPossibleMovesFromCoordinate(allCoordinates, coordinate, colour):
    """
        Given a specific coordinate 'coordinate' and a list of the coordinates of all
        counters on the board, finds all possible moves that can be made from
        'coordinate' for a counter of colour 'colour'
    """

    possibleMoves = []

    # Check for exit moves
    if canCounterExitAtCoordinate(coordinate, colour):
        possibleMoves.append(('EXIT', coordinate))
        # Assume that if we can make an exit, we will always want to make an exit
        return possibleMoves

    # Check for step moves
    neighbourCoordinates = getStepMovesFromCoordinate(coordinate)
    for nCoordinate in neighbourCoordinates:
        if not nCoordinate in allCoordinates:
            possibleMoves.append(('MOVE', (coordinate, nCoordinate)))

    # Check for jump moves
    q = coordinate[0]
    r = coordinate[1]
    jumpNeighbourCoordinates = getJumpMovesFromCoordinate(coordinate)
    for jnCoordinate in jumpNeighbourCoordinates:
        jnq = jnCoordinate[0]
        jnr = jnCoordinate[1]
        jumpOverCoordinate = (int((jnq + q)/2), int((jnr + r)/2))
        if not jnCoordinate in allCoordinates:
            if jumpOverCoordinate in allCoordinates:
                possibleMoves.append(('JUMP', (coordinate, jnCoordinate)))

    return possibleMoves


def getStepMovesFromCoordinate(coordinates):
    """
        Finds all hexes that a counter can step to from a hex with coordinate
        'coordinates', disregarding obstacles and other counters
    """
    q = coordinates[0]
    r = coordinates[1]
    neighbours = [(q, r-1), (q+1, r-1), (q+1,r),
                                 (q, r+1), (q-1, r+1), (q-1, r)]
    return [coordinate for coordinate in neighbours
                                 if (abs(coordinate[0] + coordinate[1]) < 4)
                                 and (abs(coordinate[0]) < 4) and (abs(coordinate[1]) < 4)]

def getJumpMovesFromCoordinate(coordinates):
    """
        Finds all hexes that a counter can jump to from a hex with coordinate
        'coordinates', disregarding obstacles and other counters
    """
    q = coordinates[0]
    r = coordinates[1]
    jumpNeighbours = [(q, r-2), (q+2, r-2), (q+2, r), (q, r+2), (q-2, r+2), (q-2, r)]
    return [coordinate for coordinate in jumpNeighbours if (abs(coordinate[0]
            + coordinate[1]) < 4) and (abs(coordinate[0]) < 4) and (abs(coordinate[1]) < 4)]

def canCounterExitAtCoordinate(coordinate, colour):
    """
        Evaluates whether a counter of colour 'colour' can exit the board at a
        hex of coordinate 'coordinate'. Returns true if yes, false if no.
    """
    q = coordinate[0]
    r = coordinate[1]
    if r == 3 and colour == "green":
        return True
    elif q == 3 and colour == "red":
        return True
    elif r + q == -3 and colour == "blue":
        return True

    return False

def getNextColour(colour):
    """
        Given the current player's colour, returns the next player's colour.
    """

    if colour == "red":
        return "green"
    elif colour == "green":
        return "blue"
    else:
        return "red"

def boardStateToNNInput(boardState, colour, exits):
    """
        Description:
        Converts current board and game data to a 3D stack of 6 7x7 matrices,
        where the first three matrices represent the current counter positions of
        the current, next and next next players (0 if no counter, 1 if counter), and the next three
        represent the current number of exits made by each of the current, next and next next players
        (the matrix is a single number repeated 49 times, with that number being
        the number of exits made so far by a given player).

        TLDR:
        Converts current board state information into a format that can be
        inputted into our NN

        Input:
        - 'boardState', a dict containing player colours as keys and a list of
           the coordinates of their counters as values
        - 'colour', the colour of the current player
        - 'exits', a dict containing player colours as keys and the number of exits
           made by each player as values


       Output:
       A 7x7x6 Tensor
    """

    nninput = []
    players = (colour, getNextColour(colour), getNextColour(getNextColour(colour)))

    # Rotate board first so that the current player is always trying to exit at
    # the top right
    inputBS = boardState
    if colour in ("green", "blue"):
        inputBS = rotatedBoard(inputBS, colour)

    # Encode board state first
    for player in players:
        bsMat = np.zeros((7,7))
        for coordinate in inputBS[player]:
            bsMat.itemset((coordinate[0]+3,coordinate[1]+3),1)
        nninput.append(bsMat)


    # Encode player exits
    for player in players:
        nninput.append(np.full((7,7), exits[player]))

    return np.array(nninput)

def adjustNNPriors(P, counterPositions, colour):
    """
        Description:
        The P values outputted by the neural net will contain many values
        that should be zero but aren't. This alters 'P', a 457x1 vector
        of move prior probabilities, so that it only contains NN-calculated
        probabilities of valid moves.

        TLDR:
        Invalid moves in P are set to zero, then the vector is renormalised.

        Input:
        - P, a 457x1 vector whose values form a probability distribution of
          possible next moves
        - 'validMoves', a list of the moves that should be represented in P

        Output:
        An adjusted P vector with the same dimensions, 457x1
    """

    rotatedCounterPos = rotatedBoard(counterPositions, colour)
    validMoves = getNextMoves(rotatedCounterPos, colour)

    newP = np.zeros(config.P_OUTPUT_DIMS)
    for move in validMoves:
        e1 = encodeMove(move)
        actualMove = rotateMoveBack(move, colour)
        e2 = encodeMove(actualMove)
        newP.itemset(e2, P[e1])
    # for move in validMoves:
    #     e = encodeMove(move)
    #     newP.itemset(e, P[e])

    # Normalise the array
    return normalise(newP)

def normalise(arr):
    """
        Adjusts the values in array 'arr' so that they form a probability distribution.
    """

    norm = np.linalg.norm(arr, ord=1)
    if norm == 0:
        norm = 1

    return arr/norm

def getPositionFromMove(move, counterPositions, colour):
    """
        Returns the set of counter positions that would result from a move 'move'
        being made on a board with existing counter positions 'counterPositions'
    """

    actionType = move[0]
    newCounterPositions = deepcopy(counterPositions)

    if actionType == "PASS":
        return newCounterPositions
    elif actionType == "EXIT":
        newCounterPositions[colour] = [c for c in counterPositions[colour] if c != move[1]]
    elif actionType == "MOVE":
        fromCoordinate = move[1][0]
        toCoordinate = move[1][1]
        newCoordinates = []
        newCounterPositions[colour] = [c if c != fromCoordinate else toCoordinate for c in counterPositions[colour]]
    elif actionType == "JUMP":
        fromCoordinate = move[1][0]
        toCoordinate = move[1][1]
        jumpedOverCoordinate = (int((fromCoordinate[0] + toCoordinate[0])/2), int((fromCoordinate[1] + toCoordinate[1])/2))

        # Get colour of jumpedOverCoordinate:
        jumpedOverColour = None
        for counter in newCounterPositions.keys():
            if jumpedOverCoordinate in newCounterPositions[counter]:
                jumpedOverColour = counter
                break

        # Check if a capture has occured
        if jumpedOverColour != colour:
            newCounterPositions[colour].append(jumpedOverCoordinate)
            newCounterPositions[jumpedOverColour] = [c for c in counterPositions[jumpedOverColour] if c != jumpedOverCoordinate]

        newCounterPositions[colour] = [c if c != fromCoordinate else toCoordinate for c in newCounterPositions[colour]]

    return newCounterPositions

def encodeCoordinate(coordinate):
    """
        Converts a hex coordinate of type (q,r) to a single integer [0,37).
        Easier for indexing and representation in our NN outputs.
    """
    cMap = {0:0, 1:4, 2:9, 3:15, 4:22, 5:28, 6:33}
    dMap = {0:3, 1:2, 2:1, 3:0, 4:0, 5:0, 6:0}

    q = coordinate[0]
    r = coordinate[1]

    return cMap[r+3] + q+3 - dMap[r+3]

def encodeMove(move):
    """
        Converts a move of type (move type, (from coord, to coord)) to a single
        integer [0,456]. Easier for indexing and representation in our NN outputs.

        Inputs:
        - 'move', which is typically of form ('ACTION', ((q1,r1),(q2,r2))).

        Outputs:
        - An integer, [0,456]
    """
    moveType = move[0]

    if moveType == "EXIT":
        q = move[1][0]
        r = move[1][1]
        if r == 3: # exit point for green
            return 444 + q + 3
        elif q + r == -3: # exit point for blue
            return 448 + q + 3
        else: # exit point for red
            return 452 + r + 3
    elif moveType == "PASS":
        return 456
    else:
        fromCoord = encodeCoordinate(move[1][0])
        fromq = move[1][0][0]
        fromr = move[1][0][1]
        toq = move[1][1][0]
        tor = move[1][1][1]

        a = 0
        if toq == fromq:
            if tor < fromr:
                a = 0
            else:
                a = 5
        elif toq < fromq:
            if tor == fromr:
                a = 1
            else:
                a = 2
        else: #toq > fromq
            if tor == fromr:
                a = 4
            else:
                a = 3

        if moveType == "JUMP":
            a += 6

        return 12*fromCoord + a

def decodeMove(move, options):
    """
        Sister function of encodeMove(). Given an encoded move 'move', and a list
        of moves that it could be ('options'), decodes 'move' to the printable
        format.
    """
    for option in options:
        if encodeMove(option) == move:
            return option

def printBoard(counterPositions, message="", debug=False, **kwargs):
    """
    Helper function to print a drawing of a hexagonal board's contents.

    Arguments:

    * Counter positions - dictionary with string for key and list of tuples as
    a value. eg. {"red": [(0,0),(0,1)], "blue": [], "green": [(-1,3),(-2,2),(0,3)]}

    * `board_dict` -- dictionary with tuples for keys and anything printable
    for values. The tuple keys are interpreted as hexagonal coordinates (using
    the axial coordinate system outlined in the project specification) and the
    values are formatted as strings and placed in the drawing at the corres-
    ponding location (only the first 5 characters of each string are used, to
    keep the drawings small). Coordinates with missing values are left blank.

    Keyword arguments:

    * `message` -- an optional message to include on the first line of the
    drawing (above the board) -- default `""` (resulting in a blank message).
    * `debug` -- for a larger board drawing that includes the coordinates
    inside each hex, set this to `True` -- default `False`.
    * Or, any other keyword arguments! They will be forwarded to `print()`.
    """

    pmap = {"red": "R", "green": "G", "blue": "B"}
    board_dict = {}
    for player in counterPositions.keys():
        for hex in counterPositions[player]:
            board_dict[hex] = pmap[player]

    # Set up the board template:
    if not debug:
        # Use the normal board template (smaller, not showing coordinates)
        template = """# {0}
#           .-'-._.-'-._.-'-._.-'-.
#          |{16:}|{23:}|{29:}|{34:}|
#        .-'-._.-'-._.-'-._.-'-._.-'-.
#       |{10:}|{17:}|{24:}|{30:}|{35:}|
#     .-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
#    |{05:}|{11:}|{18:}|{25:}|{31:}|{36:}|
#  .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
# |{01:}|{06:}|{12:}|{19:}|{26:}|{32:}|{37:}|
# '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
#    |{02:}|{07:}|{13:}|{20:}|{27:}|{33:}|
#    '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
#       |{03:}|{08:}|{14:}|{21:}|{28:}|
#       '-._.-'-._.-'-._.-'-._.-'-._.-'
#          |{04:}|{09:}|{15:}|{22:}|
#          '-._.-'-._.-'-._.-'-._.-'"""
    else:
        # Use the debug board template (larger, showing coordinates)
        template = """# {0}
#              ,-' `-._,-' `-._,-' `-._,-' `-.
#             | {16:} | {23:} | {29:} | {34:} |
#             |  0,-3 |  1,-3 |  2,-3 |  3,-3 |
#          ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
#         | {10:} | {17:} | {24:} | {30:} | {35:} |
#         | -1,-2 |  0,-2 |  1,-2 |  2,-2 |  3,-2 |
#      ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
#     | {05:} | {11:} | {18:} | {25:} | {31:} | {36:} |
#     | -2,-1 | -1,-1 |  0,-1 |  1,-1 |  2,-1 |  3,-1 |
#  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
# | {01:} | {06:} | {12:} | {19:} | {26:} | {32:} | {37:} |
# | -3, 0 | -2, 0 | -1, 0 |  0, 0 |  1, 0 |  2, 0 |  3, 0 |
#  `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
#     | {02:} | {07:} | {13:} | {20:} | {27:} | {33:} |
#     | -3, 1 | -2, 1 | -1, 1 |  0, 1 |  1, 1 |  2, 1 |
#      `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
#         | {03:} | {08:} | {14:} | {21:} | {28:} |
#         | -3, 2 | -2, 2 | -1, 2 |  0, 2 |  1, 2 | key:
#          `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' ,-' `-.
#             | {04:} | {09:} | {15:} | {22:} |   | input |
#             | -3, 3 | -2, 3 | -1, 3 |  0, 3 |   |  q, r |
#              `-._,-' `-._,-' `-._,-' `-._,-'     `-._,-'"""

    # prepare the provided board contents as strings, formatted to size.
    ran = range(-3, +3+1)
    cells = []
    for qr in [(q,r) for q in ran for r in ran if -q-r in ran]:
        if qr in board_dict:
            cell = str(board_dict[qr]).center(5)
        else:
            cell = "     " # 5 spaces will fill a cell
        cells.append(cell)

    # fill in the template to create the board drawing, then print!
    board = template.format(message, *cells)
    print(board, **kwargs)

def rotatedBoard(counterPositions, colour):
    """
        Takes a set of counter positions and rotates the entire board so the current
        player 'colour' is always aiming to exit at the top right hexes from their
        perspective. This simplifies things for the NN.
    """

    if colour == "red":
        return deepcopy(counterPositions)

    rotmap = {"green": {(0,-3):(-3,3), (1,-3):(-3,2), (2,-3):(-3,1), (3,-3):(-3,0), (-1,-2):(-2,3),
                    (0,-2):(-2,2), (1,-2):(-2,1), (2,-2):(-2,0), (3,-2):(-2,-1), (-2,-1):(-1,3),
                    (-1,-1):(-1,2), (0,-1):(-1,1), (1,-1):(-1,0), (2,-1):(-1,-1), (3,-1):(-1,-2),
                    (-3,0):(0,3), (-2,0):(0,2), (-1,0):(0,1), (0,0):(0,0), (1,0):(0,-1),
                    (2,0):(0,-2), (3,0):(0,-3), (-3,1):(1,2), (-2,1):(1,1), (-1,1):(1,0),
                    (0,1):(1,-1), (1,1):(1,-2), (2,1):(1,-3), (-3,2):(2,1), (-2,2):(2,0), (-1,2):(2,-1), (0,2):(2,-2),
                    (1,2):(2,-3), (-3,3):(3,0),(-2,3):(3,-1), (-1,3):(3,-2), (0,3):(3,-3)},
            "blue": {(3,0):(-3,3), (2,1):(-3,2), (1,2):(-3,1), (0,3):(-3,0), (3,-1):(-2,3),
                        (2,0):(-2,2), (1,1):(-2,1), (0,2):(-2,0), (-1,3):(-2,-1), (3,-2):(-1,3),
                        (2,-1):(-1,2), (1,0):(-1,1), (0,1):(-1,0), (-1,2):(-1,-1), (-2,3):(-1,-2),
                        (3,-3):(0,3), (2,-2):(0,2), (1,-1):(0,1), (0,0):(0,0), (-1,1):(0,-1),
                        (-2,2):(0,-2), (-3,3):(0,-3), (2,-3):(1,2), (1,-2):(1,1), (0,-1):(1,0),
                        (-1,0):(1,-1), (-2,1):(1,-2), (-3,2):(1,-3), (1,-3):(2,1), (0,-2):(2,0), (-1,-1):(2,-1), (-2,0):(2,-2),
                        (-3,1):(2,-3), (0,-3):(3,0), (-1,-2):(3,-1), (-2,-1):(3,-2), (-3,0):(3,-3)}}

    rotatedBoard = {"red": [], "green": [], "blue": []}
    for player in rotatedBoard.keys():
        for counter in counterPositions[player]:
            rotatedBoard[player].append(rotmap[colour][counter])

    return rotatedBoard

def rotateMoveBack(move, colour):
    """
        The neural net only returns moves from the perspective of the player at the
        bottom left of the board. This function generalises moves made by players so that
        they are valide for players from all starting positions.
    """

    if colour == "red":
        return move

    rotmap = {'green': {(-3, 3): (0, -3), (-3, 2): (1, -3), (-3, 1): (2, -3),
                        (-3, 0): (3, -3), (-2, 3): (-1, -2), (-2, 2): (0, -2),
                        (-2, 1): (1, -2), (-2, 0): (2, -2), (-2, -1): (3, -2),
                        (-1, 3): (-2, -1), (-1, 2): (-1, -1), (-1, 1): (0, -1),
                        (-1, 0): (1, -1), (-1, -1): (2, -1), (-1, -2): (3, -1),
                        (0, 3): (-3, 0), (0, 2): (-2, 0), (0, 1): (-1, 0),
                        (0, 0):(0, 0), (0, -1):(1, 0), (0, -2):(2, 0), (0, -3):(3, 0),
                        (1, 2):(-3, 1), (1, 1):(-2, 1), (1, 0):(-1, 1), (1, -1):(0, 1),
                        (1, -2):(1, 1), (1, -3):(2, 1), (2, 1):(-3, 2), (2, 0):(-2, 2),
                        (2, -1):(-1, 2), (2, -2):(0, 2), (2, -3):(1, 2), (3, 0):(-3, 3),
                        (3, -1):(-2, 3), (3, -2): (-1, 3), (3, -3): (0, 3)},
                    'blue': {(-3, 3): (3, 0), (-3, 2): (2, 1), (-3, 1): (1, 2),
                                (-3, 0): (0, 3), (-2, 3): (3, -1), (-2, 2): (2, 0),
                                (-2, 1): (1, 1), (-2, 0): (0, 2), (-2, -1): (-1, 3),
                                (-1, 3): (3, -2), (-1, 2): (2, -1), (-1, 1): (1, 0),
                                (-1, 0): (0, 1), (3, -1): (-1, 2), (-1, -2): (-2, 3),
                                (0, 3): (3, -3), (0, 2): (2, -2), (0, 1): (1, -1),
                                (0, 0): (0, 0), (0, -1): (-1, 1), (0, -2): (-2, 2),
                                (0, -3): (-3, 3), (1, 2): (2, -3), (1, 1): (1, -2),
                                (1, 0): (0, -1), (1, -1): (-1, 0), (1, -2): (-2, 1),
                                (1, -3): (-3, 2), (2, 1): (1, -3), (2, 0): (0, -2),
                                (2, -1): (-1, -1), (-1,-1):(-1,2), (2, -2): (-2, 0), (2, -3): (-3, 1),
                                (3, 0): (0, -3), (3, -2): (-2, -1), (3, -3): (-3, 0)}}

    if move[0] in ("JUMP", "MOVE"):
        return (move[0], (rotmap[colour][move[1][0]], rotmap[colour][move[1][1]]))
    elif move[0] == "EXIT":
        return (move[0], rotmap[colour][move[1]])
    else:
        return move

def addDirichletNoise(priors):
    newPriors = np.zeros(config.P_OUTPUT_DIMS)
    nonZeros = np.nonzero(priors)
    numNonZero = len(nonZeros[0])

    dirch = np.random.dirichlet([1]*numNonZero, 1)[0]

    i = 0
    for index in nonZeros:
        newPriors[index] = 0.75*priors[index] + 0.25*dirch[i]
        i += 1

    return newPriors
