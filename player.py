from random import randint

class RandomPlayer:

    def __init__(self, colour):
        """
        This method is called once at the beginning of the game to initialise
        your player. You should use this opportunity to set up your own internal
        representation of the game state, and any other information about the
        game state you would like to maintain for the duration of the game.

        The parameter colour will be a string representing the player your
        program will play as (Red, Green or Blue). The value will be one of the
        strings "red", "green", or "blue" correspondingly.
        """
        # TODO: Set up state representation.

        self.boardState = {"red": [(-3,0), (-3,1), (-3,2), (-3,3)], "blue": [(0,3), (1,2), (2,1), (3,0)],
                      "green": [(0,-3), (1,-3), (2,-3), (3,-3)]}
        self.colour = colour
        self.stepMovesFromCoordinate = {}
        self.jumpMovesFromCoordinate = {}

        self.getStepJumpMoves()


    def action(self):
        """
        This method is called at the beginning of each of your turns to request
        a choice of action from your program.

        Based on the current state of the game, your player should select and
        return an allowed action to play on this turn. If there are no allowed
        actions, your player must return a pass instead. The action (or pass)
        must be represented based on the above instructions for representing
        actions.
        """
        # TODO: Decide what action to take.

        # Make a random move
        moves = []
        for c in self.boardState[self.colour]:
            moves += self.getPossibleMovesFromCoordinate(c)
        if len(moves) > 0:
            moveInt = randint(0, len(moves)-1)
            return moves[moveInt]

        return ("PASS", None)


    def update(self, colour, action):
        """
        This method is called at the end of every turn (including your player’s
        turns) to inform your player about the most recent action. You should
        use this opportunity to maintain your internal representation of the
        game state and any other information about the game you are storing.

        The parameter colour will be a string representing the player whose turn
        it is (Red, Green or Blue). The value will be one of the strings "red",
        "green", or "blue" correspondingly.

        The parameter action is a representation of the most recent action (or
        pass) conforming to the above in- structions for representing actions.

        You may assume that action will always correspond to an allowed action
        (or pass) for the player colour (your method does not need to validate
        the action/pass against the game rules).
        """
        # TODO: Update state representation in response to action.

        actionType = action[0]

        if actionType == "EXIT":
            self.boardState[colour] = [c for c in self.boardState[colour] if c != action[1]]
        elif actionType == "MOVE":
            fromCoordinate = action[1][0]
            toCoordinate = action[1][1]
            newCoordinates = []
            for c in self.boardState[colour]:
                if c != fromCoordinate:
                    newCoordinates.append(c)
                else:
                    newCoordinates.append(toCoordinate)
            self.boardState[colour] = newCoordinates
        elif actionType == "JUMP":
            fromCoordinate = action[1][0]
            toCoordinate = action[1][1]
            jumpedOverCoordinate = (int((fromCoordinate[0] + toCoordinate[0])/2), int((fromCoordinate[1] + toCoordinate[1])/2))

            # Get colour of jumpedOverCoordinate:
            jumpedOverColour = None
            for counter in ("red", "green", "blue"):
                if jumpedOverCoordinate in self.boardState[counter]:
                    jumpedOverColour = counter
                    break

            # Check if a capture has occured
            if jumpedOverColour != colour:
                self.boardState[colour].append(jumpedOverCoordinate)
                self.boardState[jumpedOverColour] = [c for c in self.boardState[jumpedOverColour] if c != jumpedOverCoordinate]

            self.boardState[colour] = [c if c != fromCoordinate else toCoordinate for c in self.boardState[colour]]



    def getPossibleMovesFromCoordinate(self, coordinate):

        possibleMoves = []

        allCoordinates = [j for i in self.boardState.values() for j in i]

        # Check for exit moves
        if self.canCounterExitAtCoordinate(coordinate):
            possibleMoves.append(('EXIT', coordinate))
            # Assume that if we can make an exit, we will always want to make an exit
            return possibleMoves

        # Check for step moves
        neighbourCoordinates = self.stepMovesFromCoordinate[coordinate]
        for nCoordinate in neighbourCoordinates:
            if not nCoordinate in allCoordinates:
                possibleMoves.append(('MOVE', (coordinate, nCoordinate)))

        # Check for jump moves
        q = coordinate[0]
        r = coordinate[1]
        jumpNeighbourCoordinates = self.jumpMovesFromCoordinate[coordinate]
        for jnCoordinate in jumpNeighbourCoordinates:
            jnq = jnCoordinate[0]
            jnr = jnCoordinate[1]
            jumpOverCoordinate = (int((jnq + q)/2), int((jnr + r)/2))
            if not jnCoordinate in allCoordinates:
                if jumpOverCoordinate in allCoordinates:
                    possibleMoves.append(('JUMP', (coordinate, jnCoordinate)))

        return possibleMoves

    def getStepMovesFromCoordinate(self, coordinates):
        q = coordinates[0]
        r = coordinates[1]
        neighbours = [(q, r-1), (q+1, r-1), (q+1,r),
                                     (q, r+1), (q-1, r+1), (q-1, r)]
        return [coordinate for coordinate in neighbours
                                     if (abs(coordinate[0] + coordinate[1]) < 4)
                                     and (abs(coordinate[0]) < 4) and (abs(coordinate[1]) < 4)]

    def getJumpMovesFromCoordinate(self, coordinates):
        q = coordinates[0]
        r = coordinates[1]
        jumpNeighbours = [(q, r-2), (q+2, r-2), (q+2, r), (q, r+2), (q-2, r+2), (q-2, r)]
        return [coordinate for coordinate in jumpNeighbours if (abs(coordinate[0]
                + coordinate[1]) < 4) and (abs(coordinate[0]) < 4) and (abs(coordinate[1]) < 4)]

    def canCounterExitAtCoordinate(self, coordinate):
        q = coordinate[0]
        r = coordinate[1]
        if r == 3 and self.colour == "green":
            return True
        elif q == 3 and self.colour == "red":
            return True
        elif r + q == -3 and self.colour == "blue":
            return True

        return False

    def getStepJumpMoves(self):
        for q in range(-3, 4):
            for r in range(-3, 4):
                if abs(q + r) < 4:
                    self.stepMovesFromCoordinate[(q,r)] = self.getStepMovesFromCoordinate((q,r))
                    self.jumpMovesFromCoordinate[(q,r)] = self.getJumpMovesFromCoordinate((q,r))
