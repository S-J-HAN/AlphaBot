from chexers import *
import config
from puct import PUCT

class MCTS:

    def __init__(self, rootState, numSims, gameNum, moveNum):
        self.rootState = rootState # of type BoardState
        self.numSims = numSims

        self.gameNum = gameNum
        self.moveNum = moveNum

    def search(self):
        """
            Performs a search of possible next moves and returns a vector π,
            a probability distribution for the most appropriate next moves where
            a higher value = better move.

            The tree itself is first built by running self.sim() many times.

            π is a 457x1 vector - each value represents the probability of a
            player making a specific move. 457 is obtained by multiplying 37 by
            12, then adding 12 again, then 1. The rational for this is that there
            are 37 hexes and a maximum of 12 possible moves from each hex. The
            addition of another 12 units comes from the fact that there are 12
            hexes that can be exited from, and the final unit represents a pass
            move. This of course is a pretty imprecise way of representing moves,
            as there on average far fewer than 12 moves per hex, but it simplifies
            things by alot so it's good enough.
        """

        # Build our tree using many simulations
        for i in range(0, self.numSims):

            # Check if the current player can only pass - we'll just skip this search in that case
            if i == 1:
                if len(self.rootState.nextMoves.keys()) == 1 and list(self.rootState.nextMoves.keys())[0][0] == "PASS":
                    searchProbs = np.zeros(config.P_OUTPUT_DIMS)
                    searchProbs[config.P_OUTPUT_DIMS - 1] = 1
                    return searchProbs
                    break

            self.sim()

            if i%100==0:
                print("game {}     move {}     simulation {}       player {}".format(self.gameNum, self.moveNum, i, self.rootState.colour))

        # Create probability distribution of moves
        searchProbs = np.zeros(config.P_OUTPUT_DIMS)
        totalProbVal = sum([c.N**(1/config.T) for c in self.rootState.nextMoves.values()])

        if totalProbVal == 0:
            totalProbVal = 1
        for move in self.rootState.nextMoves.keys():
            nextState = self.rootState.nextMoves[move]
            probVal = (nextState.N**(1/config.T))/totalProbVal
            searchProbs[encodeMove(move)] = probVal

        return searchProbs

    def sim(self):
        """
            A single pass of sim() builds up the MCTS tree. Traverses down the
            existing tree first by choosing child nodes with max PUCT scores.
            Upon hitting a leaf node, sim() expands the leaf node and updates
            the Q value for each of its ancestors.
        """
        # Traverse down the tree first
        currentState = self.rootState
        currentState.N += 1
        while currentState.nextMoves.keys():
            currentState = currentState.getNextState()
            currentState.N += 1

        # Expand our leaf node
        currentState.expand()

        # Update Q values
        parentState = currentState.parent
        while parentState != None:
            c = 1 if parentState.colour == self.rootState.colour else -1
            parentState.vSum += currentState.V*c
            parentState.Q = (1/parentState.N)*parentState.vSum
            parentState = parentState.parent
