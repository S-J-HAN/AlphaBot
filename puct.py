import config

def PUCT(boardState):
    """
        Simple implentation of polynomial upper confidence trees (PUCT)
        evaluation algorithm for MCTS. This is straight from the AlphaZero Nature
        paper (Silver et al., 2017). It's pretty hard to find information about
        puct online so this might not really be correct. We will have to wait
        and see.
    """
    return boardState.Q + config.PUCT_C*boardState.prior/(boardState.N + 1)
