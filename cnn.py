import os

import keras
from chexers import *

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers

import config
from loss import softmaxCrossEntropyWithLogits


"""
    Heavily based on this implementation:
    https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning
"""

class CNN:

    def __init__(self, modelOutputFile):

        self.modelOutputFile = modelOutputFile

        if os.path.isfile(modelOutputFile):
            # If we have an existing model then we want to refine its weights
            self.model = load_model(modelOutputFile,
									custom_objects={'softmaxCrossEntropyWithLogits': softmaxCrossEntropyWithLogits})
        else:
            # Otherwise we start from scratch
            input = Input(shape=(7,7,6))
            x = self.convLayer(input, config.FILTERS, config.KERNEL_SIZE)

            for i in range(config.HIDDEN_LAYERS-1):
                x = self.residualLayer(x, config.FILTERS, config.KERNEL_SIZE)

            valueHead = self.valueHead(x)
            policyHead = self.policyHead(x)

            self.model = Model(inputs=[input], outputs=[valueHead, policyHead])
            self.model.compile(
                                loss={
                                    "valueHead": "mean_squared_error",
                                    "policyHead": softmaxCrossEntropyWithLogits
                                },
                                optimizer=SGD(
											lr=config.LEARNING_RATE,
                                			momentum=config.MOMENTUM),
                                loss_weights={"valueHead":0.5, "policyHead": 0.5}
                                )
            self.model.save(self.modelOutputFile)

    def evaluate(self, boardState, colour, exits):
        """
            Takes the current game state, converts it into a form that can be
            inputted into our NN and pushes it through the NN and extracts a p
            vector and a v value.
        """

        nninput = boardStateToNNInput(boardState, colour, exits)
        v, p = self.model.predict(nninput.reshape((-1,7,7,6)))

        return v[0][0], p[0]

    def train(self, data, winner, moveNum):
        """
            Takes in game data 'data' and fits the NN to it.

            Data is an array containing elements of the form:
            (moveProbs, BoardState)

            recentBoardHistory is an array of boardPositions in chronological order
        """

        # Ready up our input data first
        X = []
        y = {"valueHead": [], "policyHead": []}

        r = {"red": 0, "green": 0, "blue": 0}
        if winner and winner != "draw":
            for player in r.keys():
                if player == winner:
                    r[player] = 0.8 + 0.2*(1 - moveNum/config.MAX_MOVES) # Higher reward if win took less moves
                else:
                    r[player] = -0.8 - 0.2*(moveNum/config.MAX_MOVES) # Higher penalty if loss took less moves

        currentPlayer = "red"
        for moveNum in range(0,len(data)):
            if data[moveNum]:

                moveProbs = data[moveNum][0]

                y["valueHead"].append(r[currentPlayer])
                y["policyHead"].append(moveProbs)

                xInput = boardStateToNNInput(data[moveNum][1].counterPositions,currentPlayer,data[moveNum][1].exits)
                X.append(xInput)

            currentPlayer = getNextColour(currentPlayer)

        y["valueHead"] = np.array(y["valueHead"])
        y["policyHead"] = np.array(y["policyHead"])

        # Update the net!
        self.model.fit(np.array(X).reshape((-1,7,7,6)), y, epochs=1, verbose=1, validation_split=0)

    def save(self):
        """
            Saves the NN's weights and structure in .h5 format for future reuse.
        """
        self.model.save(self.modelOutputFile)

    def convLayer(self, input, filters, kernelSize):
        """
            A convolutional layer in the neural network
        """

        x = Conv2D(filters=filters,
                   kernel_size=kernelSize,
                   data_format="channels_first",
                   activation="linear",
                   use_bias=False,
				   padding="same",
                   kernel_regularizer=regularizers.l2(config.REG_CONST),
                   kernel_initializer='random_uniform',
                   bias_initializer='zeros'
                   )(input)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def residualLayer(self, input, filters, kernelSize):
        """
            A residual layer, which forms one of the many hidden layers in our NN
        """

        x = self.convLayer(input, filters, kernelSize)

        x = Conv2D(filters=filters,
                  kernel_size=kernelSize,
                  data_format="channels_first",
                  use_bias=False,
                  activation="linear",
				  padding="same",
                  kernel_regularizer=regularizers.l2(config.REG_CONST),
                  kernel_initializer='random_uniform',
                  bias_initializer='zeros'
                  )(x)

        x = BatchNormalization(axis=1)(x)
        x = add([input, x])
        x = LeakyReLU()(x)

        return x

    def valueHead(self, input):
        """
            Value head of the NN - outputs a single value, the V estimate
        """

        x = Conv2D(filters=1,
                kernel_size=(1,1),
                data_format="channels_first",
                use_bias=False,
                activation="linear",
				padding="same",
                kernel_regularizer=regularizers.l2(config.REG_CONST),
                kernel_initializer='random_uniform',
                bias_initializer='zeros'
                )(input)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(20,
              use_bias=True,
              activation="linear",
              kernel_regularizer=regularizers.l2(config.REG_CONST),
              kernel_initializer='random_uniform',
              bias_initializer='zeros'
              )(x)

        x = LeakyReLU()(x)

        x = Dense(1,
              use_bias=False,
              activation="tanh",
              kernel_regularizer=regularizers.l2(config.REG_CONST),
              kernel_initializer='random_uniform',
              bias_initializer='zeros',
              name="valueHead"
              )(x)

        return x

    def policyHead(self, input):
        """
            Policy head for our NN - outputs a vector of probabilities for each move type

			Output vector p has dimensions 457x1 containing probability values.
        """

        x = Conv2D(filters=2,
                  kernel_size=(1,1),
                  data_format="channels_first",
                  use_bias=False,
                  activation="linear",
				  padding="same",
                  kernel_regularizer=regularizers.l2(config.REG_CONST),
                  kernel_initializer='random_uniform',
                  bias_initializer='zeros'
                  )(input)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(config.P_OUTPUT_DIMS,
                use_bias=False,
                activation="softmax",
                kernel_regularizer=regularizers.l2(config.REG_CONST),
                kernel_initializer='random_uniform',
                bias_initializer='zeros',
                name="policyHead"
                )(x)

        return x
