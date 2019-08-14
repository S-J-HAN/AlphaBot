# AlphaBot

Chexers is a three player hex-based strategic board game that centres around capturing, posturing and movement. Think of it as Chinese checkers meets American checkers. The game was created (I believe) by Matt Farrugia for COMP30026 Artificial Intelligence, Semester 1 2018. This project was also written for that subject.

What you'll find now in this repo is a *functional* version of the AlphaZero algorithm adapted for the game of chexers. This includes a five-deep convolutional neural net implemented in Keras and a custom MCTS module for searches. While the bot trains and everything runs well, we never really got to the point of it playing well because of hardware and time constraints - as it turns out, Google Deepmind's AlphaZero bot was trained using more than FIVE THOUSAND TPUs in parallel. We were considering doing that ourselves until we looked in our bank accounts and realised that we weren't Google.

Because of this, there is a very decent probability that this program doesn't actually train properly, even with adequate hardware. In the near future, this repo will be updated to a) become modular and easy to train on any kind of deterministic sequential game and b) become working, at least for smaller games with much smaller search trees like tic tac toe.

A report is also included that documents our struggle and lessons from the project.
