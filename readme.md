# ORLA: Online Reinforcement Learning Argumentation
ORLA is a symbolic RL approach that uses arguments provided by an expert as rules and learns which rules prevail over others via RL. Ultimately, ORLA learns a ranking over the arguments, which can be transformed into a standard value-based argumentation framework (VAF).

## Guide
To get familiar with ORLA, check out the [takeaway notebook](src/takeaway.ipynb), where the Takeaway game is used as case study to show case the use of ORLA.

## Results
The final results are shown in the [analysis notebook](results/analysis.ipynb).

## Playing Takeaway with ORLA
Takeaway is implemented using the RoboCup Soccer Simulator (rcssserver) and using the keepaway library as an interface between the simulator and the learning agent. To connect ORLA with RoboCup, it is necessary to install [rcssserver_orla](https://github.com/omcandido/rcssserver_orla) and [keepaway_orla](https://github.com/omcandido/keepaway_orla). These two repositories contain all the adjustments needed for ORLA to interact with RoboCup and play Takeaway.

## Implementing other RL tasks
ORLA is readily compatible with new RL tasks. Here is [a demo](https://github.com/omcandido/RL-AA/blob/b8af1959c78c70eea610757ddd575c8308383eba/src/demo.ipynb) of ORLA used to learn to play Foggy Frozen Lake (FFL). Note that this demo is from a previous project.
The steps to adapt ORLA to learn new taks are:
- Create a new environment where you implement the MDP that ORLA has to optimise: this is a gym environment that acts as an interface between ORLA and the ultimate RL task you want to solve.
- Create a argument-action dictionary: the keys are the argument IDs and the values are the action each argument promotes.
- Create a function that outputs the list of applicable arguments given the current state of the environment.

#TODO: create additional demo to show step-by-step how ORLA can be learn a new RL task from scratch.
