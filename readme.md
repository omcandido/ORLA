# ORLA: Online Reinforcement Learning Argumentation
ORLA is a symbolic RL approach that uses arguments provided by an expert as rules and learns which rules prevail over others via RL. Ultimately, ORLA learns a ranking over the arguments, which can be transformed into a standard value-based argumentation framework (VAF).

## Guide
To get familiar with ORLA, check out the [takeaway notebook](src/takeaway.ipynb), where the Takeaway game is used as case study to show case the use of ORLA.

## Results
The final results are shown in the [analysis notebook](results/analysis.ipynb).

## Playing Takeaway with ORLA
Takeaway is implemented using the RoboCup Soccer Simulator (rcssserver) and using the keepaway library as an interface between the simulator and the learning agent. To connect ORLA with RoboCup, it is necessary to install [rcssserver_orla](https://github.com/omcandido/rcssserver_orla) and [keepaway_orla](https://github.com/omcandido/keepaway_orla). These two repositories contain all the adjustments needed for ORLA to interact with RoboCup and play Takeaway.