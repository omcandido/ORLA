# ORLA: Online Reinforcement Learning Argumentation
ORLA is a symbolic RL approach that uses arguments provided by an expert as rules and learns which rules prevail over others via RL. Ultimately, ORLA learns a ranking over the arguments, which can be transformed into a standard value-based argumentation framework (VAF).

## Original paper and citation
[ORLA: Learning Explainable Argumentation Models](https://doi.org/10.24963/kr.2023/53)
```
@inproceedings{KR2023-53,
    title     = {{ORLA: Learning Explainable Argumentation Models}},
    author    = {Otero, Cándido and Craandijk, Dennis and Bex, Floris},
    booktitle = {{Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning}},
    pages     = {542--551},
    year      = {2023},
    month     = {8},
    doi       = {10.24963/kr.2023/53},
    url       = {https://doi.org/10.24963/kr.2023/53},
  }
```

## Getting started
To adapt ORLA to your reinforcement learning (RL) task, you will need to do two things:
1. Create an expert argumentation framework (AF) for your task.
2. Extend the `Environment` class, where you will have to implement (at least) the methods:
    - ``get_premises``: from the current observation and memory, which premises hold?
    - ``get_arguments``: from the current premises, which arguments are applicable?
    - ``update_memory``: how should memory be updated at each step?
    - ``reset_memory``: how should memory be initialised?

These methods are the canonical way in which a value-based AF (VAF) is derived from the AF and used as an inference engine for your task.

We provide two implementation examples:
- **Foggy Frozen Lake (FFL)**: a variant of the classic Frozen Lake game. The [FFL](src/FFL.ipynb) notebook introduces ORLA, the FFL game, how the AF is built, how an `Environment` class is created and showcases ORLA by putting all the pieces together and solving the FFL game. This is the recommended example to start with. FFL was used to test the predecessor of ORLA (more info on my master's thesis [Explainable Online Reinforcement Learning Using Abstract Argumentation](https://studenttheses.uu.nl/handle/20.500.12932/43012)).

- **Takeaway**: a soccer-like game where takers must gain possession of the ball and keepers must prevent it. The [takeaway](src/takeaway.ipynb) notebook explains the game, the AF and how ORLA is trained. Note that this example does not follow the canonical way to create an `Environment`, since all the inference is done inside the soccer simulator (via [keepaway_orla](https://github.com/omcandido/keepaway_orla)).

## Results
The final results for Takeaway (shown in the [paper](https://doi.org/10.24963/kr.2023/53)) can be found in the [analysis notebook](results/analysis.ipynb).

## Playing Takeaway with ORLA
Takeaway is implemented using the RoboCup Soccer Simulator (rcssserver) as the world and the keepaway library as an interface between the simulator and the learning agent. To connect ORLA with RoboCup, it is necessary to install [rcssserver_orla](https://github.com/omcandido/rcssserver_orla) and [keepaway_orla](https://github.com/omcandido/keepaway_orla). These two repositories contain all the adjustments needed for ORLA to interact with RoboCup and play Takeaway.
