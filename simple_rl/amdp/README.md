# Abstract Markov Decision Processes (AMDP)

Implementation of planning framework described in *Planning with Abstract Markov Decision Processes (Gopalan et al)*

## Abstract classes
This implementation of AMDPs defines the following set of abstract (base) classes:
* `AMDPTaskNodesClass.py`: Each node in the AMDP task hierarchy is represented as a `TaskNode` object, with the option
   of being either a `PrimitiveAbstractTaskNode`, `NonPrimitiveAbstractTaskNode` or a `RootAbstractTaskNode`. Any AMDP 
   domain you implement must have a class describing grounded actions in that class that inherit from the base clasess 
   in this file
* `AMDPStateMapperClass.py`: Each node in the task hierarchy of an AMDP hands down a terminal function and reward function
    to its children. The MDP domain must query the level above it to check if it has satisfied the subgoal handed down to
    it. However, the higher level domain shouldn't have to reason about low level states. 
    The state projection function `F: S -> S'` projects low level state `S` to high level state `S'`
* `AMDPSolverClass.py`: Generic solver class for all AMDP domains
* `AMDPPolicyGeneratorClass.py`: abstract definition of how we can generate a policy for each level in the AMDP
    hierarchy on the fly
    
 ## Example Domains
 
 I have included the following example AMDP domain implementations:
 ### Abstract Four Room Domain
 The classic grid world domain is partitioned into four rooms. The grid cell location represents the L0 state whereas 
 the room number represents the L1 state. Starting from some room, the agent is given the goal of making its way to
 another room via a high level task description such as 'toRoom4'
 
### Abstract Taxi Domain
In this OOMDP domain, a taxi agent must pickup a passenger from some location and dropoff at a specified location. The 
grid cell locations of the taxi and passenger define the L0 state whereas the color of the grid defines the L1 state description. 
Given a high level description such as 'completeRide', the taxi agent must hierarchichally plan to first navigate to the
passenger, pick them up, navigate to the destination and dropoff. 

### Requirements for defining your own AMDP domain
Let us consider the Abstract Taxi Domain as an example of how to use the AMDP interface. 
* `AbstractTaxiMDPClass`: This class defines the MDPs at all levels of hierarchy -- including the transition functions, 
    reward functions, and high level actions
* `AbstractTaxiStateMapperClass`: You need to define how to project each lower level state to one level higher in the AMDP hierarchy
* `AbstractTaxiPolicyGeneratorClass`: Given a subgoal (represented by a higher level task node) on the fly, the policy generators
    must define how to generate a policy over the sub-mdp
   
### Coming soon
I am currently working on the following AMDP domains:
* Multi-passenger taxi domain
* Cleanup domain (L2 layer) 
* Propositional functions

If you could benefit from a specific domain implementation, let me know and I will try to add support for it here.

### Questions and suggestions
This is very much work in progress. If you have any suggestions for improvements, please let me
know and I will do my best to address them! You can email me at `akhil_bagaria@brown.edu`

 



