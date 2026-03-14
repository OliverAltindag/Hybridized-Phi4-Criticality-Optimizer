## Hybridized-Phi4-Criticality-Optimizer

A percolation-physics rooted optimization tool for the Phi4 Model, successfully recovering the industry-standard values (https://arxiv.org/pdf/2512.16536). The model is based on the logic of the Invaded Cluster Algorithm (Machta, 1995), leveraging the notion that a system is critical when percolation occurs, but adapted to serve as an optimization tool. The advantage of this model is its topological awareness and its homing towards criticality at each Monte Carlo step; it also provides a significant speedup by measuring critical components natively rather than creating an ensemble of traditional measurements (20x+).

The framework provided here can easily be extended to other models; however, the code provided is strictly for the Phi4 model and is not production-optimized.
