# Work Log

## May 15
- _Convext Optimization_, Boyd & Vandenberghe, Chapter 1, Introduction
- _BV_ Appendix A, Mathematical Background
- _BV_ Chapter 2, Convex Sets and Topology

## May 16
- _BV_ Chapter 3, Convex Functions
- _BV_ Chapter 6, Approximation and Fitting
- _Machine Learning_, Murphy, Chapter 1, Introduction

## May 17
- CS 229, Lecture 1, Supervised Learning, Discriminative Algorithms ([Course Website](http://cs229.stanford.edu/))
- _Murphy_, Chapter 7, Regression
- CS 229, Lecture 2, Generative Algorithms

## May 18
- _BV_, Chapter 4, Convex Optimization Problems
- _BV_, Chapter 7, Statistical Estimation
- CS 229, Lecture 3, Support Vector Machines

## May 19
- _BV_, Chapter 8 , Geometric Problems
- _Murphy_, Chapter 8, Logistic Regression
- CS 229, Lecture 4, Learning Theory

## May 22
- CS 229, Lecture 5, Regularization and Model Selection
- _BV_, Chapter 9, Unconstrained Minimization
- _BV_, Chapter 10, Equality Constrained Minimization

## May 23
- CS 229, Lecture 6, Online Learning and Perceptron Algorithm
- CS 229, Lecture 7a, Unsupervised Learning and k-Means
- _Murphy_, Chapter 9, Generalized Linear Models

## May 24
- _BV_, Chapter 11, Interior Point Methods
- CS 229, Lecture 7b, Mixture of Gaussians
- _Murphy_, Chapter 2, Probability Theory

## May 30
- Fang and Radcliffe, "On the Spectra of General Random Graphs". ([link to paper](http://www.math.ucsd.edu/~fan/wp/randomsp.pdf))
  
  Paper establishes bounds on the eigenvalues of the adjacency and normalized Laplacian matrices of graphs where each edge is determined by an independent random variable. While not exactly the case for the expectation of a time-varying graph, the approach is similar. However, the paper uses the probability expected adjacency matrix, whereas we plan on rounding the probabilities for a genuine adjacency matrix.
  
  Based on previous work, especially by Oliveira in "Concentratoin of the adjacency matrix and of the Laplacian in random graphs with independent edges. ([link to paper](https://arxiv.org/pdf/0911.0600.pdf))

- Ding and Jiang, "Spectral Distributions of Adjacency and Laplacian Matrices of Random Graphs" ([link to paper](https://arxiv.org/pdf/1011.2608.pdf))
  
  Paper does not deal with expected values of eigenvalues or Laplacian matrices, but does have interesting convergence results for the distribution of eigenvalues of random matrices as the degree of the graph approaches infinity. 
  
## May 31
- Wrote Python wrapper script for getting eigenstuff with numpy

## June 2
- Finalized Python script for retrieving matrix values from CSV file

## June 3
- Worked through preliminary example (picture to be uploaded)

## June 4
- Worked through another small example of time-varying network

## June 6
- Example of network with few edges
- Readings on the spectral properties of the Laplacian ([link to PDF](http://www.sciencedirect.com/science/article/pii/S0898122104003074))

## June 10
- Research various graph implementations and representations in Python
- Two good libraries available: [igraph](http://igraph.org/python/) and [NetworkX](https://networkx.github.io/)
- Both support large networks, visualizations, analysis, etc.

## June 11
- Found a very useful thesis paper from UMich about large-scale single and multiple graphs ([link to paper](https://web.eecs.umich.edu/~dkoutra/Danai_Koutra_thesis_CMU-CS-15-126.pdf))
- TimeCrunch algorithm for summarizing temporal graphs by Shah et al ([link to paper](https://www.cs.cmu.edu/~neilshah/research/papers/TimeCrunch.KDD.2015.pdf))
- Algorithm for determining coherent patterns in dynamic networks, and condensing large-scale graphs into important temporal structures
- Started implementing automated graph and Laplacian generation using both igraph and NetworkX
- NetworkX is slightly more convenient as it has built-in integration with numpy and linear algebra methods
- Tutorial on Modeling and Analysis of Dynamic Social Networks ([link to paper](https://arxiv.org/pdf/1701.06307.pdf))
- Another relevant thesis, though it is translated from German, so translation issues occur ([link to paper](http://www.iiserkol.ac.in/~anirban.banerjee/Banerjee_PhD_Thesis.pdf))

## June 16
- Began implementing time-varying network methods with NetworkX in Python 2.7
