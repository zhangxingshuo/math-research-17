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
  
  Based on previous work, especially by Oliveira in "Concentration of the adjacency matrix and of the Laplacian in random graphs with independent edges. ([link to paper](https://arxiv.org/pdf/0911.0600.pdf))

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

## June 17
- Added functionality for generating a random separated graph and probabilistically adding and removing edges while preserving separation

## June 18
- Finished Python script, can now generate many graphs, connected and separated, and analyze for expected Laplacian
- Retrieve and begin parsing large network datsets
- Enron: email dataset containing approximately 500,000 emails between 150 users, mostly senior management at Enron, over a period of 163 weeks ([link to dataset](https://www.cs.cmu.edu/~./enron/))
- Wikipedia: a large dataset containing user edits to Talk pages over a span of 2320 days ([link to datset](https://snap.stanford.edu/data/wiki-talk-temporal.html)) 
- DBLP: computer science bibliography containing collaboration networks and coauthorships spanning 25 years ([link to dataset](http://dblp.uni-trier.de/xml/))

## June 24
- Parse DBLP, Wikitalk, EU email, and Enron datasets
- Convert to temporal networks in NetworkX format
- Run Python scripts on generated graphs and analyze expected values and spectrum

## June 26
- Parsed EU email, Wikitalk, and CollegeMsg datasets into NetworkX .graphml format
- Ran Python script on all datasets
- The expected spectrum does not fall in between the lower and upper expected Laplacian spectrums?

## June 30
- Extract the location matrix (P such that L = PDP^-1) with using numpy and eigenvalues

## July 1
- Implement matrix average using logarithm definition

## July 5
- Use methodology on a structured example with well-defined pattern and expected graph

## July 7
- Generate a small-scale network that remains mostly connected

## July 8
- New algorithm for creating an expected Laplacian:
- Use eigendecomposition to get matrices M_i and D_i such that L_i = (M_i)(D_i)(M_i)^T, where D_i is a diagonal matrix of eigenvalues and M_i is a matrix of the corresponding eigenvectors
- Take the arithmetic mean of D_i to obtain D_e
- For each M_i, use polar decomposition into U_i and P_i, where U_i is a unitary matrix. 
- Take the arithmetic mean of P_i to obtain P_e
- Take the arithmetic mean of U_i to obtain U_bar. Define U_e to be the unitary matrix from the polar decomposition of U_bar.
- Define M_e as (U_e)(P_e)
- Define L_e as (M_e)(D_e)(M_e)^T
- Actually, taking the arithmetic of P_i is redundant, as each P_i is just the identity matrix. This is because the eigenvectors that compose each M_i are normalized to unit length. So, we can simply sum the polar decompositions of each matrix and take the polar decomposition of the sum to obtain the average rotation matrix M_e.

## July 12
- Fixed errors in dataset parsing
- Created final datasets for college-message and eu-email

## July 16
- Performed analysis of college-message and eu-email datasets
- Naive method of arithemetic mean of adjacency matrices does not work - too sparse
- Polar decomposition method works reasonable well by visual analysis, need a metric to define distance from one graph to another

## July 23
- Read and researched covariance matrices and probability theory
- Created covariance matrix for small network example
- Cleaned up code and documentation

## July 29
- Investigate large covariance matrices
- Conclusion: need to be smart about memory management
- Upload updated code to GitHub (haven't done that in a while)

## July 30
- Research previous work on graphs and covariance matrices
- A lot of work in the other direction, i.e. generating a graph from covariance matrix, but not so much in our direction
- Need to work around large dataset and array sizes
- Using brute force calculation, can generate .csv files representing large covariance matrices for big datasets
