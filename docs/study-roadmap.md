# Study Roadmap

This project is meant to support academic learning, not only implementation.

## Learning Sequence

### 1. Graph Foundations

- What is a graph?
- Directed vs undirected graphs.
- Weighted vs unweighted graphs.
- Adjacency matrix and degree matrix.
- Neighborhood aggregation intuition.

### 2. Why Standard Neural Networks Struggle on Graphs

- Graphs have irregular structure.
- Node order should not change meaning.
- Connectivity matters as much as raw features.

### 3. Message Passing

- Each node gathers information from its neighbors.
- A node updates its representation using local structure.
- Repeating this process allows wider context to flow through the graph.

### 4. Graph Convolutional Networks

- Learn the standard GCN layer.
- Study normalized adjacency.
- Understand why self-loops are often added.

### 5. A First Task

- Node classification on a small dataset.
- Train/evaluation split.
- Accuracy and interpretation.

### 6. Common Problems

- Over-smoothing.
- Over-squashing.
- Shallow receptive fields.
- Dependence on graph quality.

## How Each Topic Will Be Documented

For every topic we study, we will record:

- A plain-language explanation.
- Why the idea is used.
- The main mathematical form.
- A small implementation.
- A short list of references.
