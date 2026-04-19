# GNN Mini Project

This repository is organized for an academic, concept-first mini project on Graph Neural Networks (GNNs).

Our workflow will be:

1. Study the concept.
2. Understand why we use it.
3. Review the core math behind it.
4. Point to good learning resources.
5. Implement a small, clear version in code.
6. Reflect on results and limitations.

## Project Goals

- Learn the foundations of graph-based machine learning.
- Build small GNN components step by step instead of jumping straight to a full model.
- Keep theory notes close to the code so the project is useful for coursework, reports, and revision.

## Model Architecture

The main model direction for this repository is a small message-passing GNN, with the first concrete baseline being a Graph Convolutional Network (GCN)-style architecture for node-level learning.

At a low level, the model takes three core inputs:

- `X`: node feature matrix with shape `(num_nodes, num_node_features)`
- `A`: adjacency information describing which nodes are connected
- `y`: labels for the supervised task, usually node labels or graph labels depending on the experiment

The basic idea is that each layer updates a node representation by combining:

- the node's own current features
- information aggregated from its immediate neighbors
- a learned linear transformation that maps the aggregated signal into a new feature space

For a standard GCN layer, we usually work with:

- `A_hat = A + I`
- `D_hat`: degree matrix of `A_hat`
- normalized propagation matrix `D_hat^(-1/2) A_hat D_hat^(-1/2)`

The layer update is:

```text
H^(l+1) = sigma(D_hat^(-1/2) A_hat D_hat^(-1/2) H^(l) W^(l))
```

Where:

- `H^(l)` is the node embedding matrix at layer `l`
- `H^(0) = X`
- `W^(l)` is the learnable weight matrix for layer `l`
- `sigma` is a nonlinearity such as ReLU

This means one layer does four conceptual operations:

1. Add self-loops so a node can keep part of its own information.
2. Normalize the graph structure so high-degree nodes do not dominate aggregation.
3. Mix neighbor features according to the graph connectivity.
4. Apply a learnable projection and nonlinearity.

In practical model terms, the baseline architecture we are building toward is:

```text
Input node features
-> graph convolution layer
-> nonlinearity
-> optional dropout
-> graph convolution layer
-> task head
```

For node classification, the task head is usually:

- a final linear projection to `num_classes`
- followed by softmax during evaluation or cross-entropy during training

For graph classification, we would insert a graph-level readout after the node encoder, such as:

- global mean pooling
- global sum pooling
- global max pooling

Then we would send the pooled graph embedding into a classifier.

## Low-Level Design Discussion

At the implementation level, we want to keep the architecture modular so each stage is easy to inspect:

- `data/` handles graph loading, preprocessing, feature preparation, and train/validation/test splits
- `layers/` will contain reusable message-passing or graph convolution layers
- `models/` will compose layers into full GNN architectures
- `training/` will own loss computation, optimization, metrics, and evaluation loops
- `utils/` will hold shared helpers such as seeding, logging, and tensor utilities

This separation matters because in GNN work, the model's behavior is shaped by both:

- neural network parameters
- graph preprocessing decisions

For example, changing self-loops, normalization, edge direction handling, or node feature scaling can materially change results even if the Python class for the model stays the same.

From a tensor-flow perspective, the forward pass should be easy to reason about:

- start with node features `X`
- propagate information across edges
- produce hidden node embeddings
- optionally repeat propagation for deeper receptive fields
- map embeddings to logits for the final task

After one graph convolution layer, each node embedding contains information from its 1-hop neighborhood. After two layers, each node embedding contains information from roughly its 2-hop neighborhood. This is one reason shallow GNNs are often effective, but it also explains why deeper stacks can cause over-smoothing, where node embeddings become too similar.

For this project, a small 2-layer GCN is a good baseline because it is:

- mathematically simple
- strongly connected to the original message-passing intuition
- easy to debug in notebooks
- a useful reference point before trying more advanced architectures such as GraphSAGE, GAT, or GIN

## What The Model Is Learning

The trainable part of the model is not the adjacency matrix itself. The graph structure tells the network where messages are allowed to flow. The learnable part is mainly:

- per-layer weight matrices
- optional bias terms
- any classifier head parameters

So the graph defines the communication pattern, while the learned weights define how information should be transformed at each step.

## Structure

- `docs/`: theory notes, math explanations, roadmap, and references.
- `src/gnn_project/`: reusable Python package for datasets, layers, models, training, and utilities.
- `notebooks/`: guided experiments and visual exploration.
- `experiments/`: runnable experiment scripts and logs.
- `data/`: raw and processed datasets.
- `reports/`: figures and short writeups for academic reporting.
- `tests/`: lightweight tests as the codebase grows.
- `configs/`: experiment configuration files.

## Commit Message Guide

When writing Git commits for this project, use the commit prefix guide here:

- `docs/git-commit-prefixes.md`

That document explains which prefixes to use, when to use them, and gives examples such as `feat:`, `fix:`, `docs:`, `perf:`, `ci:`, and `revert:`.

## How We Will Work

For each topic, we will keep the same pattern:

- `Concept`: what it means.
- `Why it matters`: why we apply it in GNNs.
- `Math`: the main equations and intuition.
- `Implementation`: a small code version.
- `Resources`: papers, articles, or videos worth reading.

## Suggested First Topics

- Graph basics: nodes, edges, adjacency matrix, degree matrix.
- Message passing.
- Graph convolution.
- Node classification.
- Over-smoothing and limitations of shallow/deep GNNs.

## Next Step

We can next choose one of these:

1. Set up the Python environment and dependencies.
2. Start with graph fundamentals and adjacency matrix math.
3. Build the first notebook for a toy graph example.
