# Foundations of Graph Neural Networks

## Concept

A graph is a structure made of:

- Nodes (also called vertices), which represent entities.
- Edges, which represent relationships between entities.

Examples:

- Social network: users are nodes, friendships are edges.
- Citation network: papers are nodes, citations are edges.
- Molecules: atoms are nodes, bonds are edges.

## Why We Use Graphs

Traditional machine learning often assumes data has a regular shape, like:

- rows in a table,
- pixels in an image,
- words in a sequence.

Graphs are different because relationships are part of the data itself. In many academic problems, the connection pattern contains important information that we do not want to ignore.

## Core Math

For a graph with `n` nodes:

- Adjacency matrix: `A in R^(n x n)`
- Degree matrix: `D in R^(n x n)`

Where:

- `A_ij = 1` if node `i` is connected to node `j`, otherwise `0` in the unweighted case.
- `D_ii = sum_j A_ij`

These matrices are central because many GNN layers use them to control how information moves between neighboring nodes.

## Intuition

A GNN updates a node representation by combining:

- the node's current features,
- information from its neighbors.

This is often called message passing or neighborhood aggregation.

## Resources

- Thomas Kipf and Max Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
- William L. Hamilton, "Graph Representation Learning"
- Stanford CS224W materials on graph machine learning

## What We Will Do Next

Before writing a GNN layer, we should first build intuition for:

- adjacency matrices,
- degree matrices,
- self-loops,
- normalized aggregation.
