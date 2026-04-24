# Graph-Based Recommendation System

## 1. Overview

This document reframes the project as an architecture-first graph recommendation system.

The core idea is simple:

- users and items form a bipartite graph,
- interactions become edges,
- the model learns user and item embeddings from that graph,
- recommendations are produced by scoring user-item pairs.

The focus here is not exhaustive theory. The focus is how the system is structured, what each layer is responsible for, and how the parts fit together.

## 2. Problem Framing

We want to recommend the top-`K` items each user is most likely to interact with next.

Examples:

- movies for viewers,
- products for customers,
- songs for listeners,
- articles for readers.

We assume implicit feedback for the first version:

- click,
- watch,
- purchase,
- listen,
- wishlist add.

That makes the project a ranking problem over a user-item graph.

## 3. Graph Formulation

The base representation is a bipartite graph:

- user nodes `U`,
- item nodes `I`,
- interaction edges `E subseteq U x I`.

Formally:

- `G = (V, E)`
- `V = U union I`

The equivalent adjacency structure is:

```text
A = [ 0  R ]
    [ R^T 0 ]
```

where `R` is the user-item interaction matrix.

This graph is enough for a first recommender because it already captures collaborative filtering structure:

- users connect through shared items,
- items connect through shared users,
- multi-hop neighborhoods expose similarity patterns.

## 4. Architecture At A Glance

The system should be treated as a small recommendation pipeline with five layers:

1. data ingestion
2. graph construction
3. embedding model
4. ranking and evaluation
5. experiment orchestration

High-level flow:

```text
Raw interactions
    -> preprocessing and ID mapping
    -> bipartite graph builder
    -> train/validation/test split
    -> negative sampler
    -> recommender model
    -> user-item scoring
    -> top-K ranking
    -> evaluation metrics
```

## 5. System Components

### 5.1 Data Ingestion Layer

Responsibilities:

- load raw interaction data,
- normalize schema,
- map raw user/item IDs to contiguous integer indices,
- store lookup tables for reverse decoding,
- compute basic dataset statistics.

Input:

- interaction records such as `(user_id, item_id, timestamp, rating?)`

Output:

- indexed interaction table,
- user mapping,
- item mapping,
- optional side-feature tables.

For the first implementation, MovieLens 100K is a good fit because it is small, familiar, and enough to validate the full pipeline.

### 5.2 Graph Construction Layer

Responsibilities:

- create the bipartite graph,
- build adjacency structures,
- generate edge index tensors,
- optionally attach edge weights such as rating or interaction strength.

This layer is the boundary between recommendation data engineering and graph learning.

The initial version can stay simple:

- unweighted edges,
- one edge per observed interaction,
- no extra node types.

Possible later extensions:

- genre nodes,
- category nodes,
- tag nodes,
- item-item similarity edges,
- user-user relation edges.

### 5.3 Split And Sampling Layer

Responsibilities:

- split interactions into train, validation, and test sets,
- avoid leakage,
- generate negative samples for ranking loss,
- prepare mini-batches if needed.

Recommended evolution:

- start with a simple holdout split,
- move to temporal split once the training pipeline is stable.

Negative sampling is necessary because implicit feedback provides positives but not explicit negatives.

### 5.4 Embedding Model Layer

This is the core architecture layer.

The model learns:

- a user embedding matrix,
- an item embedding matrix,
- optional propagated graph embeddings.

The recommended progression is staged.

#### Baseline A: Popularity

Purpose:

- sanity check,
- verify evaluation pipeline,
- establish a floor.

#### Baseline B: Matrix Factorization

Architecture:

- learn user embedding `P`,
- learn item embedding `Q`,
- score with dot product.

Scoring:

`score(u, i) = p_u^T q_i`

This baseline isolates the value of latent factors without graph propagation.

#### Baseline C: Basic Graph Recommender

Architecture:

- initialize node embeddings,
- propagate information across the interaction graph,
- extract user and item representations,
- score candidate pairs with dot product.

Generic message passing view:

`h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u in N(v)}))`

Interpretation:

- users absorb signal from interacted items,
- items absorb signal from connected users,
- deeper layers encode higher-order collaborative structure.

#### Target Model: LightGCN

LightGCN is a strong architecture choice for this project because it keeps only the parts that matter most for collaborative filtering:

- embedding lookup,
- normalized neighborhood propagation,
- layer aggregation,
- dot-product scoring.

Propagation:

`E^(k+1) = D^(-1/2) A D^(-1/2) E^(k)`

Final embedding:

`E = sum_k alpha_k E^(k)`

Why this architecture fits well:

- simpler than feature-heavy GNNs,
- aligned with recommendation tasks,
- easier to explain and implement cleanly,
- strong enough to be meaningful in experiments.

### 5.5 Training Layer

Responsibilities:

- coordinate forward pass,
- generate positive-negative pairs,
- compute loss,
- update parameters,
- log metrics,
- save checkpoints.

Preferred training objective for graph recommendation:

- pairwise ranking loss,
- especially Bayesian Personalized Ranking (BPR).

BPR objective:

`L = -sum log sigma(score(u, i_pos) - score(u, i_neg))`

This matches the actual product behavior better than plain binary classification because the system ultimately returns ranked lists.

### 5.6 Retrieval And Ranking Layer

Responsibilities:

- compute scores for candidate items,
- mask training-seen items when required,
- rank by score,
- return top-`K` recommendations.

In this project, the learned graph recommender can act as:

- the full ranking model for offline experiments, or
- a candidate generation block in a larger production-style system.

### 5.7 Evaluation Layer

Responsibilities:

- run offline ranking evaluation,
- compare baselines,
- report reproducible metrics.

Core metrics:

- Recall@K
- Precision@K
- Hit Rate@K
- NDCG@K

These are more useful than plain accuracy because recommendation is fundamentally a ranking task.

## 6. Reference Architecture

The repository can be organized around clear architectural boundaries:

- `src/gnn_project/data/recommendation/`
- `src/gnn_project/models/recommendation/`
- `src/gnn_project/training/recommendation/`
- `configs/recommendation/`
- `experiments/recommendation/`
- `reports/recommendation/`

Suggested module responsibilities:

### `data/recommendation`

- dataset loaders
- ID encoders
- split builders
- graph builders
- negative samplers

### `models/recommendation`

- popularity baseline
- matrix factorization
- basic graph recommender
- LightGCN

### `training/recommendation`

- trainers
- loss functions
- ranking evaluators
- logging utilities

### `configs/recommendation`

- dataset config
- model config
- training config
- evaluation config

### `experiments/recommendation`

- reproducible experiment entry points
- ablation runs
- hyperparameter comparisons

### `reports/recommendation`

- metric tables
- plots
- short architecture notes
- result summaries

## 7. Model Data Flow

A clean implementation should follow this flow:

1. Load raw interaction data.
2. Encode user and item IDs.
3. Build train, validation, and test splits.
4. Construct the bipartite graph from training interactions.
5. Initialize user and item embeddings.
6. Propagate embeddings through the graph.
7. Sample negatives during training.
8. Score positive and negative user-item pairs.
9. Optimize ranking loss.
10. Generate top-`K` recommendations at evaluation time.
11. Report Recall@K and NDCG@K against baselines.

This sequence is important because it keeps graph construction, learning, and evaluation clearly separated.

## 8. Design Choices

### Keep The Graph Minimal First

Start with:

- user nodes,
- item nodes,
- observed interaction edges.

Do not begin with heterogeneous graph complexity unless the base pipeline is already stable.

### Prefer Baselines Before GNN Depth

The architecture is only convincing if it beats or complements simpler methods.

Minimum comparison set:

- popularity,
- matrix factorization,
- LightGCN or a basic graph recommender.

### Use Shallow Propagation

Recommendation graphs are vulnerable to over-smoothing.

A shallow architecture is usually enough:

- 1 to 3 propagation layers,
- layer-wise embedding aggregation,
- direct dot-product scoring.

### Treat Time Carefully

If timestamps exist, the more realistic evaluation setup is:

- train on past interactions,
- validate on later interactions,
- test on future interactions.

That turns the architecture into something closer to a deployable recommender rather than a classroom-only exercise.

## 9. Risks That Affect Architecture

### Cold Start

Pure graph structure struggles when a user or item has no edges.

Architectural implication:

- leave room for side features or hybrid embeddings later.

### Popularity Bias

Popular items can dominate training and retrieval.

Architectural implication:

- evaluate beyond raw relevance,
- consider reranking or debiasing later.

### Sparsity

User-item graphs are usually sparse.

Architectural implication:

- graph propagation is useful because it spreads signal beyond direct edges.

### Data Leakage

Improper splits can make the system look better than it is.

Architectural implication:

- splitting logic should be a first-class module, not an afterthought.

## 10. Recommended Build Order

1. Implement the data ingestion and ID mapping layer.
2. Build the interaction graph and split pipeline.
3. Add a popularity baseline.
4. Add matrix factorization.
5. Add ranking metrics.
6. Implement a basic graph recommender or LightGCN.
7. Compare results in a short report.
8. Extend with temporal split or side information if needed.

## 11. Final Direction

The best first version of this project is a lean, architecture-driven graph recommender:

- MovieLens 100K as the dataset,
- bipartite user-item graph as the data model,
- matrix factorization and popularity as baselines,
- LightGCN as the main graph architecture,
- Recall@K and NDCG@K as the main evaluation metrics.

That scope is small enough to finish, but structured enough to grow into a stronger recommendation system later.
