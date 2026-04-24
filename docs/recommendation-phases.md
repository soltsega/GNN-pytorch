# Recommendation System Project Phases

## Purpose Of This Document

This document converts the recommendation-system idea into a sequence of detailed, accomplishable phases.

Each phase is designed to answer five questions clearly:

1. What are we building?
2. Why are we building it now?
3. What concepts and math must be understood first?
4. What concrete tasks should be completed?
5. What output proves the phase is finished?

The goal is to keep the project realistic. We do not want a plan that sounds impressive but becomes impossible to execute. We want a plan that is:

- academically rigorous,
- professionally structured,
- feasible for a mini project,
- strong enough to evolve into a portfolio case study.

## Phase Design Principles

The phases follow a strict logic:

- first understand the problem,
- then understand the data,
- then build simple baselines,
- then add graph modeling,
- then add stronger graph recommendation,
- then package and evaluate the work professionally.

This matters because a graph model without a reliable data pipeline and baseline is not a strong project. It becomes difficult to explain, difficult to debug, and difficult to defend in an academic or interview setting.

## Phase 1: Problem Framing And Dataset Understanding

### Objective

Define the recommendation problem clearly and understand the dataset before writing model code.

### Why This Phase Comes First

Many projects fail early because they start with architecture before understanding the learning task. Recommendation systems are not only about training a model. They begin with a precise question:

What do we want to recommend, to whom, and based on what evidence?

Before we build a graph, we must know:

- what counts as a user,
- what counts as an item,
- what counts as an interaction,
- what prediction target we care about,
- what data fields are available.

### Core Concept

A recommender system learns from historical interactions to estimate future user-item relevance.

### Why We Apply This Concept

If we do not define the problem carefully, later choices such as loss function, split strategy, and metrics become inconsistent.

### Math And Theory To Understand

Understand the interaction matrix:

`R in R^(m x n)`

Where:

- `m` is the number of users,
- `n` is the number of items,
- `R_ui` indicates whether user `u` interacted with item `i`.

For the first project version, we will treat recommendation as an implicit-feedback ranking problem.

That means:

- an observed interaction is positive,
- missing interaction is unobserved, not guaranteed negative.

This distinction is foundational for the entire project.

### Concrete Tasks

- Choose the dataset.
- Write a short problem statement.
- Inspect dataset columns and file formats.
- Identify user ID, item ID, timestamp, and optional rating fields.
- Compute basic descriptive statistics.
- Write down what the first prediction task will be.

### Suggested Deliverables

- a dataset summary note,
- a table of core statistics,
- a short document answering:
  what is the task, what is the input, what is the output, and why is this a graph problem.

### Success Criteria

Phase 1 is complete when we can clearly explain:

- the dataset schema,
- the user-item interaction meaning,
- the chosen prediction target,
- why a graph formulation is justified.

### What Not To Do Yet

- do not build GNN models,
- do not tune hyperparameters,
- do not add complex node types yet.

## Phase 2: Data Preparation And Bipartite Graph Construction

### Objective

Turn raw interaction records into a clean, indexed bipartite graph representation.

### Why This Phase Matters

The model can only be as good as the graph we build. If user IDs, item IDs, or edge construction are inconsistent, the entire recommendation system becomes unreliable.

This is the phase where recommendation data becomes graph-learning data.

### Core Concept

A bipartite graph contains two node sets:

- users,
- items.

Edges exist only between the two sets, representing observed interactions.

### Why We Apply This Concept

A bipartite graph is the simplest and most natural graph formulation for collaborative filtering. It is expressive enough for meaningful recommendation experiments without introducing unnecessary complexity.

### Math And Theory To Understand

The graph adjacency form of the user-item interaction matrix is:

```text
A = [ 0  R ]
    [ R^T 0 ]
```

This means:

- users connect to items,
- items connect back to users,
- higher-order similarity emerges through multi-hop connectivity.

### Concrete Tasks

- Map raw user IDs to contiguous integer indices.
- Map raw item IDs to contiguous integer indices.
- Build an indexed interaction table.
- Create train, validation, and test interaction splits.
- Construct the bipartite graph from training interactions only.
- Create edge-index or adjacency structures for graph learning.
- Save preprocessing metadata for reproducibility.

### Suggested Deliverables

- a preprocessing script,
- saved ID mapping tables,
- a graph-construction utility,
- a notebook or report section showing the graph statistics.

### Success Criteria

Phase 2 is complete when we can:

- reproduce the same indexed graph from raw data,
- explain how users and items were encoded,
- show the number of nodes and edges,
- confirm that no test leakage entered the training graph.

### Common Risks

- data leakage from building the graph with all interactions,
- inconsistent user/item indexing,
- losing timestamp information too early.

## Phase 3: Evaluation Design And Baseline Metrics

### Objective

Define how the recommendation system will be judged before building advanced models.

### Why This Phase Matters

A project becomes professional when evaluation is defined before the main model. Without this, it is easy to over-claim results or optimize the wrong behavior.

Recommendation is a ranking problem, so the evaluation layer must reflect ranked outputs.

### Core Concept

A recommender is successful when relevant items appear near the top of the ranked list for each user.

### Why We Apply This Concept

Loss values alone do not tell us whether the top recommendations are useful. We need ranking metrics that measure retrieval quality.

### Math And Theory To Understand

Understand the main ranking metrics:

- `Recall@K`: how many relevant items appear in the top `K`.
- `Precision@K`: what fraction of the top `K` items are relevant.
- `HitRate@K`: whether at least one relevant item appears in the top `K`.
- `NDCG@K`: rewards placing relevant items higher in the ranking.

These metrics are more meaningful than plain accuracy in recommendation settings.

### Concrete Tasks

- Define the evaluation protocol.
- Decide whether to use random split first or temporal split.
- Implement top-`K` ranking evaluation.
- Implement masking of already-seen training items during evaluation.
- Create a reusable metrics module.

### Suggested Deliverables

- evaluation utility functions,
- a document section defining each metric,
- a small sanity-check experiment on dummy scores.

### Success Criteria

Phase 3 is complete when:

- metrics can be computed correctly for a user,
- the system can evaluate a ranked list end to end,
- the evaluation protocol is documented clearly enough for replication.

### Common Risks

- evaluating on items seen during training,
- mixing classification metrics with ranking tasks,
- using leakage-prone split logic.

## Phase 4: Non-Graph Baselines

### Objective

Build simple recommenders that establish a fair reference point for the graph models.

### Why This Phase Matters

A graph recommender is only convincing if it improves on simpler methods or reveals a useful tradeoff. Baselines are not optional. They are part of the scientific validity of the project.

### Core Concept

Before learning from graph propagation, we should learn from simpler signals:

- item popularity,
- latent user-item embeddings from matrix factorization.

### Why We Apply This Concept

These baselines tell us:

- whether the pipeline works,
- whether the metrics work,
- whether a graph model is actually necessary.

### Math And Theory To Understand

#### Popularity Baseline

Recommend the items with the highest global interaction counts.

This captures a very strong signal in many systems: popular items are often widely relevant.

#### Matrix Factorization

Learn:

- user embedding matrix `P`,
- item embedding matrix `Q`.

Predict score by:

`score(u, i) = p_u^T q_i`

This learns latent preference structure without graph message passing.

### Concrete Tasks

- Implement a popularity recommender.
- Implement matrix factorization.
- Train and evaluate both baselines.
- Compare them using the same ranking metrics.
- Write down the strengths and limitations of each.

### Suggested Deliverables

- baseline model modules,
- result table for baseline performance,
- short analysis comparing popularity and matrix factorization.

### Success Criteria

Phase 4 is complete when:

- we have at least two working baselines,
- metrics are reproducible,
- we know the performance floor that the graph model must beat or justify.

### Common Risks

- skipping baselines to save time,
- comparing models with inconsistent splits,
- treating popularity as too trivial to matter.

## Phase 5: First Graph Recommender

### Objective

Build the first graph-based model that uses the user-item interaction graph directly.

### Why This Phase Matters

This is the phase where the project becomes a true graph recommendation project instead of a standard recommender system.

The goal here is not maximum performance yet. The goal is conceptual correctness and a clean end-to-end graph learning pipeline.

### Core Concept

Nodes update their representations by aggregating information from connected neighbors.

In recommendation:

- a user learns from interacted items,
- an item learns from connected users.

### Why We Apply This Concept

This lets the system model collaborative structure more explicitly than matrix factorization alone.

Two-hop propagation can capture patterns such as:

- users linked by shared items,
- items linked by shared audiences.

### Math And Theory To Understand

Generic message passing:

`h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u in N(v)}))`

Graph-convolution style propagation:

`H^(l+1) = sigma(D_hat^(-1/2) A_hat D_hat^(-1/2) H^(l) W^(l))`

Key meaning:

- adjacency controls where information flows,
- normalization prevents high-degree dominance,
- multiple layers extend the collaborative neighborhood.

### Concrete Tasks

- Create node embeddings for users and items.
- Build the graph-propagation forward pass.
- Score user-item pairs using learned embeddings.
- Train with a simple ranking objective.
- Evaluate against baselines.
- Write an explanation of what information one layer and two layers capture.

### Suggested Deliverables

- first graph recommender implementation,
- training script,
- side-by-side result comparison with non-graph baselines,
- explanation note on message passing in the recommender context.

### Success Criteria

Phase 5 is complete when:

- the graph recommender trains end to end,
- evaluation runs on the same protocol as the baselines,
- the model behavior can be explained mathematically and intuitively.

### Common Risks

- adding too many layers too early,
- making the architecture too complex to debug,
- not checking whether the model actually improves ranking behavior.

## Phase 6: LightGCN And Stronger Graph Recommendation

### Objective

Upgrade the first graph recommender into a stronger, recommendation-focused graph model.

### Why This Phase Matters

The first graph recommender proves the concept. This phase moves the project closer to the level of a serious recommendation-system case study.

LightGCN is a good choice because it is both academically meaningful and practically relevant.

### Core Concept

In collaborative filtering, simple propagation can outperform more complicated graph transformations.

### Why We Apply This Concept

LightGCN removes unnecessary operations and focuses on what matters most:

- learnable user/item embeddings,
- normalized propagation,
- layer aggregation,
- ranking-based optimization.

### Math And Theory To Understand

Propagation:

`E^(k+1) = D^(-1/2) A D^(-1/2) E^(k)`

Final embedding:

`E = sum_k alpha_k E^(k)`

Training objective:

`L = -sum log sigma(score(u, i_pos) - score(u, i_neg))`

This combines graph propagation with ranking-aware learning.

### Concrete Tasks

- Implement LightGCN propagation.
- Add negative sampling cleanly to the training loop.
- Compare multiple propagation depths.
- Compare embedding dimensions.
- Study whether performance changes with layer aggregation choices.
- Document why LightGCN is more suitable than a generic GCN for this task.

### Suggested Deliverables

- LightGCN model module,
- experiment comparison table,
- ablation note on depth and embedding size,
- technical explanation of why simpler propagation helps recommendation.

### Success Criteria

Phase 6 is complete when:

- LightGCN trains reliably,
- results are compared to all earlier baselines,
- the project can clearly justify the move from generic graph recommender to recommendation-specific graph modeling.

### Common Risks

- introducing too many tuning variables at once,
- changing split logic during model comparison,
- making conclusions from a single lucky run.

## Phase 7: Real-World Extensions

### Objective

Move the project from a strong academic prototype toward a more realistic recommendation system.

### Why This Phase Matters

This is where the project becomes especially valuable for job applications. It shows that you understand not only how to train a model, but how recommendation systems behave in real products.

### Core Concept

A real recommender is affected by system constraints, data quality, and product tradeoffs.

### Why We Apply This Concept

A model with strong offline metrics can still fail in practice if it ignores:

- cold start,
- popularity bias,
- data freshness,
- scalability,
- business constraints.

### Areas To Extend

#### Temporal Splits

Train on past interactions and evaluate on future interactions.

Why:

- more realistic,
- lower leakage risk,
- stronger credibility.

#### Side Information

Add:

- user metadata,
- item metadata,
- genre or category features,
- text or content embeddings.

Why:

- helps cold start,
- makes the graph richer,
- increases realism.

#### Bias And Quality Analysis

Evaluate:

- popularity concentration,
- coverage,
- diversity,
- novelty.

Why:

- a good recommender is not only accurate,
- it should also provide useful and balanced exposure.

### Concrete Tasks

- switch from random split to temporal split,
- add at least one form of side information,
- analyze popularity bias and recommendation diversity,
- document system limitations and deployment concerns.

### Suggested Deliverables

- extended evaluation report,
- one realistic feature extension,
- a section on production considerations and tradeoffs.

### Success Criteria

Phase 7 is complete when the project can answer:

- how would this behave in a more realistic environment,
- what are the biggest practical limitations,
- what would be needed for deployment.

## Phase 8: Professional Packaging And Portfolio Presentation

### Objective

Turn the finished project into a polished, reviewable artifact for academic and career use.

### Why This Phase Matters

A technically good project can still be overlooked if it is poorly presented. Packaging is what converts technical work into a portfolio asset.

### Core Concept

Professional value comes from both engineering and communication.

### Why We Apply This Concept

Employers, supervisors, and reviewers need to understand:

- what problem you solved,
- why the approach was reasonable,
- what you built,
- what you learned,
- what the limitations are.

### Concrete Tasks

- write a polished project summary,
- create final result tables and figures,
- include baseline-to-advanced comparison,
- add architecture diagrams if possible,
- write a lessons-learned section,
- summarize future improvements,
- clean up repo structure and file names.

### Suggested Deliverables

- professional README updates,
- final report or case study,
- clean experiment summary,
- interview-ready project narrative.

### Success Criteria

Phase 8 is complete when the project can be shown to:

- an instructor,
- a hiring manager,
- a technical interviewer,
- a teammate who has never seen the repo before.

They should be able to understand the project direction, the methodology, and the value without needing you to explain every detail live.

## Recommended Phase Order Summary

1. Problem framing and dataset understanding.
2. Data preparation and bipartite graph construction.
3. Evaluation design and baseline metrics.
4. Non-graph baselines.
5. First graph recommender.
6. LightGCN and stronger graph recommendation.
7. Real-world extensions.
8. Professional packaging and portfolio presentation.

## Recommended Immediate Next Step

The best next actionable step is Phase 1.

Specifically, we should:

- choose the exact dataset,
- write the problem statement,
- inspect the schema,
- compute the first statistics table,
- define the initial ranking task.

Once that is done, the rest of the project becomes much easier to execute in a disciplined way.
