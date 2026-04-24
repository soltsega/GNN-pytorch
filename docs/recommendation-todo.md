# Recommendation System TODO List

This is the direct execution checklist for the graph-based recommendation project.

## Phase 1: Problem And Dataset

- [ ] Choose the dataset for the first version.
- [ ] Confirm the project will use implicit-feedback recommendation.
- [ ] Write a 1-paragraph problem statement.
- [ ] Identify the user column.
- [ ] Identify the item column.
- [ ] Identify the timestamp column if available.
- [ ] Identify whether rating data exists.
- [ ] Compute the number of users.
- [ ] Compute the number of items.
- [ ] Compute the number of interactions.
- [ ] Compute interaction sparsity.
- [ ] Write a short note explaining why this is a graph problem.

## Phase 2: Preprocessing And Graph Construction

- [ ] Create user ID to index mapping.
- [ ] Create item ID to index mapping.
- [ ] Build an indexed interaction table.
- [ ] Save mapping tables for reproducibility.
- [ ] Split interactions into train, validation, and test sets.
- [ ] Make sure test data is not used in graph construction.
- [ ] Build the bipartite graph from training interactions only.
- [ ] Create adjacency or edge-index representation.
- [ ] Save graph statistics.
- [ ] Write a short explanation of the bipartite graph structure.

## Phase 3: Evaluation Setup

- [ ] Decide the first split strategy.
- [ ] Implement top-K recommendation evaluation.
- [ ] Implement `Recall@K`.
- [ ] Implement `Precision@K`.
- [ ] Implement `HitRate@K`.
- [ ] Implement `NDCG@K`.
- [ ] Mask training-seen items during evaluation.
- [ ] Test the metrics on a small toy example.
- [ ] Document the evaluation protocol.

## Phase 4: Baselines

- [ ] Implement popularity baseline.
- [ ] Evaluate popularity baseline.
- [ ] Implement matrix factorization baseline.
- [ ] Train matrix factorization baseline.
- [ ] Evaluate matrix factorization baseline.
- [ ] Compare both baselines in one result table.
- [ ] Write a short note on strengths and weaknesses of each baseline.

## Phase 5: First Graph Recommender

- [ ] Create learnable user embeddings.
- [ ] Create learnable item embeddings.
- [ ] Implement graph propagation over the user-item graph.
- [ ] Score user-item pairs with dot product.
- [ ] Add a simple training loop.
- [ ] Train the first graph recommender.
- [ ] Evaluate it with the same ranking metrics.
- [ ] Compare it against the baselines.
- [ ] Write a note explaining what one-layer and two-layer propagation mean.

## Phase 6: LightGCN

- [ ] Implement LightGCN propagation.
- [ ] Add negative sampling.
- [ ] Add BPR loss.
- [ ] Train LightGCN.
- [ ] Evaluate LightGCN.
- [ ] Compare LightGCN with popularity, matrix factorization, and the first graph model.
- [ ] Run depth comparison experiments.
- [ ] Run embedding-dimension comparison experiments.
- [ ] Write a short explanation of why LightGCN fits recommendation well.

## Phase 7: Real-World Extensions

- [ ] Add temporal split evaluation.
- [ ] Add at least one side feature.
- [ ] Analyze cold-start limitations.
- [ ] Analyze popularity bias.
- [ ] Analyze coverage or diversity.
- [ ] Write a short production-considerations note.

## Phase 8: Final Packaging

- [ ] Clean the repository structure.
- [ ] Finalize the README project summary.
- [ ] Add final metrics tables.
- [ ] Add final figures or plots.
- [ ] Write a short final report.
- [ ] Add a lessons-learned section.
- [ ] Add a future-work section.
- [ ] Prepare a portfolio-ready project description.

## Immediate Next TODO

- [ ] Select the dataset.
- [ ] Write the recommendation problem statement.
- [ ] Inspect the raw dataset schema.
- [ ] Compute the first dataset statistics table.
