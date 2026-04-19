# Git Commit Prefix Guide

This project can use a simple Conventional Commits style so commit history stays easy to scan.

## Format

Use this structure for most commits:

```text
<prefix>: <short description>
```

Example:

```text
feat: add GCN baseline training loop
```

## Prefixes And When To Use Them

### `feat`

Use `feat` when you add a new user-facing or developer-facing capability.

Examples:

```text
feat: add GraphSAGE model implementation
feat: support loading the MUTAG dataset from config
```

### `fix`

Use `fix` when you correct a bug, wrong behavior, or broken result.

Examples:

```text
fix: correct edge index shape in data loader
fix: prevent training crash when validation set is empty
```

### `docs`

Use `docs` when the change only affects documentation.

Examples:

```text
docs: add setup steps for local development
docs: clarify message passing notes in roadmap
```

### `style`

Use `style` for formatting-only changes that do not affect behavior.

Examples:

```text
style: reformat training module with black
style: fix markdown heading spacing
```

### `refactor`

Use `refactor` when you improve code structure without changing behavior.

Examples:

```text
refactor: split dataset utilities into separate module
refactor: simplify model factory logic
```

### `test`

Use `test` when adding or updating tests.

Examples:

```text
test: add unit tests for graph batching
test: cover invalid config handling in trainer
```

### `chore`

Use `chore` for maintenance work that does not fit feature, fix, or docs.

Examples:

```text
chore: add python cache files to gitignore
chore: update development dependencies
```

### `perf`

Use `perf` when you improve performance without changing the intended behavior.

Examples:

```text
perf: reduce graph batching overhead during training
perf: cache dataset statistics during preprocessing
```

### `build`

Use `build` when changing packaging, dependencies, or build-related tooling.

Examples:

```text
build: update torch-geometric dependency versions
build: add packaging metadata for local installs
```

### `ci`

Use `ci` when changing continuous integration or automation workflows.

Examples:

```text
ci: add pytest run to GitHub Actions workflow
ci: cache pip dependencies in CI pipeline
```

### `revert`

Use `revert` when undoing a previous commit.

Examples:

```text
revert: remove broken early stopping change
revert: undo dataset normalization update
```

### `init`

Use `init` for the first setup of a project, module, or major new area. This is less standard than the others, so prefer it only for genuine initialization work.

Examples:

```text
init: scaffold training package structure
init: create initial experiment tracking layout
```

### `ops`

Use `ops` for operational or deployment-related changes that are not mainly CI or build changes. This is a team-specific prefix, so use it only if your team wants to separate operations work from `ci` and `chore`.

Examples:

```text
ops: add production environment variables documentation
ops: update deployment config for model serving
```

### `improvement`

Use `improvement` only if your team explicitly wants a broad, non-standard bucket for general enhancements. In most cases, it is better to use a more precise prefix such as `feat`, `fix`, `refactor`, or `perf`.

Examples:

```text
improvement: simplify training logs for readability
improvement: polish experiment configuration defaults
```

## Quick Decision Guide

Use:

- `feat` for new functionality
- `fix` for bug fixes
- `docs` for documentation only
- `style` for formatting only
- `refactor` for code cleanup without behavior changes
- `test` for test-only changes
- `chore` for maintenance and repository housekeeping
- `perf` for performance improvements
- `build` for dependency or build-system changes
- `ci` for pipeline and automation changes
- `revert` for undoing an earlier commit
- `init` for first-time scaffolding or setup
- `ops` for deployment or operational changes if your team wants it
- `improvement` only as a last resort when no precise standard prefix fits

## Tips

- Keep the description short and specific.
- Use present tense, like `add`, `fix`, `update`, or `remove`.
- Prefer one logical change per commit.
- If a commit mixes multiple kinds of work, choose the prefix based on the main purpose of the commit.
- Prefer standard prefixes like `feat`, `fix`, and `docs` when they fit, and use less common ones like `init` only when they add clarity.
- Avoid inventing too many broad prefixes. The more precise the prefix, the easier commit history is to scan later.

## Recommended Usage For The Prefixes You Asked About

- `style`: yes, this is a good and commonly understood prefix for formatting-only changes.
- `init`: acceptable, but use it sparingly for real first-time scaffolding.
- `ops`: optional, useful if the team has deployment or environment work that deserves its own label.
- `improvement`: not recommended as a default prefix because it overlaps with `feat`, `refactor`, `fix`, and `perf`.

## Good Examples

```text
feat: add configurable hidden dimensions for GNN models
fix: handle missing node features in dataset preprocessing
docs: document experiment folder structure
refactor: move optimizer setup into training utilities
test: add coverage for early stopping
chore: rename config files for consistency
perf: speed up neighbor sampling
build: pin torch version for reproducible installs
ci: run lint and tests on pull requests
revert: undo incorrect default learning rate change
ops: update deployment settings for inference service
```
