# Guidance for coding agents

## Comments and docstrings

Keep them terse. A docstring says what the function does in one or two lines; add a short note only when the behaviour is not obvious from the code. A comment explains a non-obvious why in a sentence; it does not restate the code, narrate the design history, or argue a case. No essays.

Put reasoning about tradeoffs, alternatives, and field evidence in the PR description or the commit message, not in the source.

## Style

Follow the existing code. Run `pre-commit run -a` and `pytest` before opening a PR. PR titles follow conventional commits; see CONTRIBUTING.md.
