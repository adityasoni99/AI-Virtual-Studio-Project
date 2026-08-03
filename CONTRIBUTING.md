# Contributing to AI Virtual Studio

Thanks for your interest in contributing! We welcome improvements, bug fixes, documentation updates, and new features.

## Code of Conduct
Please read and follow the project's Code of Conduct in `CODE_OF_CONDUCT.md`.

## Getting started
1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-change
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## CI instructions
We use GitHub Actions for continuous integration. The expected CI checks include:
- Running the test suite with `pytest`
- Linting / formatting checks (optional)

Local CI-like checks you should run before opening a PR:
```bash
# run unit tests
pytest

# run a single test file
pytest tests/test_example.py

# (Optional) run linters/formatters if you use them
# e.g., with flake8 / black
black --check .
flake8
```

If you want to add or modify CI workflows, put them under `.github/workflows/` as YAML files. A suggested CI workflow name is `ci.yml` that runs on `push` and `pull_request` and executes `pip install -r requirements.txt` and `pytest`.

## Code style
- Follow standard Python conventions (PEP8).
- Prefer type hints where helpful.
- Keep functions and modules focused and well-documented.

## Tests
- Tests live in `tests/` and use pytest.
- Add tests for bug fixes and new features.
- Run the test suite locally before submitting:
  ```bash
  pytest
  ```

## API changes
- If you change or add API endpoints, update `docs/api.md` and the README examples.
- Include request/response examples and validation notes.

## Model / infra changes
- If your change affects model loading or resource usage (GPU/MPS/CPU), document expected memory/cost impacts and optional flags to disable heavy behavior for local development.

## Commit messages and PRs
- Write clear commit messages: short summary on the first line, optional body for details.
- Open a Pull Request and describe:
  - What problem you are solving
  - How to test the change (commands and example requests)
  - Any performance or compatibility notes
- Link related issues if available.

## Review process
- At least one approving review is required before merging.
- Maintain backwards compatibility for public API routes where possible; if incompatible changes are necessary, document migration steps.

## License and CLA
- By contributing you agree your contributions will be licensed under the repository license (see LICENSE).
- If you require a contributor license agreement (CLA) we can add one.

## Contact / Support
- Open an issue for feature requests or bugs.
- For quick questions, tag the repository maintainers on GitHub or open an issue and mention maintainers.
