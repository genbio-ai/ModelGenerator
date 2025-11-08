# Contributing to AIDO.ModelGenerator
Thank you for considering to contribute to AIDO.ModelGenerator!

## Merge Requests
We welcome your merge requests (MRs).
For minor fixes (e.g., documentation improvements), feel free to submit a MR directly.
If you would like to implement a new feature or a bug, please make sure you (or someone else) has opened an appropriate [issue](https://github.com/genbio-ai/ModelGenerator/issues) first; in your MR, please mention the issue it addresses.

### Creating a Merge Request
1. [Fork](https://github.com/genbio-ai/ModelGenerator/forks) this repository.
2. Install locally with `pip install -e .[dev]`.
3. Make your code changes locally.
4. **Set up commit hooks:**
   Initialize [pre-commit](https://pre-commit.com/) hooks:
   ```bash
   pre-commit install
   ```
   This will automatically check formatting (Ruff with max line length 100), trailing whitespace, end-of-file, YAML syntax, and large files before each commit.
5. Run `pytest tests/` to test your code.
6. If dependencies changed, rebuild the lock file with `poetry lock`
7. Check that your code is properly documented by going into the `docs` directory and running `mkdocs serve` to build the documentation and view it in your browser.
8. Issue a MR to merge your changes into the `main` branch.


## Issues
We use GitHub issues to track bugs and feature requests.
Before submitting an issue, please make sure:

1. You have read the README and documentation and your question is NOT addressed there.
2. You have done your best to ensure that your issue is NOT a duplicate of one of [the previous issues](https://github.com/genbio-ai/ModelGenerator/issues).
3. Your issue is either a bug (unexpected/undesirable behavior) or a feature request.

## License
By contributing to AIDO.ModelGenerator, you agree that your contributions will be licensed
under the LICENSE file in the root directory of the source tree.
