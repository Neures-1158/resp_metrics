# Contributing to resp_metrics

Thank you for your interest in contributing to resp_metrics! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Python version and OS

### Suggesting Features

Feature requests are welcome! Please open an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install in development mode**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/resp_metrics.git
   cd resp_metrics
   pip install -e .[dev]
   ```
3. **Make your changes** with clear, descriptive commits
4. **Add tests** for any new functionality
5. **Run the test suite** to ensure nothing is broken:
   ```bash
   pytest tests/ --cov
   ```
6. **Submit a pull request** with a clear description of changes

### Code Style

- Use type hints for all function signatures
- Follow NumPy docstring format
- Run `ruff check src/` before submitting

### Testing

- Maintain test coverage above 90%
- Use pytest fixtures for reusable test data
- Mock external dependencies (e.g., file I/O)

## Questions?

Feel free to open an issue or reach out to the maintainers.

Thank you for contributing! 🙏
