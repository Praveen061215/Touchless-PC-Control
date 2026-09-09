# Contributing Guidelines

Thank you for your interest in contributing to **Touchless PC Control**!

## How to Contribute

1. **Fork the Repository**: Create your personal fork on GitHub.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Set Up Development Environment**:
   ```bash
   py -m pip install -r requirements.txt
   py -m pip install -e ".[dev]"
   ```
4. **Run Diagnostics & Unit Tests**:
   Ensure all tests pass before opening a PR:
   ```bash
   py run_tests.py
   py VirtualMouse.py --diagnostics
   ```
5. **Code Style**:
   - Write clean, well-documented Python following PEP 8.
   - Decouple computer vision algorithms and business logic from hardware calls to maintain unit testability.
6. **Submit a Pull Request**: Provide a clear description of your changes and reference any related issues.
