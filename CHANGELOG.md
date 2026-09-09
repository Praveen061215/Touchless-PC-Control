# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-09

### Added
- **Central Configuration Management**: Added `config.py` and `config.json` for persistent user preferences and dynamic tuning without modifying code.
- **Adaptive 1€ Filter**: Integrated Casiez et al. 1€ adaptive low-pass filter in `filters.py` for eliminating cursor jitter during stationary poses while preserving zero-lag response during rapid motion.
- **System Utilities Module**: Extracted `AudioController` and `BrightnessController` into `system_control.py` with defensive cross-platform fallbacks and support for modern PyCaw `EndpointVolume`.
- **Decoupled Gesture Logic**: Modularized gesture classification and stateful majority-vote stabilization into `gestures.py`.
- **Low-Light Vision Enhancement**: Added `vision_utils.LowLightEnhancer` featuring luminance-channel CLAHE histogram equalization with on-the-fly 'L' hotkey toggle and HUD status badge.
- **Sensory Feedback**: Added visual click ripple animations and asynchronous auditory cues in `feedback.py` (toggleable via 'S' hotkey or config).
- **Command-Line Interface**: Added `cli.py` with argument parsing for `--camera`, `--filter`, `--width`, `--height`, `--no-hud`, and `--diagnostics`.
- **Automated Test Suite**: Added 28 unit tests across `tests/` covering configuration, filters, gesture classification, system controllers, vision enhancement, feedback, and CLI parsing.
- **CI/CD & Packaging**: Added GitHub Actions workflow (`.github/workflows/ci.yml`) and `pyproject.toml` configuration for standard packaging and continuous integration.

### Changed
- Refactored monolithic script in `VirtualMouse.py` into a modular, testable architecture.
- Improved gesture stability with decoupled `GestureStabilizer` majority-vote buffer.
- Upgraded default cursor smoothing from simple linear interpolation to adaptive 1€ filtering.

---

## [1.0.0] - 2026-08-26
- Added virtual mouse functionality using MediaPipe and PyAutoGUI.
- Supported 13 unique hand gestures.
- Initial release.
