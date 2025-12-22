# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) and the changes for the upcoming release can be found in [changes](changes).

<!-- towncrier release notes start -->

## [mammos-analysis 0.4.0](https://github.com/MaMMoS-project/mammos-analysis/tree/0.4.0) – 2025-12-22

### Added

- `extract_BHmax`: calculate BHmax based on M, H and demagnization_coefficient. This replaces `extract_maximum_energy_product()` which has the same purpose but a different user interface. ([#53](https://github.com/MaMMoS-project/mammos-analysis/pull/53))

### Removed

- `extract_maximum_energy_product()`. Use `extract_BHmax` instead. ([#53](https://github.com/MaMMoS-project/mammos-analysis/pull/53))

## [mammos-analysis 0.3.0](https://github.com/MaMMoS-project/mammos-analysis/tree/0.3.0) – 2025-12-17

### Added

- Initial guesses for the Kuz'min fit are now allowed. ([#50](https://github.com/MaMMoS-project/mammos-analysis/pull/50))


## [mammos-analysis 0.2.0](https://github.com/MaMMoS-project/mammos-analysis/tree/0.2.0) – 2025-11-27

### Added

- Added `celsius=True` option in the `plot` methods for the `kuzmin` module to generate plots in degree Celsius. ([#40](https://github.com/MaMMoS-project/mammos-analysis/pull/40))


## [mammos-analysis 0.1.5](https://github.com/MaMMoS-project/mammos-analysis/tree/0.1.5) – 2025-11-03

### Fixed

- The function `kuzmin_properties` will not assume the magnetization input is in `A/m`. If the input is in a unit not convertible to `A/m` (e.g., Tesla), an error is raised. ([#31](https://github.com/MaMMoS-project/mammos-analysis/pull/31))


## [mammos-analysis 0.1.4](https://github.com/MaMMoS-project/mammos-analysis/tree/0.1.4) – 2025-08-12

### Misc

- Improve documentation of `mammos_analysis.kuzmin`: add equation of Kuz'min model in docstrings. ([#24](https://github.com/MaMMoS-project/mammos-analysis/pull/24))
- Use [towncrier](https://towncrier.readthedocs.io) to generate changelog from fragments. Each new PR must include a changelog fragment. ([#25](https://github.com/MaMMoS-project/mammos-analysis/pull/25))
