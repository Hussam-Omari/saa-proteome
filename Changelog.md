# Changelog

All notable changes to this project are documented in this file.

## v0.4.0 – 2026-07-22

### Added

- Added generalized sliding-window analysis for user-defined amino-acid groups.

- Added maximum localized cysteine percentage per sliding window.

- Added maximum localized total sulfur amino-acid percentage per sliding window.

- Exported `max_group_pct_per_window` through the public package interface.

- Added localized cysteine and total S-AA metrics to protein-level outputs and ranking tables.

### Changed

- Retained `max_residue_pct_per_window` as a backward-compatible wrapper.

- Updated the package version from 0.3.0 to 0.4.0.

- Extended Figure 3 analytical support to methionine, cysteine, and total S-AA.

---

## v0.3.0 – 2026-04-20

### Changed

- Refactored FASTA validation logic to improve structural consistency and parsing robustness

- Improved defensive checks and clarified docstrings

- Refined internal consistency across modules

### Fixed

- Enforced exclusion of zero-length adjusted sequences in composition functions

- Removed unused internal variables in summary functions

---

## v0.2.0 – 2026-03-15

### Added

- Initial stable implementation of proteome-level sulfur-containing amino acid (SAA) analysis

- Implemented amino acid composition, protein ranking, sliding-window metrics, and QC summary functions

---

## v0.1.0 – 2026-02-10

### Added

- Initial development version

- Implemented core amino acid metrics and SAA calculations
