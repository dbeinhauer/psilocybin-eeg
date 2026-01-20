# 2026-01-20
## How to organize the whole codebase
What I would like to have:
    - `tests/` - unit tests
        - maybe further structure later
    - `src/` - main source of the analysis tools etc.
        - `data/` - data processing
        - `definitions/` - class, type, constants etc. definition
        - `features/` - main analysis features
        - `utils/` - helper features used across codebase
    - `notebooks/` - where jupyter notebooks with analyses should be placed
    - `scripts/` - mock scripts for playing with data etc.
    - `logs/` - place where run logs will be stored
    - `data/` - where data are stored (both raw and already preprocessed)
    - `results/` - where analysis results will be stored