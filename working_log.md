# 2026-01-30
- I use coordinates transformation defined in the MNE library
    - if I understand it correctly it converts the SFP file into the correct coordinate system that is used in MNE 
        - as majority of the analysis will be done in MNE it make sense IMHO
        - I like the MNE logging as after the conversion the plots of layout looks similar as the ones from the official tutorial from the cap documentation
            - see https://www.fieldtriptoolbox.org/template/layout/

# 2026-01-27
- Should I use fiducials?
    - they transform coordinates
    - but might be trans usable for different caps
    - for now I will use them as it seems to be standardized library conversion of the data from `FieldTrip` repository
    -> At the end I should not use fiducials for the moment -> maybe later

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