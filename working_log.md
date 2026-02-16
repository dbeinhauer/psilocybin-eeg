# 2026-02-16
- starting to work on the data alignment using music (stimulus) channel
- second stricter preprocessing pipeline
    - be aware of the `bad epochs detection`
        - it may not work well (but it is cleaner for the paper)
    - majority of notes in shared Google document

# 2026-02-09
- 2 Participants did not participate in all experimental conditions:
    -  `PSI032` (2 conditions finished - only PLACEBO) - `PSI034` (3 conditions finished - missing PSYTRANCE under PSILOCYBIN)
    - probably excluded them from analysis?

# 2026-02-02
- First the data needs to be preprocessed:
    - trim 10s window from start and from end\
        - there is typically a movement and noise
    - exclude electrodes that are from the boundaries of the cap
        - typically cheeks etc. (the noisy ones)
        - approx 150 electrodes
        - ATiN has set of the excluded electrodes
    - log number of selected ICA components (and which components are excluded)
    - store Raw data series before ICA and after ICA
    - store IC component probabilities (muscle, eye, heart, brain)
        - check whether the components are assigned correctly
    - store ICs from ICA in time 
        - we can examine the muscle activity from there (if correctly filtered)
    - plot average power spectrums in pre-, post-, and ICA
        - should be alpha peek
        - should reasonably decline
        - we are mostly interested in the spectrum 0Hz to approx. 50 Hz
    - plot topomaps for Alpha, excluded, and through all included exponents
        - ideally alpha power - should be occipital (back)
        - muscles - typically circular
            - or very centralized
        - maybe only through components
            - excluded vs. filtered data
- check participants
    - if not all conditions - exclude
    - do stats of the dataset
- check seed in ICA in MNE
    - if it makes the same things in different runs

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