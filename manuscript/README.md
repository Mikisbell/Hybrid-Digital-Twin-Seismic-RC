# Manuscript — HRPUB Civil Engineering and Architecture

This directory contains the working manuscript targeting
**Civil Engineering and Architecture** (HRPUB, Q2).

## Structure

```
manuscript/
├── 00_highlights.md          # (Not included in HRPUB submission)
├── 01_introduction.md        # Title, abstract, keywords, §1 Introduction
├── 02_objectives.md          # §2 Objectives
├── 03_methods.md             # §3: NLTHA model, PgNN architecture, FEM-guided loss
├── 04_results.md             # §4: Simulation outputs, training, predictions
├── 05_discussion.md          # §5: Interpretation, comparison, whiplash effect
├── 06_conclusions.md         # §6: Key findings, contributions, future work
├── 07_acknowledgements.md    # Acknowledgements
├── references.bib            # BibTeX bibliography — 31 references [1]–[31]
├── figures/                  # Publication-ready figures (300 DPI, PNG)
└── tables/                   # Formatted tables
```

## Section Numbering (HRPUB Format)

1. Introduction
2. Objectives
3. Materials and Methods
4. Results
5. Discussion
6. Conclusions
7. Acknowledgements

## Build

```bash
python scripts/build_docx.py
```
