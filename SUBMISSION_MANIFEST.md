# Applied Acoustics — Submission Package Manifest

This folder is self-contained: `main.tex` compiles standalone against the
files here (verified). It mirrors what Elsevier's Editorial Manager expects
for a LaTeX submission, per the journal's Guide for Authors.

## Files in this package → where they go in Editorial Manager

| File | Upload as / where it's used |
|---|---|
| `main.tex` | Manuscript source file (item type: "Manuscript" / LaTeX source) |
| `references.bib` | Manuscript source file (same upload step as `main.tex`) |
| `figures/*.png` (11 files) | Upload individually as "Figure" items, in this exact citation order: (1) `spectrogram_exemplars.png`, (2) `fig_architecture.png`, (3) `confusion_matrices.png`, (4) `roc_curves.png`, (5) `training_convergence.png`, (6) `representation_ablation_plot.png`, (7) `architecture_ablation_plot.png`, (8) `snr_sweep_plot.png`, (9) `fig_pipeline.png`, (10) `thread_sweep_plot.png`, (11) `latency_distribution_plot.png` |
| `graphical_abstract_standalone.png` | Upload as item type "Graphical Abstract" (separate upload slot, per the Guide — this is in addition to it being embedded inline in the compiled PDF) |
| `highlights.txt` | Upload as item type "Highlights" — filename already contains "highlights" as required |
| `cover_letter.md` | Paste into the "Cover Letter" text box in Editorial Manager (plain text, no file upload needed) |
| `main_reference_copy.pdf` | Not uploaded — this is your own reference copy of what Editorial Manager's auto-generated review PDF should look like. Elsevier builds the reviewer-facing PDF itself from the source files above. |

## Things that are NOT files — fill these into Editorial Manager's web forms

These are already written into `main.tex` (so the content is finalized and
consistent with the manuscript), but Elsevier also wants them entered
separately into specific steps of the online submission form:

- **Declaration of competing interest** — Editorial Manager has its own
  "declarations tool" that generates a signed Word document when you answer
  its questions online (select "I have nothing to declare" if that's still
  accurate). You cannot pre-supply this file; it's generated during
  submission. The manuscript's in-text "Declaration of competing interest"
  section is a convenience copy of the same statement, not a substitute for
  the tool.
- **Funding statement** — copy the text from `main.tex`'s "Funding" section
  into the submission system's funding step.
- **Data availability statement** — copy the text from `main.tex`'s "Data
  availability" section (including the anonymized repository link) into the
  submission system's data statement step.
- **CRediT author contributions** — copy from `main.tex`'s "CRediT authorship
  contribution statement" section if Editorial Manager has a separate CRediT
  entry form for your submission track.

## Not included (intentionally)

- `Acoustic Anomaly manuscript.docx` — a courtesy full-text copy for
  co-author review, not part of the LaTeX submission (main.tex is the one
  editable source being submitted).
- `figures/split_sensitivity_plot.png`, `figures/Grpahical Abstract.png` —
  present in the working `paper/figures/` folder but not cited by any
  `\includegraphics` in `main.tex`; excluded here to keep this package
  matching exactly what the manuscript actually uses.

## Pre-flight checklist (from the Guide for Authors)

- [x] Corresponding author designated with full contact details (Tao Zhang,
      `taozhang2021@hau.edu.cn`)
- [x] All in-text citations resolve to the reference list and vice versa (37
      entries, verified via clean compile with no undefined citations)
- [x] Abstract ≤250 words (currently 244)
- [x] Highlights: 3–5 bullets, each ≤85 characters (verified)
- [x] Graphical abstract provided, dimensions exceed Elsevier's minimum
      (1916×821 px vs. required 1328×531 px minimum)
- [x] Generative AI use declared in-text (names Claude/Anthropic)
- [ ] CRediT statement, funding, and data availability — verify current
      text in `main.tex` still matches what you enter into the online forms
      at submission time (in case of last-minute manuscript edits)
- [ ] Complete Elsevier's online declarations tool for competing interests
      during submission
