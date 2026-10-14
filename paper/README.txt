Adaptive Memory Evolution: Controlling Stability-Plasticity in Agent Memory
=========================================================================

This is a complete English research manuscript, not an experimental report
outline. The target layout is eight main-text pages, followed by references
and a reproducibility appendix. The PDF and LaTeX use the official, unmodified
NeurIPS 2026 style in preprint mode. No venue submission or acceptance is implied.

Read main.pdf. Edit main.tex and sections/*.tex.

Public implementation and reproduction guide:
https://github.com/Nanoer-Wang/adaptive-memory-evolution
References use numbered square-bracket citations, sorted/compressed within
citations, with the bibliography ordered by first citation (natbib/unsrtnat).
The appendix includes a primary-source protocol comparison with related work;
no incompatible external performance numbers are pooled with project results.

Compile the source
------------------
All required vector figures, generated tables, bibliography, and style are
included. No dataset, model download, API key, GPU, or experimental rerun is
required to compile the manuscript.

Option A: Upload the source to Overleaf, select main.tex as the main document,
and use pdfLaTeX (BibTeX is handled automatically).

Option B: With a local TeX distribution:
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Option C: With Tectonic:
    tectonic --keep-logs main.tex

On Windows, build.ps1 finds Tectonic or latexmk, builds in build/, and copies
the result to main.pdf. The portable Tectonic executable used for local QA is
not included in the ZIP. Its version and official download URL are recorded
in tools/tectonic_release.json.

Evidence and reproducibility
----------------------------
tables/ and figures/ are derived from frozen project outputs.
evidence/ contains unrounded summaries, confidence intervals, paired tests,
protocols, selected traces, and diagnostic records. source_manifest.json
maps source paths in the original project to SHA-256 checksums.
evidence/claim_audit.json verifies central numerical claims and records limits.
evidence/reference_metadata.json records primary-source reference metadata.
evidence/raw evidence indexes point to full traces in the original project.

The multi-gigabyte event archives, language API responses, experimental source,
and model checkpoints remain in the original project; this ZIP is a manuscript
and analysis-evidence package, not a standalone full experiment distribution.

To regenerate manuscript assets inside the original project, from its root:
    python paper/tools/build_assets.py
This needs numpy and matplotlib and the already completed runs/ directories.
It reads results; it does not train or optimize any policy. The included
figures and tables make this step unnecessary for ordinary compilation.

For rendered PDF review:
    python paper/tools/inspect_pdf.py
This needs pypdf, pypdfium2, and Pillow. It renders every page and inventories
unresolved references and the main-text page count. Review renders and build
intermediates are intentionally excluded from the ZIP.

Scope of this paper
-------------------
Exactly three contributions: state formulation, explicit hybrid evolution,
and controlled stability-plasticity characterization. Strength interventions
change I/C/U flow rates with controller weights frozen. Alpha zero is not a
fully frozen-state control. Replacement matching is partial. Semantic and
conversational failures are reported. No LNN, universal accuracy, or runtime
superiority is claimed.

Before submission, supply the real author information or select the target
venue's anonymous style, check its then-current page and checklist rules,
and complete any required code/data release and author declarations.

Evidence files preserve original machine paths. Before a double-blind submission,
anonymize a separate submission copy while retaining this provenance archive.
