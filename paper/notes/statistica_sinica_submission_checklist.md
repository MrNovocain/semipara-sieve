# Statistica Sinica Submission Checklist

> Status note: this checklist is retained as a historical submission-readiness note. The current manuscript source of truth is `paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex`; noncanonical manuscript files are inventoried in `paper/notes/source_notes/noncanonical_manuscript_sources.md`.

Official source checked: https://www3.stat.sinica.edu.tw/statistica/author_instru.html
Official template checked: https://www3.stat.sinica.edu.tw/statistica/latex.html

## Verified

- Main document uses the Statistica Sinica article template geometry: 12pt article, `natbib`, 31.9pc text width, 46.5pc text height, 1pc side margins, double spacing.
- Supplement uses the Statistica Sinica supplement template structure: 10pt book class, supplement header, S-numbered sections and equations.
- Main manuscript is 24 double-spaced pages including references, below the 40-page limit stated in the official instructions.
- The paper contains theoretical support for the main claims: stable-score decomposition, omitted-break failure, known-partition Wilks theorem, estimated-partition equivalence, unknown-q corollary, and local-power theorem.
- The paper contains numerical evidence: 300-replication Monte Carlo table and figure, plus the Ghana cocoa/weather empirical application.
- Numerical evidence should be reproduced from the active config-driven runners (`python -m pseel.run`, `scripts/workbook_break_mc.py`, and `scripts/empirical_goyal_welch.py`); the old break-adaptive/weather script is archived under `archive/code_20260628/scripts/`.
- The latest LaTeX logs contain no unresolved references, citation warnings, TeX errors, overfull boxes, or underfull boxes. The only remaining warnings are inherited `sectsty` command-change warnings from the template stack.
- Rendered PDF pages were checked for blank pages and content clipping.

## Deferred Author Metadata

At the time of the historical readiness pass, the author affiliation and e-mail block were treated as deferred and not blocking. The historical draft front matter kept `KU`; add the full affiliation and e-mail block before final production submission if the editorial office requires it.
