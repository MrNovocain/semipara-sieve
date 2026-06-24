# Statistica Sinica Submission Checklist

Official source checked: https://www3.stat.sinica.edu.tw/statistica/author_instru.html
Official template checked: https://www3.stat.sinica.edu.tw/statistica/latex.html

## Verified

- Main document uses the Statistica Sinica article template geometry: 12pt article, `natbib`, 31.9pc text width, 46.5pc text height, 1pc side margins, double spacing.
- Supplement uses the Statistica Sinica supplement template structure: 10pt book class, supplement header, S-numbered sections and equations.
- Main manuscript is 24 double-spaced pages including references, below the 40-page limit stated in the official instructions.
- The paper contains theoretical support for the main claims: stable-score decomposition, omitted-break failure, known-partition Wilks theorem, estimated-partition equivalence, unknown-q corollary, and local-power theorem.
- The paper contains numerical evidence: 300-replication Monte Carlo table and figure, plus the Ghana cocoa/weather empirical application.
- Numerical evidence is reproducible from `scripts/break_adaptive_results.py` and the data in `data/processed/cocoa_ghana.csv`.
- The latest LaTeX logs contain no unresolved references, citation warnings, TeX errors, overfull boxes, or underfull boxes. The only remaining warnings are inherited `sectsty` command-change warnings from the template stack.
- Rendered PDF pages were checked for blank pages and content clipping.

## Deferred Author Metadata

Per the current readiness pass, the author affiliation and e-mail block are treated as deferred and not blocking. The current manuscript front matter keeps `KU`; add the full affiliation and e-mail block before final production submission if the editorial office requires it.
