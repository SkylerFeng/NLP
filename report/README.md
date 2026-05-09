# ACL Report

This folder contains the current project report in ACL-style LaTeX.

- `report.tex`: main report source.
- `custom.bib`: bibliography entries.

To compile with the official ACL template files available:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

The official ACL style files `acl.sty` and `acl_natbib.bst` are included locally so this folder can compile without a global ACL template installation. The author names in `report.tex` are placeholders and should be replaced with the real team member names.
