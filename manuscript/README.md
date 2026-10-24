# Overleaf manuscript package

This package contains the audited manuscript and supplementary material dated
24 October 2026, the shared bibliography, the journal class, and every figure
used by either document. The scientific text and figure files are unchanged.

1. In Overleaf, choose **New Project → Upload Project** and upload the ZIP.
2. Select **sn-article.tex** as the main document and **pdfLaTeX** as compiler.
3. Compile. To export the supplement, select **Supplementary.tex** as the main
   document and compile again. Download both PDFs with their existing names.

Keep `latexmkrc` in the project root. It builds the companion document so that
references between the two documents resolve automatically. Its child build
does not load this configuration again, preventing recursive builds. The
approach follows [Overleaf's xr cross-reference guidance](https://www.overleaf.com/learn/how-to/Cross_referencing_with_the_xr_package_in_Overleaf).
It needs no shell-escape setting or external data download.

Local check with an existing TeX Live installation:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error sn-article.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error Supplementary.tex
```

The package includes editable LaTeX sources, not just rendered pages.
Cross-document PDF links require both exported PDFs in the same folder.
The LaTeX class and bibliography styles retain their original notices.
