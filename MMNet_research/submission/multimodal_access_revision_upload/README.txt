MM-Net -- IEEE Access revision upload
=====================================

Drop these into the existing Overleaf project, overwriting. Nothing else in the
project needs to change: the class file, the fonts, the .bst and the four
unchanged figures are already there.

  multimodal_access.tex        the manuscript
  references.bib               52 entries; two added this round (Huttunen 2023,
                               and one further multitask paper), and reference 34
                               corrected to Physiol Rev 100(2):805-868, 2020

  figures/fig_mm_apnea_prev.pdf   Figure 1, restyled
  figures/fig_architecture.pdf    Figure 2, fusion block redrawn to match the
                                  model that was actually run
  figures/fig_roc_pr.pdf          Figure 5, panel (a) replaced with the
                                  event-level operating characteristic

  bio_suva.jpg                 re-cropped so it fills the frame like the others
  bio_abha.jpg                 new portrait

NOT included, because they have not changed since the last upload:
  fig_hypnogram.pdf, fig_confusion.pdf, fig_ahi.pdf, fig_learning_curve.pdf
  bio_sajin.jpg, bio_mobin.jpg
  ieeeaccess.cls, IEEEtran.bst, spotcolor.sty, logo.png, notaglinelogo.png,
  bullet.png, and the 40 Type 1 font files

Compile with pdfLaTeX. The class uses a pdfTeX-only Pantone spot colour, so
XeLaTeX and LuaLaTeX will fail on it.
