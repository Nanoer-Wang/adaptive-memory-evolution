Figures 1 and 2: illustrated scientific mechanisms (revision 3)

concept = an immutable content card surrounded by a mutable I/C/U/T shell;
          three qualitative snapshots contrast with the fixed-decay reference.
framework = interleaved experience, rate controls and per-item state, score-gated
            writes into K resident slots, query addressing, and delayed feedback.

Shared visual vocabulary:
  teal = importance, blue = observed consistency, gold = usage, violet = age.
  The four-state shell appears around immutable content in both figures.
  Each resident slot contains a content card and all four state marks.
  The score gate controls retention; it does not rank answers.
  Only overflowing writes can reject a candidate or evict a resident.

Scientific interpretation:
  Arc lengths and slider positions are qualitative design glyphs, not measured,
  calibrated, or normalized state values. I/U/T are not bounded to [0,1].
  The left curve in Figure 1 is the analytic fixed-decay reference only.
  No experimental trajectory or performance curve has been invented.
  All content snapshots refer to a record that remains resident.
  Previously held inputs drive flow before the current event is consumed;
  new context is held forward. Feedback arrives later and only affects
  a retrieved item if it remains resident. The mechanism stays unchanged.

PDF: embedded vector text/geometry for LaTeX.
SVG: editable vector elements.
PNG: 360 dpi preview.

Redraw with: python paper/tools/draw_method_figures.py
Dependencies: matplotlib and numpy.
Full asset regeneration now calls the same renderer.
