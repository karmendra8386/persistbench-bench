# Hand-labelling the pilot write sample

Each row of `handlabel_sample.csv` is one Session-1 memory write from
the pilot, with the model's raw summary, the stored write, and the
database truth for that chain.

Your job (human labeller): decide, from the stored write and the truth
alone, whether the write asserts any fact the truth contradicts.

- Fill `hand_label_false_memory` with 1 (a contradicted fact is asserted
  as stored fact) or 0 (no contradicted fact asserted).
- Optionally list the contradicted keys in `hand_falsified_fields`
  (comma-separated) and add a note.
- Do not look at the `auto_*` columns until your labels are complete.

The pilot gate is >= 90% agreement between your labels and the
`auto_label_false_memory` column. Disagreements are classifier bugs or
definition edge cases; record them in the notes column, do not relabel
to match.
