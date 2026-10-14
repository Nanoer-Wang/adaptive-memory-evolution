# Third-party resources

The root MIT license applies to original project code. It does not relicense
upstream datasets, model weights, or the conference style file.

- **bAbI**: public synthetic facts from [Facebook's bAbI tasks](https://github.com/facebookarchive/bAbI-tasks), described by Weston et al. (2015). The preparer downloads the original public archive and recreates the exact study selection. Full raw stories are not redistributed in this repository. The frozen projection cache contains short synthetic fact strings and their numerical features.
- **MiniLM**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Downloaded from the recorded revision; its model card and upstream license accompany the download.
- **LongMemEval**: [official repository](https://github.com/xiaowu0162/LongMemEval) and its pinned public dataset revision.
- **LoCoMo**: [official repository](https://github.com/snap-research/locomo), pinned to the recorded commit.
- **MemoryBench**: [official repository](https://github.com/THUIR/MemoryBench) and pinned NFCats Arrow files.
- **MemOps**: [official repository](https://github.com/MemTensor/MemOps), with pinned evidence files for controlled semantic replay.
- **NeurIPS 2026 style**: the unmodified style in `paper/neurips_2026.sty` retains its original header and terms.

URLs, revisions, and SHA-256 checksums are in `repro/upstream_sources.json` and
the encoder source manifest. Downloads remain subject to their upstream terms.
The repository includes derived metrics and selected diagnostic evidence; these
do not constitute a redistribution of the full conversational datasets.
