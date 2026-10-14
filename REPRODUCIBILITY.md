# Reproducing Adaptive Memory Evolution

## Scope and input integrity

Use a fresh checkout for reproduction. Frozen policy code and every bundled
input are checked against SHA-256 hashes. Git preserves exact file bytes because
historical hashes include line endings. `repro/bootstrap_manifest.json` records
input hashes; `repro/upstream_sources.json` records public URLs and revisions.
All 123-parameter heads (seeds 11, 23, 42) are included. No retraining is needed.
The training implementation is retained in `scripts/evolving_memory_study.py`;
newly trained heads are a separate experiment, not interchangeable with these.

`repro/prepare.py --synthetic` reconstructs the exact 120/40/200 bAbI story split
from the public archive and reuses the frozen 32D projection cache. It refuses
mismatched existing files. This is the synthetic input split, not B8's language
question split. Data download and simulation do not need an API key.

## Installation and quick verification

```bash
python -m venv .venv
# Activate .venv (Windows: .venv\Scripts\Activate.ps1; Unix: source .venv/bin/activate).
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python repro/prepare.py --synthetic
python -m pytest -q tests/test_b1_interleaved.py tests/test_b1_attribution.py tests/test_validation_matrix.py tests/test_characterization.py tests/test_semantic_characterization.py tests/test_stress_characterization.py
python scripts/characterization.py --batch C1 --mini-only --workers 1
python scripts/validation_matrix.py --batch B2 --mini-only
```

Reference platform and package versions are archived in
`paper/evidence/C1/protocol.json` and the other batch protocols. CPU timings vary
by host; task metrics and deterministic traces are the primary replay checks.
The complete historical test suite includes earlier branches that require
additional datasets; the command above targets the released memory mechanism.

## Formal synthetic experiments

After the corresponding mini checks:

```bash
python scripts/b1_attribution.py --output runs/reproduction-b1 --stage all
python scripts/validation_matrix.py --batch B2
python scripts/validation_matrix.py --batch B3 --mini-only
python scripts/validation_matrix.py --batch B3
# Repeat mini, then formal, for B4, B5, B6, B7.
python scripts/characterization.py --batch C1 --workers 4
python scripts/characterization.py --batch C2 --mini-only --workers 1
python scripts/characterization.py --batch C2 --workers 4
python scripts/characterization_stress.py --mini-only
python scripts/characterization_stress.py
```

Use `characterization_stress.py` for the paper's C4 result. The older C4 path in
`characterization.py` is preserved for historical provenance and does not use the
final identical-prefix protocol. Formal sweeps may take hours and generate
multiple GB of traces; do not interpret a mini run as a formal result.

The bootstrap B1 protocol is a frozen input ledger used by B2-B7. Re-run B1 into
a NEW output directory, as shown above, rather than overwriting that ledger.
Completed outputs are verified before reuse. Do not alter scripts/configurations
while resuming a registered batch; use a fresh checkout for a new protocol.

## C3: semantic migration

```bash
python -m pip install -r requirements-content.txt
python repro/prepare.py --semantic
python scripts/characterization_semantic.py --mini-only
python scripts/characterization_semantic.py
```

The preparer downloads the pinned MiniLM encoder and MemOps evidence files and
checks their hashes. C3 is controlled recombination of public quotes, not an
official MemOps score or autonomous agent task.

## B8: language transfer

First complete B2-B7 in this checkout. Then:

```bash
python -m pip install -r requirements-content.txt
python repro/prepare.py --language
# Copy .env.example to .env and set your own AGNES_AI_API locally.
python scripts/validation_language.py --stage prepare
python scripts/validation_language.py --stage mini
python scripts/validation_language.py --stage answer-mini
python scripts/validation_language.py --stage judge-mini
python scripts/validation_language.py --stage formal
python scripts/validation_language.py --stage answer
python scripts/validation_language.py --stage judge
```

`--language` downloads LongMemEval (~277 MB), LoCoMo, NFCats and encoder assets.
It removes only an untouched bootstrap B8 protocol so the language runner can
register the new checkout; it refuses to replace existing B8 result data.
Do not run another bootstrap preparation between this step and B8 preparation.

Answer/judge stages call an external service; all earlier steps are local or
public downloads. The frozen protocol uses `agnes-3.0-flash` at
`https://api.agnes-ai.cn/v1`, with a shared limit of 9 requests per rolling minute,
persistent caching, retry/backoff, and saved answers. API availability and model
behavior can change, so fresh language answers are not promised bitwise identical.

B8 has 24 fixed questions per dataset, three policy seeds, seven methods. It is
not the official full LongMemEval/LoCoMo/MemoryBench benchmark. NFCats quality is
not binary factual QA accuracy. Raw datasets and private API caches are excluded.

## Evidence and paper

`paper/evidence/` contains original aggregate metrics, intervals, statistical
comparisons and selected traces. Every predeclared outcome remains, including
negative results. New run outputs go to `runs/` and do not replace that archive.

The LaTeX manuscript can be compiled without any experiment/data download:

```bash
cd paper
tectonic --keep-logs main.tex
# Alternatively: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

See `paper/README.txt` for figure generation and evidence auditing. Recreating
all assets with `build_assets.py` also requires original run outputs; the committed
PDF/vector figures already suffice for manuscript compilation.
