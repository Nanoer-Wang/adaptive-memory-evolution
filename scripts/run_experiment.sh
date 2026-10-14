
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.cli download
python -m src.cli prepare
for method in lcm gru transformer; do
  if [ -f "artifacts/checkpoints/$method/last.pt" ]; then
    python -m src.cli train --method "$method" --resume
  else
    python -m src.cli train --method "$method"
  fi
done
for method in no_memory recent tfidf hash_rag lcm gru transformer; do
  python -m src.cli evaluate --method "$method" --retrieval-only
done
