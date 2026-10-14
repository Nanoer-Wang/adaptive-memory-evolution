"""Export a comparison table, keeping run IDs and experimental conditions visible."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
records = []
for path in sorted((root / "runs").rglob("summary.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    if 'run' not in summary or 'metrics' not in summary:
        continue  # Research-round aggregate, not an individual evaluation.
    run, metrics = summary["run"], summary["metrics"]
    records.append({"run": path.parent.relative_to(root / 'runs').as_posix(), "method": run["method"], "mode": run["mode"],
                    "split": run["split"], "completed": metrics["completed"], "requested": len(run["question_ids"]),
                    "session_recall_at_k": metrics["session_recall_at_k"], "token_f1": metrics["token_f1"],
                    "custom_judge": metrics["custom_judge_correct"], "code": run["code_sha256"][:12],
                    "model": run["model"], "top_k": run["retrieval"]["top_k"]})
if records:
    with (root / "runs/comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(records, ensure_ascii=False, indent=2))
