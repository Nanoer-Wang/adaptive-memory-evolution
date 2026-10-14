from __future__ import annotations
import argparse
import json
import logging
from src.config import load_config, resolve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adaptive Memory Evolution; legacy experiments remain reproducible")
    commands = parser.add_subparsers(dest="command", required=True)
    evolving = commands.add_parser('evolving-memory', help='Controller-agnostic evolving memory benchmarks')
    evolving.add_argument('--phase', choices=('prepare','train','evaluate','report','all'), default='report')
    consolidation = commands.add_parser('consolidation', help='New external-memory consolidation framework')
    consolidation.add_argument('--phase', choices=('prepare','train','evaluate','continual','controls','report','all'), default='report')
    utility = commands.add_parser('continual-utility', help='Long-horizon future-utility training and evaluation')
    utility.add_argument('--phase', choices=('prepare','train','evaluate','diagnostics','report','all'), default='report')
    evolution = commands.add_parser('memory-evolution', help='Adaptive per-item memory state benchmarks')
    evolution.add_argument('--phase', choices=('prepare','train','evaluate','diagnostics','boundary','report','all'), default='report')
    for name in ("download", "prepare", "inspect", "doctor", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/default.yaml")
        if name == "doctor":
            command.add_argument("--online", action="store_true", help="Make one tiny Agnes request")
        if name == "train":
            command.add_argument("--method", choices=("lcm", "gru", "transformer", "lcm_v2", "ranker", "gru_v2"), default="lcm")
            command.add_argument("--epochs", type=int)
            command.add_argument("--output")
            command.add_argument("--resume", action="store_true")
            command.add_argument("--no-consolidation", action="store_true")
            command.add_argument("--fixed-time", action="store_true")
            command.add_argument("--reset-history", action="store_true", help="Reset recurrent state before each session, retaining parameters")
        if name == "evaluate":
            from src.evaluation.runner import METHODS
            command.add_argument("--method", choices=METHODS, default="tfidf")
            command.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
            command.add_argument("--limit", type=int)
            command.add_argument("--checkpoint")
            command.add_argument("--retrieval-only", action="store_true")
            command.add_argument("--judge", action="store_true")
            command.add_argument("--output")
            command.add_argument("--allow-train-overlap", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        if args.command == 'evolving-memory':
            from scripts import evolving_memory_study as study
            study.torch.set_num_threads(1)
            for name,fn in [('prepare',study.prepare),('train',study.train),('evaluate',study.evaluate),('report',study.report)]:
                if args.phase in (name,'all'):fn()
            return 0
        if args.command == 'memory-evolution':
            from scripts import adaptive_evolution_study as study
            study.torch.set_num_threads(1)
            for name,fn in [('prepare',study.prepare),('train',study.train),('evaluate',study.evaluate),
                            ('diagnostics',study.diagnostics),('boundary',study.boundary_control),('report',study.report)]:
                if args.phase in (name,'all'):fn()
            return 0
        if args.command == 'continual-utility':
            from scripts import continual_utility_study as study
            study.torch.set_num_threads(1)
            for name,fn in [('prepare',study.prepare),('train',study.train),('evaluate',study.evaluate)]:
                if args.phase in (name,'all'):fn()
            if args.phase in ('diagnostics','all'):
                from scripts.continual_utility_diagnostics import main as diagnostics
                diagnostics()
            if args.phase in ('report','all'):study.report()
            return 0
        if args.command == 'consolidation':
            from scripts import consolidation_study, consolidation_continual, report_consolidation
            if args.phase in ('prepare','all'):consolidation_study.prepare()
            if args.phase in ('train','all'):consolidation_study.train()
            if args.phase in ('evaluate','all'):consolidation_study.evaluate()
            if args.phase in ('continual','all'):consolidation_continual.main()
            if args.phase=='controls':consolidation_continual.main(True)
            if args.phase=='report':report_consolidation.main()
            return 0
        config = load_config(args.config)
        if args.command == "download":
            from src.data.download import download
            result = {"file": str(download(resolve(config["data"]["file"]).parent))}
        elif args.command == "prepare":
            from src.data.longmemeval import prepare
            manifest = prepare(config)
            result = {"protocol": manifest["protocol"], "sizes": {key: len(value) for key, value in manifest["splits"].items()}}
        elif args.command == "inspect":
            from src.data.longmemeval import load_rows
            from collections import Counter
            rows = load_rows(config["data"]["file"])
            result = {"instances": len(rows), "types": dict(Counter(r["question_type"] for r in rows)),
                      "abstentions": sum(str(r["question_id"]).endswith("_abs") for r in rows),
                      "file_mb": resolve(config["data"]["file"]).stat().st_size / 1e6}
        elif args.command == "doctor":
            import importlib.util
            from dotenv import dotenv_values
            from src.config import ROOT
            values = dotenv_values(ROOT / ".env", encoding="utf-8-sig")
            result = {"key_present_in_env_file": bool(values.get("AGNES_AI_API") or values.get("AGNES_API_KEY")),
                      "torch_available": importlib.util.find_spec("torch") is not None,
                      "dataset_exists": resolve(config["data"]["file"]).exists(), "rpm": config["api"]["rpm"]}
            if args.online:
                from src.api.agnes import AgnesClient
                client = AgnesClient(config["api"])
                try:
                    response = client.complete([{"role": "user", "content": "Reply with exactly OK."}], namespace="connectivity-v1", max_tokens=16)
                    result.update({"online": "ok", "model": response["model"], "usage": response["usage"], "cache_hit": response["cache_hit"]})
                finally:
                    client.close()
        elif args.command == "train":
            if args.epochs is not None and args.epochs < 1:
                raise ValueError("epochs must be positive")
            from src.training.train import train
            result = train(config, args.method, args.epochs, args.output, args.resume, not args.no_consolidation, args.fixed_time, args.reset_history)
        else:
            from src.evaluation.runner import evaluate
            result = evaluate(config, args.method, args.split, args.limit, args.checkpoint,
                              args.retrieval_only, args.judge, args.output, args.allow_train_overlap)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("Paused. Completed evaluation records and epoch checkpoints are saved. Repeat the command to resume evaluation; use --resume for training.")
        return 130
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
