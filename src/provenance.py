"""Small reproducibility bundle: source/config/dependency files, never .env or data."""
import hashlib
import importlib.metadata
import platform
import sys
import zipfile
from datetime import datetime, timezone
from src.config import ROOT


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def runtime_info():
    packages = {}
    for name in ("torch", "numpy", "scikit-learn", "httpx", "PyYAML", "python-dotenv", "filelock", "psutil", "onnxruntime", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages,
            "device": "cpu", "timestamp_utc": utc_now()}


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return None


def snapshot_source():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    paths += sorted((ROOT / "tests").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.ps1")) + sorted((ROOT / "scripts").glob("*.sh"))
    paths += sorted((ROOT / "configs").glob("*.yaml"))
    paths += [ROOT / name for name in ("requirements.txt", "requirements-dev.txt", "requirements-content.txt", "pyproject.toml")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    archive = ROOT / "artifacts/source" / (signature + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(str(archive) + ".lock"):
        if not archive.exists():
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
                for path in paths:
                    output.write(path, path.relative_to(ROOT).as_posix())
    return signature, archive.relative_to(ROOT).as_posix()
