"""Label-free, query-conditioned ranking features, fitted within each history."""
import hashlib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.config import resolve
from src.models.anchored_memory import FEATURE_VERSION


def unit_peak(values):
    values = np.asarray(values, dtype=np.float32)
    return values / max(float(values.max(initial=0)), 1e-8)


def build_features(history):
    texts, query = history.texts, history.question
    n = len(texts)
    output = np.zeros((n, 5), dtype=np.float32)
    if not n:
        return output
    for column, stop in [(0, None), (1, 'english')]:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=stop)
            matrix = vectorizer.fit_transform(texts)
            q = vectorizer.transform([query])
            output[:, column] = unit_peak((matrix @ q.T).toarray().ravel())
            if column == 1:
                for index, text in enumerate(texts):
                    windows = [text[start:start + 1200] for start in range(0, len(text), 900)] or ['']
                    output[index, 4] = (vectorizer.transform(windows) @ q.T).toarray().max(initial=0)
        except ValueError:
            pass
    try:
        counter = CountVectorizer(stop_words='english')
        counts = counter.fit_transform(texts).tocsr()
        qindices = counter.transform([query]).indices
        if len(qindices):
            df = np.asarray((counts > 0).sum(axis=0)).ravel()[qindices]
            idf = np.log1p((n - df + .5) / (df + .5))
            length = np.asarray(counts.sum(axis=1)).ravel()
            tf = counts[:, qindices].toarray()
            norm = 1.2 * (.25 + .75 * length / max(float(length.mean()), 1))
            output[:, 2] = unit_peak((idf * tf * 2.2 / (tf + norm[:, None])).sum(axis=1))
            output[:, 3] = ((tf > 0) * idf).sum(axis=1) / max(float(idf.sum()), 1e-8)
    except ValueError:
        pass
    output[:, 4] = unit_peak(output[:, 4])
    return output


def cached_features(config, history, with_status=False):
    # Content-addressed: never reuse features from a changed question or history.
    payload = json.dumps([FEATURE_VERSION, history.question, history.texts], ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    shared = config.get('retrieval', {}).get('feature_cache')
    root = resolve(shared) if shared else resolve(config['data']['prepared']) / 'ranking_features'
    root.mkdir(parents=True, exist_ok=True)
    path = root / (digest + '.npy')
    if path.exists():
        result = np.load(path, allow_pickle=False)
        return (result, True) if with_status else result
    result = build_features(history)
    # Atomic unique temporary file also supports concurrent readers/writers.
    import tempfile
    with tempfile.NamedTemporaryFile(dir=root, suffix='.tmp', delete=False) as stream:
        np.save(stream, result, allow_pickle=False)
        temporary = stream.name
    from pathlib import Path
    Path(temporary).replace(path)
    return (result, False) if with_status else result
