import copy
import hashlib
import zipfile
import pytest
from scripts.ame_validation_batches import CONFIG, read, validate, verify_manifest


def test_plan_covers_required_experiments_and_is_valid():
    validate(read(CONFIG))


@pytest.mark.parametrize('mutation', ['missing_baseline', 'seed_overlap', 'insufficient_keys', 'bad_dependency', 'larger_network'])
def test_invalid_research_plan_fails_closed(mutation):
    cfg = copy.deepcopy(read(CONFIG))
    if mutation == 'missing_baseline': cfg['methods'].remove('random')
    if mutation == 'seed_overlap': cfg['common']['world_seeds']['test']['start'] = 61000
    if mutation == 'insufficient_keys': cfg['common']['core_keys'] = 20; cfg['common']['distractor_keys'] = 4
    if mutation == 'bad_dependency': cfg['batches'][1]['depends_on'] = ['B7']
    if mutation == 'larger_network': cfg['common']['adaptive_head_parameters'] = 763
    with pytest.raises(AssertionError): validate(cfg)


def test_freeze_detects_changed_evidence(tmp_path):
    evidence = tmp_path / 'result.json'; evidence.write_bytes(b'{"accuracy":0.5}')
    archive = tmp_path / 'snapshot.zip'
    with zipfile.ZipFile(archive, 'w') as z: z.write(evidence, 'result.json')
    manifest = dict(files={'result.json': hashlib.sha256(evidence.read_bytes()).hexdigest()},
                    archive='snapshot.zip', archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    verify_manifest(manifest, tmp_path)
    evidence.write_bytes(b'{"accuracy":1.0}')
    with pytest.raises(ValueError, match='changed'): verify_manifest(manifest, tmp_path)


def test_freeze_rejects_external_manifest_path(tmp_path):
    with pytest.raises(ValueError): verify_manifest({'files': {'../outside': 'x'}}, tmp_path)
