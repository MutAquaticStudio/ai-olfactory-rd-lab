import numpy as np
import pytest
import torch
import json
import hashlib
import time
import pandas as pd
from rdkit import Chem

from olfactory.training.pu_exploratory import nnpu_risk, retrieval_metrics, prior_assumption
from olfactory.training.pu_exploratory import prepare_snapshot, grouped_bootstrap, write_json
from olfactory.data_foundation.evidence import file_hash
from olfactory.training.splits import chemical_group_calibrated_split
from olfactory.training.benchmark import assert_no_leakage
from train_pu_exploratory import train_one, verify_files


def test_nnpu_matches_independent_constant_logit_formula():
    # At zero logits, R = pi*log(2) + max(0, log(2)-pi*log(2)).
    logits = torch.zeros((3, 2), requires_grad=True)
    observed = torch.tensor([[True, False], [False, True], [False, False]])
    loss = nnpu_risk(logits, observed, torch.tensor([.2, .4]), torch.tensor([True, True]))
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_nnpu_nonnegative_correction_and_inactive_gradient():
    logits = torch.tensor([[5., 3.], [-5., 2.]], requires_grad=True)
    p = torch.tensor([[True, False], [False, False]])
    risk = nnpu_risk(logits, p, torch.tensor([.9, .2]), torch.tensor([True, False]))
    assert risk.item() == pytest.approx(.9 * np.log1p(np.exp(-5)), rel=1e-4)
    risk.backward()
    assert logits.grad[:, 1].abs().sum() == 0


def test_retrieval_does_not_score_unknown_as_absent():
    p = np.array([[True, False, True], [False, False, False]])
    scores = np.array([[.2, .9, .1], [.1, .2, .3]])
    result = retrieval_metrics(scores, p, np.array([True, True, False]))
    assert result['evaluated_rows'] == 1
    assert result['mean_reciprocal_rank'] == .5
    assert result['observed_positive_coverage'] == .5
    assert 'accuracy' not in result
    assert 'average_precision' not in result


def test_prior_is_explicit_train_only_sensitivity():
    assert prior_assumption(np.array([.2, .4]), .05).tolist() == pytest.approx([.24, .43])
    with pytest.raises(ValueError):
        prior_assumption(np.array([.2]), -.1)


def fixture_snapshot(path):
    path.mkdir()
    labels = [f'label{i}' for i in range(113)]
    structures = ['CCO', 'CCO', 'CC(O)CC', 'C[C@H](O)CC']
    rows, records = [], []
    for i, smiles in enumerate(structures):
        mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(mol)
        key = Chem.MolToInchiKey(mol)
        rows.append({'source_row': i, 'canonical_isomeric_smiles': canonical, 'inchikey': key,
                     'connectivity_key': key.split('-')[0], 'sources': '["fixture"]',
                     'mapped_terms': json.dumps({labels[i]: [labels[i]]})})
        for j, label in enumerate(labels):
            records.append({'source_row': i, 'descriptor': label, 'presence_state': 'PRESENT' if i == j else 'UNASSESSED'})
    pd.DataFrame(rows).to_parquet(path / 'molecules.parquet')
    pd.DataFrame(records).to_parquet(path / 'draft_assessments.parquet')
    (path / 'raw.csv').write_text('fixture raw source\n')
    label_hash = hashlib.sha256(json.dumps(labels, separators=(',', ':')).encode()).hexdigest()
    write_json(path / 'mapping.json', {'production_label_order_sha256': label_hash, 'mapping_version': 'fixture'})
    manifest = {'training_eligible': False, 'production_label_order_sha256': label_hash,
                'mapping_version': 'fixture', 'input_csv': str(path / 'raw.csv'),
                'input_csv_sha256': file_hash(path / 'raw.csv'),
                'files': {n: file_hash(path / n) for n in ('molecules.parquet', 'draft_assessments.parquet', 'mapping.json')}}
    write_json(path / 'manifest.json', manifest)
    return labels


def test_snapshot_preserves_unknown_excludes_stereo_and_deduplicates(tmp_path):
    source = tmp_path / 'source'
    labels = fixture_snapshot(source)
    original_hash = file_hash(source / 'draft_assessments.parquet')
    table, p, excluded, manifest = prepare_snapshot(source, labels)
    assert len(table) == 2
    assert p.dtype == bool  # Observation membership, not negative labels.
    assert p.sum() == 3
    assert excluded == [{'source_row': 2, 'reason': 'UNRESOLVED_STEREO'}]
    assert manifest['training_eligible'] is False
    assert file_hash(source / 'draft_assessments.parquet') == original_hash
    with pytest.raises(ValueError, match='label order'):
        prepare_snapshot(source, labels[::-1])
    (source / 'raw.csv').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        prepare_snapshot(source, labels)


def test_snapshot_assessed_absent_not_silently_pooled(tmp_path):
    source = tmp_path / 'source'
    labels = fixture_snapshot(source)
    frame = pd.read_parquet(source / 'draft_assessments.parquet')
    frame.loc[1, 'presence_state'] = 'ABSENT'
    frame.to_parquet(source / 'draft_assessments.parquet')
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest['files']['draft_assessments.parquet'] = file_hash(source / 'draft_assessments.parquet')
    write_json(source / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='assessed negatives'):
        prepare_snapshot(source, labels)


def test_chemical_groups_and_seed_stable():
    smiles = ['C[C@H](O)CC', 'C[C@@H](O)CC', 'c1ccccc1', 'Cc1ccccc1',
              'CCO', 'CCCCCCCC', 'C1CCCCC1', 'CC(=O)O', 'CN', 'CCS', 'C1CC1', 'c1ccncc1']
    labels = np.eye(len(smiles))
    a = chemical_group_calibrated_split(smiles, labels, seed=42).to_dict()
    b = chemical_group_calibrated_split(smiles, labels, seed=42).to_dict()
    assert a == b
    assert a['group_ids'][0] == a['group_ids'][1]
    assert a['group_ids'][2] == a['group_ids'][3]
    assert_no_leakage(a)


def tiny_training_data():
    generator = torch.Generator().manual_seed(12)
    x = torch.randn(10, 2048, generator=generator)
    p = torch.zeros(10, 113, dtype=torch.bool)
    p[::2, 0] = True
    p[1::2, 1] = True
    active = np.zeros(113, bool)
    active[:2] = True
    # Deliberately invalid cal/test indices: training must never access them.
    partitions = {'train': list(range(6)), 'validation': [6, 7], 'calibration': [999], 'locked_test': [999]}
    return x, p, partitions, active


def test_train_determinism_inactive_outputs_and_partition_isolation(tmp_path):
    torch.set_num_threads(1)
    x, p, partitions, active = tiny_training_data()
    kwargs = dict(architecture='linear', offset=0., seed=42, x=x, p=p, partitions=partitions,
                  active_np=active, priors_np=np.full(113, .3), device=torch.device('cpu'),
                  deadline=time.monotonic() + 60, max_epochs=2, patience=20)
    for name in ('a', 'b'):
        result = train_one(tmp_path / name, **kwargs)
        assert result['status'] == 'COMPLETE'
    a = torch.load(tmp_path / 'a/checkpoint.pth', weights_only=False)
    b = torch.load(tmp_path / 'b/checkpoint.pth', weights_only=False)
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    torch.manual_seed(42)
    original = torch.nn.Linear(2048, 113).state_dict()
    assert torch.equal(a['model']['weight'][2:], original['weight'][2:])


def test_deadline_then_resume_and_checkpoint_contract(tmp_path):
    x, p, partitions, active = tiny_training_data()
    kwargs = dict(architecture='linear', offset=0., seed=42, x=x, p=p, partitions=partitions,
                  active_np=active, priors_np=np.full(113, .3), device=torch.device('cpu'), max_epochs=2, patience=20)
    result = train_one(tmp_path / 'run', **kwargs, deadline=time.monotonic() - 1)
    assert result['status'] == 'BUDGET_EXHAUSTED' and result['epochs'] == 0
    result = train_one(tmp_path / 'run', **kwargs, deadline=time.monotonic() + 60)
    assert result['epochs'] == 2
    result = train_one(tmp_path / 'run', **kwargs, deadline=time.monotonic() + 60)
    assert result['epochs'] == 2
    kwargs['seed'] = 17
    with pytest.raises(ValueError, match='Resume configuration'):
        train_one(tmp_path / 'run', **kwargs, deadline=time.monotonic() + 60)


def test_bootstrap_identity_and_checksum_rejection(tmp_path):
    scores = np.array([[.8, .2], [.3, .7], [.5, .6]])
    observed = np.array([[True, False], [False, True], [True, False]])
    result = grouped_bootstrap(scores, scores, observed, np.array([True, True]), ['a', 'a', 'b'])
    assert result['mean_reciprocal_rank']['ci95'] == [0., 0.]
    (tmp_path / 'file').write_text('fixture')
    verify_files(tmp_path, {'file': file_hash(tmp_path / 'file')})
    with pytest.raises(ValueError, match='checksum'):
        verify_files(tmp_path, {'file': 'wrong'})
    with pytest.raises(ValueError, match='checksum'):
        verify_files(tmp_path, {'../elsewhere': 'wrong'})


def test_crash_after_atomic_checkpoint_resumes_identically(tmp_path, monkeypatch):
    import train_pu_exploratory as runner
    x, p, partitions, active = tiny_training_data()
    kwargs = dict(architecture='mlp', offset=0., seed=42, x=x, p=p, partitions=partitions,
                  active_np=active, priors_np=np.full(113, .3), device=torch.device('cpu'),
                  max_epochs=3, patience=20, deadline=time.monotonic() + 60)
    train_one(tmp_path / 'straight', **kwargs)
    save = runner.save_checkpoint

    def crash_after_commit(path, payload):
        save(path, payload)
        if path.name == 'checkpoint.pth' and payload['epoch'] == 1:
            raise RuntimeError('simulated process crash')

    monkeypatch.setattr(runner, 'save_checkpoint', crash_after_commit)
    with pytest.raises(RuntimeError, match='simulated'):
        train_one(tmp_path / 'resumed', **kwargs)
    monkeypatch.setattr(runner, 'save_checkpoint', save)
    train_one(tmp_path / 'resumed', **kwargs)
    a = torch.load(tmp_path / 'straight/checkpoint.pth', weights_only=False)
    b = torch.load(tmp_path / 'resumed/checkpoint.pth', weights_only=False)
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert a['best_score'] == b['best_score']
