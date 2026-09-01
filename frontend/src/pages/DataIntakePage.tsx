import { CheckCircle2, Database, FileSpreadsheet, LoaderCircle, UploadCloud } from 'lucide-react';
import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import {
  commitAssessment,
  commitImport,
  getDatasetVersions,
  validateAssessment,
  validateImport
} from '../api';
import ErrorNotice from '../components/ErrorNotice';
import { copy, intakeFields } from '../copy';
import type {
  AppMeta,
  AssessmentPayload,
  DatasetVersion,
  ImportValidation,
  PresenceState
} from '../types';
import AnimatedContent from '../vendor/reactbits/AnimatedContent';

const initialAssessment: AssessmentPayload = {
  study_name: 'Odorant evaluation — Phase 1',
  session_name: 'Session 01',
  assessor_id: 'OP-0001',
  blinded_sample_code: 'SMP-0001',
  smiles: 'CCO',
  descriptor: '',
  presence_state: 'PRESENT',
  concentration: 10,
  concentration_unit: 'ppm',
  solvent: 'dipropylene glycol',
  temperature_c: 22,
  confidence: 80,
  replicate_number: 1,
  intensity: 6,
  source_name: 'private_panel',
  source_version: '1',
  source_license: 'PRIVATE',
  preparation_time_minutes: 30,
  notes: null
};

function Field({ label, children, wide = false }: { label: string; children: React.ReactNode; wide?: boolean }) {
  return <label className={wide ? 'intake-field is-wide' : 'intake-field'}><span>{label}</span>{children}</label>;
}

function ValidationSummary({ validation }: { validation: ImportValidation }) {
  const errorCount = validation.issues.filter((issue) => issue.severity === 'ERROR').length;
  const warningCount = validation.issues.length - errorCount;
  return <section className={validation.is_valid ? 'validation-summary is-valid' : 'validation-summary is-invalid'} aria-live="polite">
    <header>
      {validation.is_valid ? <CheckCircle2 aria-hidden="true" /> : <FileSpreadsheet aria-hidden="true" />}
      <strong>{validation.is_valid ? copy.validationPassed : copy.validationFailed}</strong>
    </header>
    <div className="validation-counters">
      <span><strong>{validation.row_count}</strong>{intakeFields.records}</span>
      <span><strong>{validation.valid_count}</strong>{intakeFields.valid}</span>
      <span><strong>{errorCount}</strong>{intakeFields.errors}</span>
      <span><strong>{warningCount}</strong>{intakeFields.warnings}</span>
    </div>
    {validation.issues.length > 0 && <details>
      <summary>{intakeFields.validationIssues}</summary>
      <ul>{validation.issues.map((issue, index) => <li key={`${issue.row}-${issue.code}-${index}`} className={`severity-${issue.severity.toLowerCase()}`}><b>{issue.severity}</b> · row {issue.row} · {issue.field}: {issue.message}</li>)}</ul>
    </details>}
  </section>;
}

function SnapshotList({ versions }: { versions: DatasetVersion[] }) {
  return <section className="snapshot-panel panel">
    <header><Database size={18} aria-hidden="true" /><h2>{copy.datasetVersions}</h2><span>{versions.length}</span></header>
    {!versions.length ? <p>{copy.noSnapshots}</p> : <ol>{versions.slice(0, 8).map((version) => <li key={version.dataset_version}>
      <div><strong>{version.dataset_version}</strong><small>{new Date(version.created_at).toLocaleString()}</small></div>
      <span>{version.row_count} rows</span>
      <code>{version.sha256.slice(0, 12)}</code>
    </li>)}</ol>}
  </section>;
}

export default function DataIntakePage({ meta }: { meta: AppMeta }) {
  const [workflow, setWorkflow] = useState<'manual' | 'batch'>('manual');
  const [assessment, setAssessment] = useState<AssessmentPayload>({ ...initialAssessment, descriptor: meta.label_names[0] ?? '' });
  const [manualValidation, setManualValidation] = useState<ImportValidation | null>(null);
  const [batchValidation, setBatchValidation] = useState<ImportValidation | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const refreshVersions = useCallback(() => { getDatasetVersions().then(setVersions).catch(setError); }, []);
  useEffect(refreshVersions, [refreshVersions]);

  const update = <K extends keyof AssessmentPayload>(field: K, value: AssessmentPayload[K]) => {
    setAssessment((current) => ({ ...current, [field]: value }));
    setManualValidation(null);
    setSuccess(null);
  };
  const states = useMemo(() => [
    ['PRESENT', copy.present], ['ABSENT', copy.absent], ['UNASSESSED', copy.unassessed]
  ] as Array<[PresenceState, string]>, []);

  const runManualValidation = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError(null); setSuccess(null);
    try { setManualValidation(await validateAssessment(assessment)); }
    catch (requestError) { setError(requestError); }
    finally { setBusy(false); }
  };
  const runManualCommit = async () => {
    setBusy(true); setError(null);
    try {
      await commitAssessment(assessment);
      setSuccess(copy.committed);
      setManualValidation(null);
      refreshVersions();
    } catch (requestError) { setError(requestError); }
    finally { setBusy(false); }
  };
  const runBatchValidation = async () => {
    if (!file) return;
    setBusy(true); setError(null); setSuccess(null);
    try { setBatchValidation(await validateImport(file)); }
    catch (requestError) { setError(requestError); }
    finally { setBusy(false); }
  };
  const runBatchCommit = async () => {
    const token = batchValidation?.validation_token;
    if (!token) return;
    setBusy(true); setError(null);
    try {
      await commitImport(token);
      setSuccess(copy.committed);
      setBatchValidation(null);
      setFile(null);
      refreshVersions();
    } catch (requestError) { setError(requestError); }
    finally { setBusy(false); }
  };

  return <AnimatedContent className="page data-intake-page">
    <header className="intake-heading"><div><h1>{copy.dataIntakeTitle}</h1><p>{copy.dataIntakeSubtitle}</p></div><a className="secondary-button" href="/api/v1/data/templates?format=csv" download>{copy.downloadTemplate}</a></header>
    <div className="workflow-tabs" role="tablist" aria-label={copy.dataIntake}>
      <button type="button" role="tab" aria-selected={workflow === 'manual'} className={workflow === 'manual' ? 'is-active' : ''} onClick={() => setWorkflow('manual')}>{copy.manualAssessment}</button>
      <button type="button" role="tab" aria-selected={workflow === 'batch'} className={workflow === 'batch' ? 'is-active' : ''} onClick={() => setWorkflow('batch')}>{copy.batchImport}</button>
    </div>
    {error ? <ErrorNotice error={error} /> : null}
    {success ? <div className="intake-success" role="status"><CheckCircle2 />{success}</div> : null}

    <div className="intake-layout">
      <div>
        {workflow === 'manual' ? <form className="intake-form panel" onSubmit={runManualValidation}>
          <fieldset><legend>{copy.experimentalContext}</legend><div className="intake-grid">
            <Field label={intakeFields.studyName}><input value={assessment.study_name} onChange={(event) => update('study_name', event.target.value)} required /></Field>
            <Field label={intakeFields.sessionName}><input value={assessment.session_name} onChange={(event) => update('session_name', event.target.value)} required /></Field>
            <Field label={intakeFields.assessorId}><input value={assessment.assessor_id} onChange={(event) => update('assessor_id', event.target.value)} required /></Field>
            <Field label={intakeFields.blindedSampleCode}><input value={assessment.blinded_sample_code} onChange={(event) => update('blinded_sample_code', event.target.value)} required /></Field>
            <Field label={intakeFields.concentration}><input type="number" min="0.000001" step="any" value={assessment.concentration} onChange={(event) => update('concentration', Number(event.target.value))} required /></Field>
            <Field label={intakeFields.concentrationUnit}><select value={assessment.concentration_unit} onChange={(event) => update('concentration_unit', event.target.value)}>{['ppm', 'ppb', '%v/v', '%w/v', 'M', 'mM', 'uM', 'nM'].map((unit) => <option key={unit}>{unit}</option>)}</select></Field>
            <Field label={intakeFields.solvent}><input value={assessment.solvent} onChange={(event) => update('solvent', event.target.value)} required /></Field>
            <Field label={intakeFields.temperature}><input type="number" min="-80" max="150" value={assessment.temperature_c} onChange={(event) => update('temperature_c', Number(event.target.value))} required /></Field>
            <Field label={intakeFields.preparationTime}><input type="number" min="0" value={assessment.preparation_time_minutes ?? ''} onChange={(event) => update('preparation_time_minutes', event.target.value ? Number(event.target.value) : null)} /></Field>
            <Field label={intakeFields.replicate}><input type="number" min="1" value={assessment.replicate_number} onChange={(event) => update('replicate_number', Number(event.target.value))} required /></Field>
          </div></fieldset>

          <fieldset><legend>{copy.molecularIdentity}</legend><div className="intake-grid"><Field label={intakeFields.smiles} wide><input className="mono-input" value={assessment.smiles} onChange={(event) => update('smiles', event.target.value)} required spellCheck={false} /></Field></div></fieldset>

          <fieldset><legend>{copy.sensoryObservation}</legend><div className="intake-grid">
            <Field label={intakeFields.descriptor}><select value={assessment.descriptor} onChange={(event) => update('descriptor', event.target.value)}>{meta.label_names.map((label) => <option key={label}>{label}</option>)}</select></Field>
            <div className="intake-field"><span>{intakeFields.presenceState}</span><div className="state-segments" role="group" aria-label={intakeFields.presenceState}>{states.map(([value, label]) => <button type="button" key={value} aria-label={label} className={assessment.presence_state === value ? 'is-active' : ''} aria-pressed={assessment.presence_state === value} onClick={() => { update('presence_state', value); update('intensity', value === 'PRESENT' ? 5 : null); }}>{label}</button>)}</div></div>
            <Field label={`${copy.intensity} · ${assessment.intensity ?? '—'}`}><input type="range" min="0" max="10" step="0.5" value={assessment.intensity ?? 0} disabled={assessment.presence_state !== 'PRESENT'} onChange={(event) => update('intensity', Number(event.target.value))} /></Field>
            <Field label={`${copy.confidence} · ${assessment.confidence}%`}><input type="range" min="0" max="100" step="5" value={assessment.confidence} onChange={(event) => update('confidence', Number(event.target.value))} /></Field>
          </div></fieldset>

          <fieldset><legend>{copy.sourceProvenance}</legend><div className="intake-grid">
            <Field label={intakeFields.sourceName}><input value={assessment.source_name} onChange={(event) => update('source_name', event.target.value)} required /></Field>
            <Field label={intakeFields.sourceVersion}><input value={assessment.source_version} onChange={(event) => update('source_version', event.target.value)} required /></Field>
            <Field label={intakeFields.sourceLicense}><input value={assessment.source_license} onChange={(event) => update('source_license', event.target.value)} required /></Field>
            <Field label={intakeFields.notes} wide><textarea value={assessment.notes ?? ''} onChange={(event) => update('notes', event.target.value || null)} rows={3} /></Field>
          </div></fieldset>
          <footer><p>{copy.traceabilityNote}</p><div><button type="submit" className="secondary-button" disabled={busy}>{busy ? <LoaderCircle className="spin" /> : null}{copy.validateRecord}</button><button type="button" className="primary-button" disabled={busy || !manualValidation?.is_valid} onClick={runManualCommit}>{copy.commitRecord}</button></div></footer>
        </form> : <section className="batch-import panel">
          <div className="upload-zone"><UploadCloud size={34} aria-hidden="true" /><h2>{copy.chooseDataFile}</h2><p>{intakeFields.fileRequirements}</p><input type="file" accept=".csv,.xlsx,.xlsm" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setBatchValidation(null); setSuccess(null); }} /></div>
          {file ? <div className="selected-file"><FileSpreadsheet /><div><strong>{file.name}</strong><small>{(file.size / 1024).toFixed(1)} KB</small></div><button type="button" className="secondary-button" disabled={busy} onClick={runBatchValidation}>{busy ? <LoaderCircle className="spin" /> : null}{copy.validateFile}</button></div> : null}
          {batchValidation?.preview.length ? <div className="import-preview"><h3>{intakeFields.preview}</h3><div><table><thead><tr>{Object.keys(batchValidation.preview[0]).slice(0, 6).map((key) => <th key={key}>{key}</th>)}</tr></thead><tbody>{batchValidation.preview.slice(0, 5).map((row, rowIndex) => <tr key={rowIndex}>{Object.keys(batchValidation.preview[0]).slice(0, 6).map((key) => <td key={key}>{String(row[key] ?? '')}</td>)}</tr>)}</tbody></table></div></div> : null}
          <footer><button type="button" className="primary-button" disabled={busy || !batchValidation?.is_valid || !batchValidation.validation_token} onClick={runBatchCommit}>{copy.commitImport}</button></footer>
        </section>}
        {workflow === 'manual' && manualValidation ? <ValidationSummary validation={manualValidation} /> : null}
        {workflow === 'batch' && batchValidation ? <ValidationSummary validation={batchValidation} /> : null}
      </div>
      <SnapshotList versions={versions} />
    </div>
  </AnimatedContent>;
}
