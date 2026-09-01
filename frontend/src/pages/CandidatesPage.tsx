import { FlaskConical, LoaderCircle, Play, Square } from 'lucide-react';
import { Dispatch, SetStateAction, useEffect, useRef, useState } from 'react';
import { ApiError, appendBounded, streamCandidates } from '../api';
import { copy } from '../copy';
import type { AppMeta, GenerationComplete, GenerationEvent } from '../types';
import AnimatedContent from '../vendor/reactbits/AnimatedContent';
import CandidateCard from '../components/CandidateCard';
import ErrorNotice from '../components/ErrorNotice';
import GenerationStatus from '../components/GenerationStatus';
import ReviewQueue from '../components/ReviewQueue';
import TargetSelector from '../components/TargetSelector';

export interface CandidateWorkspaceState {
  initialized: boolean;
  targets: string[];
  diversity: number;
  consent: boolean;
  events: GenerationEvent[];
  result: GenerationComplete | null;
}

export const EMPTY_CANDIDATE_WORKSPACE: CandidateWorkspaceState = {
  initialized: false,
  targets: [],
  diversity: 0.8,
  consent: false,
  events: [],
  result: null
};

export function createCandidateWorkspace(labels: string[]): CandidateWorkspaceState {
  return {
    ...EMPTY_CANDIDATE_WORKSPACE,
    initialized: true,
    targets: ['jasmine', 'woody'].filter((label) => labels.includes(label))
  };
}

export default function CandidatesPage({ meta, workspace, setWorkspace }: {
  meta: AppMeta;
  workspace: CandidateWorkspaceState;
  setWorkspace: Dispatch<SetStateAction<CandidateWorkspaceState>>;
}) {
  const { targets, diversity, consent, events, result } = workspace;
  const [error, setError] = useState<unknown>(null);
  const [running, setRunning] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const requiredExternalProviders = meta.reference_verification.required_external_consents;
  const consentProviderNames = requiredExternalProviders.map((provider) => (
    meta.reference_verification.providers.find((item) => item.provider === provider)?.display_name ?? provider
  ));
  const consentIdentifiers = Array.from(new Set(
    meta.reference_verification.providers
      .filter((item) => requiredExternalProviders.includes(item.provider))
      .flatMap((item) => item.query_types)
  )).map((identifier) => ({
    ISOMERIC_SMILES: 'Isomeric SMILES',
    FULL_INCHIKEY: 'full InChIKey',
    CONNECTIVITY_INCHIKEY: 'connectivity InChIKey',
    CAS: 'CAS identifiers',
    NAME: 'ingredient names'
  }[identifier] ?? identifier)).join(' and ');
  const consentSatisfied = requiredExternalProviders.length === 0 || consent;

  useEffect(() => () => controller.current?.abort(), []);
  const stop = () => { controller.current?.abort(); setRunning(false); };
  const generate = async () => {
    if (!targets.length || !consentSatisfied || running) return;
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    setWorkspace((previous) => ({ ...previous, events: [] })); setError(null); setRunning(true);
    try {
      await streamCandidates(
        {
          target_descriptors: targets,
          sampling_diversity: diversity,
          reference_consents: consent ? requiredExternalProviders : []
        },
        {
          onProgress: (event) => setWorkspace((previous) => ({ ...previous, events: appendBounded(previous.events, event, meta.generation_limits.max_event_lines) })),
          onComplete: (complete) => { setWorkspace((previous) => ({ ...previous, result: complete })); setRunning(false); },
          onError: (streamError) => { setError(new ApiError(streamError)); setRunning(false); }
        },
        nextController.signal
      );
    } catch (requestError) {
      if (!(requestError instanceof DOMException && requestError.name === 'AbortError')) setError(requestError);
      setRunning(false);
    }
  };

  return (
    <AnimatedContent className="page page-candidates">
      <section className="candidate-controls">
        <TargetSelector labels={meta.label_names} selected={targets} onChange={(nextTargets) => setWorkspace((previous) => ({ ...previous, targets: nextTargets }))} />
        <div className="diversity-control"><label htmlFor="diversity">{copy.diversity}</label><div><output>{diversity.toFixed(1)}</output><input id="diversity" type="range" min="0.2" max="1.2" step="0.1" value={diversity} onChange={(event) => setWorkspace((previous) => ({ ...previous, diversity: Number(event.target.value) }))} /></div></div>
        {requiredExternalProviders.length > 0 && <label className="consent-control"><input type="checkbox" checked={consent} onChange={(event) => setWorkspace((previous) => ({ ...previous, consent: event.target.checked }))} /><span>{copy.consent(consentIdentifiers, consentProviderNames.join(', '))}</span></label>}
        <div className="generation-actions">
          <button type="button" className="primary-button" onClick={() => void generate()} disabled={!targets.length || !consentSatisfied || running}>{running ? <LoaderCircle className="spin" /> : <Play size={17} />}{copy.generate}</button>
          {running && <button type="button" className="secondary-button" onClick={stop}><Square size={15} />{copy.stop}</button>}
        </div>
      </section>

      {!events.length && !result && !error && <div className="candidate-empty"><FlaskConical size={30} /><p>{copy.emptyCandidates}</p></div>}
      {(events.length > 0 || running) && <GenerationStatus events={events} meta={meta} running={running} onStop={stop} />}
      {error !== null && <ErrorNotice error={error} />}
      {result && (
        <div className="candidate-results">
          {result.shortlist.length > 0 ? <><div className="section-heading"><div><h1>{copy.shortlist}</h1><p>{result.shortlist.length} {copy.rankedStructures}</p></div><span>{result.summary.attempts} {copy.attempts.toLowerCase()}</span></div><div className="candidate-list">{result.shortlist.map((candidate, index) => <CandidateCard key={candidate.isomeric_smiles} candidate={candidate} rank={index + 1} />)}</div></> : <div className="candidate-empty"><p>{result.summary.reference_unverified > 0 || result.summary.unverified > 0 ? copy.referenceUnavailable : (result.summary.reached_attempt_limit || result.summary.reached_time_limit) ? copy.statusLimit : copy.noCandidate}</p></div>}
          <ReviewQueue items={result.review_queue} />
        </div>
      )}
    </AnimatedContent>
  );
}
