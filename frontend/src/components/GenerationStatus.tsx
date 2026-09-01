import { AlertTriangle, CheckCircle2, Database, LoaderCircle, OctagonX, Search } from 'lucide-react';
import { copy, phaseCopy } from '../copy';
import type { AppMeta, GenerationEvent } from '../types';
import AnimatedList from '../vendor/reactbits/AnimatedList';
import CountUp from '../vendor/reactbits/CountUp';

function PhaseIcon({ phase }: { phase: GenerationEvent['phase'] }) {
  if (phase === 'ACCEPTED' || phase === 'REFERENCE_ACCEPTED') return <CheckCircle2 />;
  if (phase === 'REJECTED' || phase === 'INVALID') return <OctagonX />;
  if (phase === 'REVIEW' || phase === 'STEREO_REVIEW' || phase === 'PUBCHEM_UNVERIFIED' || phase === 'REFERENCE_UNVERIFIED') return <AlertTriangle />;
  if (phase === 'PUBCHEM_CHECK' || phase === 'PUBCHEM_FOUND' || phase === 'CHECKING_REFERENCES' || phase === 'CATALOG_MATCH') return <Search />;
  if (phase === 'DUPLICATE') return <Database />;
  return <LoaderCircle className="spin" />;
}

export default function GenerationStatus({ events, meta, running, onStop }: {
  events: GenerationEvent[];
  meta: AppMeta;
  running: boolean;
  onStop: () => void;
}) {
  const latest = events[events.length - 1];
  const counters = [
    [copy.attempts, latest?.attempt ?? 0],
    [copy.failedScreen, latest?.rejected ?? 0],
    [copy.referenceChecks, (latest?.reference_matches ?? 0) + (latest?.reference_unverified ?? 0) + (latest?.accepted ?? 0)],
    [copy.accepted, latest?.accepted ?? 0]
  ];
  const progress = Math.min(100, ((latest?.accepted ?? 0) / meta.generation_limits.required_candidates) * 100);
  return (
    <section className="generation-stage">
      <div className="stage-summary">
        <header><h2>{copy.status}</h2><span className={running ? 'running-state' : 'complete-state'}>{running ? copy.running : copy.complete}</span></header>
        <div className="counter-grid">{counters.map(([label, value]) => <div key={label}><strong><CountUp to={Number(value)} /></strong><span>{label}</span></div>)}</div>
        <div className="stage-progress" aria-label={copy.acceptedProgress}><span style={{ width: `${progress}%` }} /></div>
        <div className="stage-footer"><span>{running ? `${latest?.attempt ?? 0} / ${meta.generation_limits.max_attempts} ${copy.attempts.toLowerCase()}` : (latest?.accepted ?? 0) >= meta.generation_limits.required_candidates ? copy.statusSuccess : copy.statusLimit}</span>{running && <button type="button" className="secondary-button inverse" onClick={onStop}>{copy.stop}</button>}</div>
      </div>
      <div className="event-log-panel">
        <header><h2>{copy.screeningLog}</h2><span>{events.length} {copy.events}</span></header>
        <AnimatedList
          className="event-log"
          items={events.map((event, index) => ({
            id: `${event.attempt}-${event.phase}-${index}`,
            content: <div className={`event-line phase-${event.phase.toLowerCase()}`}><PhaseIcon phase={event.phase} /><span>{phaseCopy[event.phase]}</span><code>#{String(event.attempt).padStart(3, '0')}</code></div>
          }))}
        />
        {!events.length && <p className="log-empty">{copy.statusEventsEmpty}</p>}
      </div>
    </section>
  );
}
