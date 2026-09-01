import { AlertTriangle, CheckCircle2, CircleSlash2, ExternalLink, Search } from 'lucide-react';
import { copy } from '../copy';
import type { ReferenceEvidence, ReferenceGate } from '../types';

const providerNames: Record<string, string> = {
  PUBCHEM: 'PubChem',
  TGSC: 'TGSC licensed snapshot',
  SCENTREE: 'ScenTree licensed snapshot'
};

const matchLevelLabels = {
  EXACT_STEREO: copy.exactStereo,
  EXACT_CONNECTIVITY: copy.exactConnectivity,
  EXACT_CAS: copy.exactCas,
  NAME_ONLY: copy.nameOnly
} as const;

function evidenceLabel(item: ReferenceEvidence) {
  const provider = providerNames[item.provider] ?? item.provider;
  if (item.status === 'NO_MATCH') return copy.noMatchProvider(provider);
  if (item.status === 'MATCH') return item.provider === 'PUBCHEM' ? 'Structural identity match found' : copy.knownCatalog;
  if (item.status === 'AMBIGUOUS') return copy.ambiguousMatch;
  if (item.status === 'UNVERIFIED') return copy.unavailableCheck;
  return copy.notConfigured;
}

function EvidenceIcon({ status }: { status: ReferenceEvidence['status'] }) {
  if (status === 'NO_MATCH') return <CheckCircle2 aria-hidden="true" />;
  if (status === 'MATCH') return <Search aria-hidden="true" />;
  if (status === 'NOT_CONFIGURED') return <CircleSlash2 aria-hidden="true" />;
  return <AlertTriangle aria-hidden="true" />;
}

export default function ReferenceEvidencePanel({ checks, gate, compact = false }: {
  checks: ReferenceEvidence[];
  gate: ReferenceGate;
  compact?: boolean;
}) {
  return (
    <section className={`reference-evidence ${compact ? 'is-compact' : ''}`}>
      <header>
        <h3>{copy.referenceEvidence}</h3>
        <span className={`reference-gate gate-${gate.status.toLowerCase()}`}>{gate.status.replaceAll('_', ' ')}</span>
      </header>
      {gate.status === 'PASS' && <p className="reference-conclusion">{copy.noConfiguredMatch}</p>}
      <div className="reference-evidence-list">
        {checks.map((item) => (
          <div className={`reference-evidence-row status-${item.status.toLowerCase()}`} key={item.provider}>
            <EvidenceIcon status={item.status} />
            <div>
              <strong>{providerNames[item.provider] ?? item.provider}</strong>
              <span>{evidenceLabel(item)}</span>
              {item.match_level && <small>{matchLevelLabels[item.match_level]}</small>}
              {item.source_version && <small>{item.source_version}</small>}
              {item.error_code && item.status !== 'NOT_CONFIGURED' && <small>{item.error_code}</small>}
            </div>
            {item.record_urls[0] && (
              <a href={item.record_urls[0]} target="_blank" rel="noreferrer" aria-label={`Open ${item.provider} record`}>
                <ExternalLink aria-hidden="true" />
              </a>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
