import { Check, ChevronDown, X } from 'lucide-react';
import { useMemo, useState } from 'react';
import { copy } from '../copy';
import type { TargetDescriptorMeta } from '../types';

export default function TargetSelector({ labels, selected, onChange, metadata = [], maxTargets = 3 }: {
  labels: string[];
  selected: string[];
  onChange: (labels: string[]) => void;
  metadata?: TargetDescriptorMeta[];
  maxTargets?: number;
}) {
  const [query, setQuery] = useState('');
  const metadataByLabel = useMemo(() => new Map(metadata.map((item) => [item.name, item])), [metadata]);
  const filtered = useMemo(() => labels.filter((label) => label.toLowerCase().includes(query.toLowerCase())), [labels, query]);
  const toggle = (label: string) => {
    const descriptor = metadataByLabel.get(label);
    if (descriptor && !descriptor.selectable) return;
    if (!selected.includes(label) && selected.length >= maxTargets) return;
    onChange(selected.includes(label) ? selected.filter((item) => item !== label) : [...selected, label]);
  };
  return (
    <div className="target-selector">
      <label>{copy.targetProfile}</label>
      <small className="target-limit">{copy.targetLimit(maxTargets)}</small>
      <details>
        <summary>
          <span className="selected-targets">
            {selected.length ? selected.map((label) => <span key={label}>{label}<button type="button" aria-label={`Remove ${label}`} onClick={(event) => { event.preventDefault(); toggle(label); }}><X size={13} /></button></span>) : <em>{copy.selectDescriptors}</em>}
          </span>
          <ChevronDown size={18} aria-hidden="true" />
        </summary>
        <div className="target-menu">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={copy.searchDescriptors} aria-label={copy.searchTargetDescriptors} />
          <div className="target-options">
            {filtered.map((label) => {
              const descriptor = metadataByLabel.get(label);
              const disabled = descriptor?.selectable === false || (!selected.includes(label) && selected.length >= maxTargets);
              return <button type="button" key={label} disabled={disabled} className={selected.includes(label) ? 'is-selected' : ''} onClick={() => toggle(label)}><span>{label}{descriptor?.maturity === 'LIMITED_EVIDENCE' && <small>{copy.limitedEvidence}</small>}{descriptor?.maturity === 'INSUFFICIENT' && <small>{copy.insufficientEvidence}</small>}</span>{selected.includes(label) && <Check size={15} />}</button>;
            })}
          </div>
        </div>
      </details>
    </div>
  );
}
