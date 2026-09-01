import { Check, ChevronDown, X } from 'lucide-react';
import { useMemo, useState } from 'react';
import { copy } from '../copy';

export default function TargetSelector({ labels, selected, onChange }: {
  labels: string[];
  selected: string[];
  onChange: (labels: string[]) => void;
}) {
  const [query, setQuery] = useState('');
  const filtered = useMemo(() => labels.filter((label) => label.toLowerCase().includes(query.toLowerCase())), [labels, query]);
  const toggle = (label: string) => onChange(selected.includes(label) ? selected.filter((item) => item !== label) : [...selected, label]);
  return (
    <div className="target-selector">
      <label>{copy.targetProfile}</label>
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
            {filtered.map((label) => <button type="button" key={label} className={selected.includes(label) ? 'is-selected' : ''} onClick={() => toggle(label)}><span>{label}</span>{selected.includes(label) && <Check size={15} />}</button>)}
          </div>
        </div>
      </details>
    </div>
  );
}
