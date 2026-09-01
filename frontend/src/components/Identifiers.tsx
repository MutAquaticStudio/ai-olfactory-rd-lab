import { Check, Copy } from 'lucide-react';
import { useState } from 'react';
import { copy } from '../copy';

function Identifier({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div className="identifier-field">
      <label>{label}</label>
      <div><code>{value}</code><button type="button" onClick={handleCopy} aria-label={`Copy ${label}`}>{copied ? <Check size={16} /> : <Copy size={16} />}</button></div>
    </div>
  );
}

export default function Identifiers({ isomeric, canonical }: { isomeric: string; canonical: string }) {
  return <div className="identifiers"><Identifier label={copy.isomericSmiles} value={isomeric} /><Identifier label={copy.canonicalSmiles} value={canonical} /></div>;
}
