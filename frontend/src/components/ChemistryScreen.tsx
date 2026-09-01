import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import { copy } from '../copy';
import type { ChemicalScreen, DisplayDescriptors } from '../types';

const status = {
  PASS: { label: copy.chemistryPass, Icon: CheckCircle2 },
  REVIEW: { label: copy.chemistryReview, Icon: AlertTriangle },
  REJECT: { label: copy.chemistryReject, Icon: XCircle }
};

export default function ChemistryScreen({ screen, descriptors, compact = false }: {
  screen: ChemicalScreen;
  descriptors: DisplayDescriptors;
  compact?: boolean;
}) {
  const current = status[screen.decision];
  const metrics = [
    [copy.formula, descriptors.formula],
    [copy.exactMw, `${descriptors.exact_mw.toFixed(3)} Da`],
    [copy.logP, descriptors.log_p.toFixed(2)],
    [copy.tpsa, `${descriptors.tpsa.toFixed(1)} Å²`],
    [copy.rotatableBonds, descriptors.rotatable_bonds],
    [copy.heavyAtoms, descriptors.heavy_atoms],
    [copy.saScore, descriptors.sa_score.toFixed(2)],
    [copy.volatilityTier, descriptors.estimated_volatility_tier]
  ];
  return (
    <div className={`chemistry-screen decision-${screen.decision.toLowerCase()} ${compact ? 'is-compact' : ''}`}>
      <h3>{copy.chemistryScreen}</h3>
      <div className="screen-status"><current.Icon size={24} /><span>{current.label}</span></div>
      {screen.is_macrocycle && <span className="macrocycle-label">{copy.macrocycleProfile} · {screen.macrocycle_ring_size}-member ring</span>}
      <dl className="metric-grid">
        {metrics.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>
      {screen.reasons.length > 0 && screen.decision !== 'PASS' && <ul className="screen-reasons">{screen.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>}
    </div>
  );
}
