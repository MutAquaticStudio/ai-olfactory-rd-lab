import { ChevronDown, Leaf, Search } from 'lucide-react';
import { useState } from 'react';
import { copy } from '../copy';
import type { RankedCandidate } from '../types';
import ChemistryScreen from './ChemistryScreen';
import Identifiers from './Identifiers';
import Molecule2D from './Molecule2D';
import Molecule3D from './Molecule3D';
import ProbabilityBars from './ProbabilityBars';
import ReferenceEvidencePanel from './ReferenceEvidencePanel';
import SpotlightCard from '../vendor/reactbits/SpotlightCard';

export default function CandidateCard({ candidate, rank }: { candidate: RankedCandidate; rank: number }) {
  const [open, setOpen] = useState(false);
  const providerNames: Record<string, string> = {
    PUBCHEM: 'PubChem',
    TGSC: 'TGSC licensed snapshot',
    SCENTREE: 'ScenTree licensed snapshot'
  };
  const noMatchChecks = candidate.reference_checks.filter((item) => item.status === 'NO_MATCH');
  return (
    <SpotlightCard className="candidate-spotlight"><article className={`candidate-card ${open ? 'is-open' : ''}`}>
      <button type="button" className="candidate-summary" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <strong className="candidate-rank">{rank}</strong>
        <Molecule2D svg={candidate.structure_2d_svg} compact />
        <div className="candidate-identity"><h3>{copy.candidate} {String(rank).padStart(2, '0')}</h3><div className="candidate-badges"><span><Leaf size={13} />{copy.newLocal}</span>{noMatchChecks.map((item) => <span key={item.provider}><Search size={13} />{copy.noMatchProvider(providerNames[item.provider] ?? item.provider)}</span>)}</div></div>
        <code className="candidate-smiles">{candidate.isomeric_smiles}</code>
        <div className="fit-cell"><span>{copy.targetFit}</span><strong>{(candidate.target_fit * 100).toFixed(1)}%</strong><div><i style={{ width: `${candidate.target_fit * 100}%` }} /></div></div>
        <span className="screen-chip">{copy.acceptedForScoring}</span>
        <ChevronDown className="candidate-chevron" aria-hidden="true" />
      </button>
      {open && (
        <div className="candidate-detail">
          <div className="candidate-visuals"><Molecule2D svg={candidate.structure_2d_svg} /><Molecule3D result={candidate.conformer_ensemble} /></div>
          <div className="candidate-data">
            <Identifiers isomeric={candidate.isomeric_smiles} canonical={candidate.canonical_smiles} />
            <ChemistryScreen screen={candidate.chemistry_screen} descriptors={candidate.display_descriptors} compact />
          </div>
          <div className="candidate-profile"><div><h3>{copy.targetDescriptors}</h3><ProbabilityBars items={candidate.target_probabilities} /></div><div><h3>{copy.supportingDescriptors}</h3><ProbabilityBars items={candidate.supporting_descriptors} /></div></div>
          <ReferenceEvidencePanel checks={candidate.reference_checks} gate={candidate.reference_gate} />
          <p className="panel-note">{copy.verificationNote} {copy.screenNote}</p>
        </div>
      )}
    </article></SpotlightCard>
  );
}
