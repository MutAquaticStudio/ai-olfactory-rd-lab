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
import AcademicEvidencePanel from './AcademicEvidencePanel';
import SpotlightCard from '../vendor/reactbits/SpotlightCard';

export default function CandidateCard({ candidate, rank }: { candidate: RankedCandidate; rank: number }) {
  const [open, setOpen] = useState(false);
  const providerNames: Record<string, string> = {
    PUBCHEM: 'PubChem',
    TGSC: 'TGSC licensed snapshot',
    SCENTREE: 'ScenTree licensed snapshot'
  };
  const noMatchChecks = candidate.reference_checks.filter((item) => item.status === 'NO_MATCH');
  const match = candidate.target_match;
  const calibratedProbability = Boolean(match?.calibrated && match.uses_absolute_probability_gate);
  const displayedFit = match?.robust_target_fit ?? candidate.target_fit;
  return (
    <SpotlightCard className="candidate-spotlight"><article className={`candidate-card ${open ? 'is-open' : ''}`}>
      <button type="button" className="candidate-summary" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <strong className="candidate-rank">{rank}</strong>
        <Molecule2D svg={candidate.structure_2d_svg} compact />
        <div className="candidate-identity"><h3>{copy.candidate} {String(rank).padStart(2, '0')}</h3><div className="candidate-badges"><span><Leaf size={13} />{copy.newLocal}</span>{match && <span className={`match-badge match-${match.tier.toLowerCase()}`}>{match.met_requested_gate ? copy.strictMatch : copy.relaxedMatch}</span>}{noMatchChecks.map((item) => <span key={item.provider}><Search size={13} />{copy.noMatchProvider(providerNames[item.provider] ?? item.provider)}</span>)}</div></div>
        <code className="candidate-smiles">{candidate.isomeric_smiles}</code>
        <div className="fit-cell"><span>{copy.robustTargetFit}</span><strong>{calibratedProbability ? `${(displayedFit * 100).toFixed(1)}%` : `${displayedFit.toFixed(3)} score`}</strong><div><i style={{ width: `${Math.max(0, Math.min(100, displayedFit * 100))}%` }} /></div></div>
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
          {match && <section className={`target-match-panel match-${match.tier.toLowerCase()}`}><header><div><h3>{match.met_requested_gate ? copy.strictMatch : copy.relaxedMatch}</h3><p>{match.calibrated ? copy.calibrated : copy.uncalibrated}</p></div><strong>{calibratedProbability ? `${(match.robust_target_fit * 100).toFixed(1)}%` : `${match.robust_target_fit.toFixed(3)} score`}</strong></header><dl><div><dt>{copy.requestedThreshold}</dt><dd>{(match.requested_fit_floor * 100).toFixed(1)}%</dd></div><div><dt>{copy.appliedThreshold}</dt><dd>{(match.applied_fit_floor * 100).toFixed(1)}%</dd></div><div><dt>{copy.reliability}</dt><dd>{candidate.reliability_state.replaceAll('_', ' ')}</dd></div>{candidate.training_similarity !== null && <div><dt>{copy.nearestSimilarity}</dt><dd>{candidate.training_similarity.toFixed(3)}</dd></div>}{candidate.prediction_provenance?.model_version && <div><dt>{copy.modelVersion}</dt><dd>{candidate.prediction_provenance.model_version}</dd></div>}{candidate.prediction_provenance?.dataset_version && <div><dt>{copy.datasetVersion}</dt><dd>{candidate.prediction_provenance.dataset_version}</dd></div>}</dl><div className="target-evidence-list">{match.targets.map((target) => <div key={target.name}><span><strong>{target.name}</strong><small>{target.maturity.replaceAll('_', ' ')}</small></span><code>{calibratedProbability && target.maturity === 'SUPPORTED' ? `${(target.conservative_probability * 100).toFixed(1)}%` : `${target.conservative_probability.toFixed(3)} score`}</code><small>± {target.uncertainty.toFixed(3)} · floor {target.applied_floor.toFixed(3)}</small></div>)}</div></section>}
          <div className="candidate-profile"><div><h3>{copy.targetDescriptors}</h3><ProbabilityBars items={candidate.target_probabilities} calibrated={calibratedProbability} /></div><div><h3>{copy.supportingDescriptors}</h3><ProbabilityBars items={candidate.supporting_descriptors} calibrated={false} /></div></div>
          <ReferenceEvidencePanel checks={candidate.reference_checks} gate={candidate.reference_gate} />
          {candidate.academic_evidence && <AcademicEvidencePanel summary={candidate.academic_evidence} />}
          <section className="synthesis-evidence"><h3>{copy.synthesisEvidence}</h3><strong>{candidate.synthesis_assessment.status.replaceAll('_', ' ')}</strong>{candidate.synthesis_assessment.warnings.map((warning) => <p key={warning}>{warning}</p>)}</section>
          <p className="panel-note">{copy.verificationNote} {copy.screenNote}</p>
        </div>
      )}
    </article></SpotlightCard>
  );
}
