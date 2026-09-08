import { ChevronDown } from 'lucide-react';
import { copy } from '../copy';
import type { AnalysisResult } from '../types';
import ChemistryScreen from './ChemistryScreen';
import AcademicEvidencePanel from './AcademicEvidencePanel';
import Identifiers from './Identifiers';

export default function Inspector({ result }: { result: AnalysisResult }) {
  const integrity = result.analysis_state === 'COMPLETE' ? result.prediction_v2 : null;
  const shadow = result.analysis_state === 'COMPLETE' ? result.shadow_prediction : null;
  const productionTop = integrity
    ? [...integrity.presence_predictions].sort((a, b) => b.probability - a.probability).slice(0, 5)
    : [];
  const shadowTop = shadow
    ? [...shadow.presence_predictions].sort((a, b) => b.probability - a.probability).slice(0, 5)
    : [];
  const reliabilityCopy = integrity ? {
    IN_DOMAIN: copy.inDomain,
    LIMITED_EVIDENCE: copy.limitedEvidence,
    OUT_OF_DOMAIN: copy.outOfDomain
  }[integrity.reliability_state] : null;
  return (
    <aside className="analysis-inspector">
      <section className="inspector-section"><h2>{copy.structuralIdentifiers}</h2><Identifiers isomeric={result.identifiers.isomeric_smiles} canonical={result.identifiers.canonical_smiles} /></section>
      <section className="inspector-section"><ChemistryScreen screen={result.chemistry_screen} descriptors={result.display_descriptors} /></section>
      <section className="inspector-section"><AcademicEvidencePanel summary={result.academic_evidence ?? null} /></section>
      <details className="disclosure technical-disclosure"><summary>{copy.technicalDetails}<ChevronDown size={17} /></summary><div className="disclosure-body technical-copy">
        {integrity ? <dl className="integrity-grid">
          <div><dt>{copy.modelVersion}</dt><dd>{integrity.model_version}</dd></div>
          <div><dt>{copy.datasetVersion}</dt><dd>{integrity.dataset_version}</dd></div>
          <div><dt>{copy.calibration}</dt><dd>{integrity.calibrated ? integrity.calibration_version : copy.uncalibrated}</dd></div>
          <div><dt>{copy.reliability}</dt><dd className={`reliability-${integrity.reliability_state.toLowerCase()}`}>{reliabilityCopy}</dd></div>
          <div><dt>{copy.nearestSimilarity}</dt><dd>{integrity.nearest_training_similarity === null ? '—' : `${(integrity.nearest_training_similarity * 100).toFixed(1)}%`}</dd></div>
        </dl> : null}
        {shadow ? <section className="shadow-comparison" aria-label={copy.shadowComparison}>
          <div className="shadow-comparison-heading"><h3>{copy.shadowComparison}</h3><span>{copy.shadowBadge}</span></div>
          <dl className="integrity-grid">
            <div><dt>{copy.modelVersion}</dt><dd>{shadow.model_version}</dd></div>
            <div><dt>{copy.datasetVersion}</dt><dd>{shadow.dataset_version}</dd></div>
            <div><dt>{copy.calibration}</dt><dd>{shadow.calibrated ? shadow.calibration_version : copy.uncalibrated}</dd></div>
            <div><dt>{copy.reliability}</dt><dd>{shadow.reliability_state}</dd></div>
            <div><dt>{copy.nearestSimilarity}</dt><dd>{shadow.nearest_training_similarity === null ? '—' : `${(shadow.nearest_training_similarity * 100).toFixed(1)}%`}</dd></div>
          </dl>
          <div className="shadow-output-grid">
            <div><h4>{copy.productionModel}</h4>{productionTop.map((item) => <p key={item.name}><span>{item.name}</span><strong>{item.probability.toFixed(3)} score</strong></p>)}</div>
            <div><h4>{copy.researchModel}</h4>{shadowTop.map((item) => <p key={item.name}><span>{item.name}</span><strong>{(item.probability * 100).toFixed(1)}%</strong></p>)}</div>
          </div>
          {shadow.limitations.map((limitation) => <p key={limitation}>{limitation}</p>)}
        </section> : result.analysis_state === 'COMPLETE' && result.shadow_prediction_error ? <p>{copy.shadowUnavailable}: {result.shadow_prediction_error}</p> : null}
        <p>{copy.technicalOdorModel}</p><p>{copy.technicalSampler}</p><p>{copy.technicalEvidence}</p>
        {integrity?.limitations.map((limitation) => <p key={limitation}>{limitation}</p>)}
      </div></details>
    </aside>
  );
}
