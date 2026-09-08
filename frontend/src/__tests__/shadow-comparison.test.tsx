import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import Inspector from '../components/Inspector';
import type { AnalysisResult, PredictionV2 } from '../types';

const prediction = (model: string, calibrated: boolean): PredictionV2 => ({
  model_version: model,
  dataset_version: 'legacy-clean-3522',
  calibration_version: calibrated ? 'ensemble-platt-tier-v1' : 'uncalibrated',
  model_status: calibrated ? 'SHADOW_ONLY' : 'LEGACY_BASELINE',
  calibrated,
  nearest_training_similarity: 0.72,
  reliability_state: 'IN_DOMAIN',
  presence_predictions: [
    { name: 'floral', probability: 0.61, expected_intensity: null, uncertainty: calibrated ? 0.03 : null, decision_threshold: calibrated ? 0.3 : null },
    { name: 'woody', probability: 0.22, expected_intensity: null, uncertainty: calibrated ? 0.02 : null, decision_threshold: calibrated ? 0.25 : null }
  ],
  limitations: []
});

describe('Judge shadow comparison', () => {
  it('keeps the research result inside Technical details', () => {
    const result = {
      analysis_state: 'COMPLETE',
      identifiers: { isomeric_smiles: 'CCO', canonical_smiles: 'CCO' },
      chemistry_screen: { decision: 'PASS', reason_codes: [], reasons: [], descriptors: {}, is_macrocycle: false, macrocycle_ring_size: null, macrocycle_carbon_fraction: 0, macrocycle_heteroatoms: 0, alerts: [] },
      display_descriptors: {
        formula: 'C2H6O', exact_mw: 46.0419, log_p: -0.001, tpsa: 20.2,
        rotatable_bonds: 0, heavy_atoms: 3, sa_score: 1.2,
        estimated_volatility_tier: 'Top', volatility_basis: 'MW-based estimate'
      },
      academic_evidence: null,
      prediction_v2: prediction('judge-v1-legacy', false),
      shadow_prediction: prediction('judge-v2-shadow-test', true)
    } as unknown as AnalysisResult;

    render(<Inspector result={result} />);

    expect(screen.getByText('Technical details').closest('details')).toContainElement(
      screen.getByText('Research model comparison')
    );
    expect(screen.getByText('Shadow model — not used for candidate ranking')).toBeInTheDocument();
    expect(screen.getByText('judge-v2-shadow-test')).toBeInTheDocument();
  });
});
