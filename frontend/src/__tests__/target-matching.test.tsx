import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import CandidateCard from '../components/CandidateCard';
import TargetSelector from '../components/TargetSelector';
import type { RankedCandidate, TargetDescriptorMeta } from '../types';


const descriptor = (
  name: string,
  maturity: TargetDescriptorMeta['maturity'],
  selectable = true
): TargetDescriptorMeta => ({
  name,
  positive_support: maturity === 'SUPPORTED' ? 80 : 20,
  assessed_negative_support: maturity === 'SUPPORTED' ? 80 : 0,
  maturity,
  decision_threshold: maturity === 'SUPPORTED' ? 0.3 : 0.12,
  calibration_method: maturity === 'SUPPORTED' ? 'per_label_platt' : 'rare_tier_platt',
  selectable
});

test('target selector enforces three descriptors and disables insufficient evidence', async () => {
  const user = userEvent.setup();
  const metadata = [
    descriptor('floral', 'SUPPORTED'),
    descriptor('woody', 'SUPPORTED'),
    descriptor('citrus', 'SUPPORTED'),
    descriptor('musk', 'LIMITED_EVIDENCE'),
    descriptor('unsupported', 'INSUFFICIENT', false)
  ];
  function Harness() {
    const [selected, setSelected] = useState(['floral', 'woody']);
    return <TargetSelector labels={metadata.map((item) => item.name)} metadata={metadata} selected={selected} onChange={setSelected} maxTargets={3} />;
  }
  render(<Harness />);

  await user.click(screen.getByRole('button', { name: /citrus/i }));
  expect(screen.getByRole('button', { name: /musk/i })).toBeDisabled();
  expect(screen.getByRole('button', { name: /unsupported/i })).toBeDisabled();
});

test('uncalibrated relaxed candidate is shown as a score rather than a probability', () => {
  const candidate = {
    isomeric_smiles: 'CCO', canonical_smiles: 'CCO', target_fit: 0.15,
    target_probabilities: [{ name: 'musk', probability: 0.15 }],
    supporting_descriptors: [], structure_2d_svg: '<svg xmlns="http://www.w3.org/2000/svg"/>',
    conformer_ensemble: { available: false, method: null, requested_count: 50, embedded_count: 0, converged_count: 0, is_macrocycle: false, error: 'test', conformers: [] },
    chemistry_screen: { decision: 'PASS', reason_codes: [], reasons: [], descriptors: {}, is_macrocycle: false, macrocycle_ring_size: null, macrocycle_carbon_fraction: 0, macrocycle_heteroatoms: 0, alerts: [] },
    display_descriptors: { formula: 'C2H6O', exact_mw: 46, log_p: 0, tpsa: 20, rotatable_bonds: 0, heavy_atoms: 3, sa_score: 1, estimated_volatility_tier: 'Top', volatility_basis: 'MW' },
    novelty: { status: 'NOT_FOUND', cids: [], error_code: null }, reference_checks: [], reference_gate: { status: 'PASS', blocking_providers: [], reason_code: null },
    target_match: { target_fit: 0.15, robust_target_fit: 0.15, requested_fit_floor: 0.12, applied_fit_floor: 0.09, relaxation_factor: 0.75, tier: 'RELAXED', met_requested_gate: false, calibrated: false, uses_absolute_probability_gate: false, targets: [{ name: 'musk', probability: 0.15, uncertainty: 0, conservative_probability: 0.15, maturity: 'LIMITED_EVIDENCE', requested_floor: 0.12, applied_floor: 0.09, passed_requested_floor: true, passed_applied_floor: true }] },
    training_similarity: null, reliability_state: 'LIMITED_EVIDENCE',
    synthesis_assessment: { status: 'NOT_CONFIGURED', method: 'AiZynthFinder', time_limit_seconds: 300, route_found: null, route_steps: null, search_time_seconds: null, warnings: [] }, academic_evidence: null
  } satisfies RankedCandidate;

  render(<CandidateCard candidate={candidate} rank={1} />);

  expect(screen.getByText('Relaxed match — requested threshold not met')).toBeVisible();
  expect(screen.getByText('0.150 score')).toBeVisible();
  expect(screen.queryByText('15.0%')).not.toBeInTheDocument();
});
