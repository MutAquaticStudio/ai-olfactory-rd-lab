import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Molecule3D from '../components/Molecule3D';
import StereoResolution from '../components/StereoResolution';
import type { ConformerEnsemble, StereoRequiredAnalysis } from '../types';

vi.mock('3dmol', () => ({
  createViewer: () => ({
    addModel: vi.fn(),
    setStyle: vi.fn(),
    zoomTo: vi.fn(),
    zoom: vi.fn(),
    render: vi.fn(),
    clear: vi.fn()
  })
}));

const pending: StereoRequiredAnalysis = {
  analysis_state: 'STEREO_REQUIRED',
  input_smiles: 'CC(O)C(=O)O',
  identifiers: { isomeric_smiles: 'CC(O)C(=O)O', canonical_smiles: 'CC(O)C(=O)O' },
  structure_2d_svg: '<svg />',
  chemistry_screen: {
    decision: 'PASS', reason_codes: ['PROFILE_ACCEPTED'], reasons: ['Accepted'], descriptors: {},
    is_macrocycle: false, macrocycle_ring_size: null, macrocycle_carbon_fraction: 0,
    macrocycle_heteroatoms: 0, alerts: []
  },
  display_descriptors: {
    formula: 'C3H6O3', exact_mw: 90, log_p: -0.7, tpsa: 57, rotatable_bonds: 1,
    heavy_atoms: 6, sa_score: 2, estimated_volatility_tier: 'Top', volatility_basis: 'MW-based estimate'
  },
  unresolved_stereo_elements: 1,
  reference_checks: [],
  reference_gate: { status: 'NOT_RUN', blocking_providers: [], reason_code: 'REFERENCE_CHECK_NOT_RUN' },
  stereo_options: [
    { isomeric_smiles: 'C[C@H](O)C(=O)O', cip_summary: 'C2 R', structure_2d_svg: '<svg />' },
    { isomeric_smiles: 'C[C@@H](O)C(=O)O', cip_summary: 'C2 S', structure_2d_svg: '<svg />' }
  ],
  predicted_odor_profile: null,
  prediction_v2: null,
  conformer_ensemble: null
};

test('stereo selector keeps prediction locked and submits the selected isomer', async () => {
  const user = userEvent.setup();
  const select = vi.fn();
  render(<StereoResolution result={pending} onSelect={select} />);

  expect(screen.getByText('Odor prediction is locked until stereochemistry is resolved.')).toBeVisible();
  const buttons = screen.getAllByRole('button', { name: 'Use this stereoisomer' });
  await user.click(buttons[1]);
  expect(select).toHaveBeenCalledWith('C[C@@H](O)C(=O)O');
});

test('conformer navigation changes the displayed relative energy', async () => {
  const user = userEvent.setup();
  const ensemble: ConformerEnsemble = {
    available: true,
    method: 'MMFF94s',
    requested_count: 50,
    embedded_count: 42,
    converged_count: 40,
    is_macrocycle: false,
    error: null,
    conformers: [
      { molblock: 'first', relative_energy: 0 },
      { molblock: 'second', relative_energy: 1.25 }
    ]
  };
  render(<Molecule3D result={ensemble} />);

  expect(screen.getByText(/ΔE 0.00 kcal\/mol/)).toBeVisible();
  await user.click(screen.getByRole('button', { name: 'Next conformer' }));
  expect(screen.getByText(/ΔE 1.25 kcal\/mol/)).toBeVisible();
  expect(screen.getByRole('combobox', { name: 'Select conformer' })).toHaveValue('1');
  expect(screen.getByText('Computational conformer ensemble — not an experimental structure.')).toBeVisible();
});
