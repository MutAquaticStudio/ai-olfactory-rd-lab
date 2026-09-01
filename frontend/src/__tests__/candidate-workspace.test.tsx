import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import { Link, MemoryRouter } from 'react-router-dom';
import type { AppMeta, GenerationComplete } from '../types';
import type { CandidateWorkspaceState } from '../pages/CandidatesPage';
import { getMeta } from '../api';
import App from '../App';

const meta: AppMeta = {
  label_names: ['jasmine', 'woody'],
  taxonomy_version: '1.2',
  projection_name: 'Osmo-compatible projection',
  generation_limits: { required_candidates: 5, shortlist_count: 3, max_attempts: 200, max_seconds: 120, max_event_lines: 30, candidate_stereo_limit: 4 },
  conformer_ensemble: { normal_sampling_count: 50, macrocycle_sampling_count: 100, max_displayed: 5, normal_cluster_rmsd: 0.75, macrocycle_cluster_rmsd: 1, cache_size: 128 },
  stereo: { analysis_option_limit: 16, candidate_variant_limit: 4 },
  capabilities: { structure_2d: true, conformer_3d: true },
  data_foundation: { available: true, label_semantics: ['PRESENT', 'ABSENT', 'UNASSESSED'], intensity_scale: [0, 10] },
  reference_verification: { providers: [], required_external_consents: [], shortlist_policy: 'PASS' },
  models: {}
};

vi.mock('../api', () => ({ getMeta: vi.fn() }));
vi.mock('../components/AppShell', () => ({ default: ({ children }: { children: ReactNode }) => <>{children}</> }));
vi.mock('../pages/AnalysisPage', () => ({
  default: () => <div>Analysis route <Link to="/candidates">Return to candidates</Link></div>
}));
vi.mock('../pages/DataIntakePage', () => ({ default: () => <div>Data intake route</div> }));
vi.mock('../pages/CandidatesPage', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../pages/CandidatesPage')>();
  return {
    ...actual,
    default: ({ workspace, setWorkspace }: {
      workspace: CandidateWorkspaceState;
      setWorkspace: Dispatch<SetStateAction<CandidateWorkspaceState>>;
    }) => (
      <div>
        <button type="button" onClick={() => setWorkspace((previous) => ({
          ...previous,
          result: {
            shortlist: [], review_queue: [],
            summary: {
              attempts: 1, accepted: 1, reviews: 0, invalid: 0, duplicates: 0, rejected: 0,
              found: 0, unverified: 0, reference_matches: 0, reference_unverified: 0,
              elapsed_seconds: 0.1, reached_attempt_limit: false, reached_time_limit: false
            }
          } satisfies GenerationComplete
        }))}>Store candidate result</button>
        {workspace.result ? <span>Candidate result retained</span> : null}
        <Link to="/analysis">Open analysis</Link>
      </div>
    )
  };
});

test('candidate workspace remains mounted at app level across route changes', async () => {
  const user = userEvent.setup();
  vi.mocked(getMeta).mockResolvedValue(meta);
  render(<MemoryRouter initialEntries={['/candidates']}><App /></MemoryRouter>);

  await user.click(await screen.findByRole('button', { name: 'Store candidate result' }));
  expect(screen.getByText('Candidate result retained')).toBeVisible();
  await user.click(screen.getByRole('link', { name: 'Open analysis' }));
  await user.click(screen.getByRole('link', { name: 'Return to candidates' }));

  expect(screen.getByText('Candidate result retained')).toBeVisible();
});
