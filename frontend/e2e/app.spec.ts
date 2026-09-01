import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="520" viewBox="0 0 720 520"><rect width="720" height="520" fill="#f7f8f7"/><path d="M190 300 280 245 370 300 460 245 540 300" stroke="#111" stroke-width="7" fill="none"/><text x="365" y="288" fill="#e32636" font-size="36">O</text></svg>';

const screen = {
  decision: 'PASS', reason_codes: ['PROFILE_ACCEPTED'], reasons: ['Within the configured screening profile'], descriptors: { exact_mw: 212.141, log_p: 2.33, tpsa: 43.4, rotatable_bonds: 5, heavy_atoms: 15, sa_score: 3.27 }, is_macrocycle: false, macrocycle_ring_size: null, macrocycle_carbon_fraction: 0, macrocycle_heteroatoms: 0, alerts: []
};

const display = { formula: 'C12H20O3', exact_mw: 212.141, log_p: 2.33, tpsa: 43.4, rotatable_bonds: 5, heavy_atoms: 15, sa_score: 3.27, estimated_volatility_tier: 'Middle', volatility_basis: 'MW-based estimate' };
const referenceMeta = {
  providers: [
    { provider: 'PUBCHEM', display_name: 'PubChem/NCBI', source_type: 'STRUCTURAL_IDENTITY', enabled: true, external: true, query_types: ['ISOMERIC_SMILES'], dataset_version: null, license_status: 'PUBLIC_API', configuration_error: null },
    { provider: 'TGSC', display_name: 'The Good Scents Company', source_type: 'FRAGRANCE_CATALOG', enabled: false, external: false, query_types: ['FULL_INCHIKEY'], dataset_version: null, license_status: 'NOT_CONFIGURED', configuration_error: null },
    { provider: 'SCENTREE', display_name: 'ScenTree', source_type: 'FRAGRANCE_CATALOG', enabled: false, external: false, query_types: ['FULL_INCHIKEY'], dataset_version: null, license_status: 'NOT_CONFIGURED', configuration_error: null }
  ],
  required_external_consents: ['PUBCHEM'],
  shortlist_policy: 'CHEMISTRY_PASS_AND_ALL_ENABLED_REFERENCES_NO_MATCH'
};
const noMatchChecks = [
  { provider: 'PUBCHEM', status: 'NO_MATCH', match_level: null, queried_identifier: 'CCO', record_ids: [], record_urls: [], checked_at: '2026-09-01T00:00:00Z', source_version: 'PUG_REST_same_stereo_isotope', error_code: null },
  { provider: 'TGSC', status: 'NOT_CONFIGURED', match_level: null, queried_identifier: null, record_ids: [], record_urls: [], checked_at: '2026-09-01T00:00:00Z', source_version: null, error_code: null },
  { provider: 'SCENTREE', status: 'NOT_CONFIGURED', match_level: null, queried_identifier: null, record_ids: [], record_urls: [], checked_at: '2026-09-01T00:00:00Z', source_version: null, error_code: null }
];
const passGate = { status: 'PASS', blocking_providers: [], reason_code: null };
const notRunGate = { status: 'NOT_RUN', blocking_providers: [], reason_code: 'REFERENCE_CHECK_NOT_RUN' };

const facets = ['Animalic', 'Citrus', 'Floral', 'Fruity', 'Green', 'Herbal', 'Industrial', 'Mineral', 'Soulful', 'Sweet/Balsamic', 'Woody'].map((name, index) => ({ name, probability: (index + 2) / 20 }));
const ensemble = {
  available: true, method: 'MMFF94s', requested_count: 50, embedded_count: 42, converged_count: 40,
  is_macrocycle: false, error: null,
  conformers: Array.from({ length: 5 }, (_, index) => ({ molblock: `test-${index}`, relative_energy: index * 0.4 }))
};
const stereoOptions = [
  'CCCC[C@H]1CC(=O)C[C@@H]1CC(=O)OC',
  'CCCC[C@@H]1CC(=O)C[C@@H]1CC(=O)OC',
  'CCCC[C@H]1CC(=O)C[C@H]1CC(=O)OC',
  'CCCC[C@@H]1CC(=O)C[C@H]1CC(=O)OC'
].map((isomeric_smiles, index) => ({ isomeric_smiles, cip_summary: index % 2 ? 'C5 S · C6 R' : 'C5 R · C6 S', structure_2d_svg: svg }));

async function mockApi(page: Page) {
  await page.route('**/api/v1/meta', (route) => route.fulfill({ json: { label_names: ['floral', 'jasmine', 'musk', 'woody'], taxonomy_version: '1.2', projection_name: 'Osmo-compatible projection', generation_limits: { required_candidates: 5, shortlist_count: 3, max_attempts: 200, max_seconds: 120, max_event_lines: 30, candidate_stereo_limit: 4 }, conformer_ensemble: { normal_sampling_count: 50, macrocycle_sampling_count: 100, max_displayed: 5, normal_cluster_rmsd: 0.75, macrocycle_cluster_rmsd: 1, cache_size: 128 }, stereo: { analysis_option_limit: 16, candidate_variant_limit: 4 }, capabilities: { structure_2d: true, conformer_3d: true }, data_foundation: { available: true, label_semantics: ['PRESENT', 'ABSENT', 'UNASSESSED'], intensity_scale: [0, 10] }, reference_verification: referenceMeta, models: { judge: { model_version: 'judge-v1-legacy' }, creator: { model_version: 'creator-v1-legacy' } } } }));
  await page.route('**/api/v1/datasets/versions', (route) => route.fulfill({ json: { versions: [] } }));
  await page.route('**/api/v1/analysis', (route) => {
    const request = route.request().postDataJSON() as { smiles: string };
    const resolved = request.smiles.includes('@');
    return route.fulfill({ json: resolved ? {
      analysis_state: 'COMPLETE', unresolved_stereo_elements: 0,
      input_smiles: request.smiles, identifiers: { isomeric_smiles: request.smiles, canonical_smiles: 'CCCCC1CC(=O)CC1CC(=O)OC' }, structure_2d_svg: svg,
      conformer_ensemble: ensemble, stereo_options: [], chemistry_screen: screen, display_descriptors: display,
      reference_checks: [], reference_gate: notRunGate,
      predicted_odor_profile: { top_descriptors: [{ name: 'floral', probability: 0.8 }], taxonomy: { facets, textures: [{ name: 'Smooth', probability: 0.5 }], sensations: [{ name: 'Warm/Rich', probability: 0.4 }], projection_name: 'Osmo-compatible projection', taxonomy_version: '1.2' }, model_output: [{ name: 'floral', probability: 0.8 }, { name: 'jasmine', probability: 0.7 }, { name: 'musk', probability: 0.4 }, { name: 'woody', probability: 0.5 }, { name: 'odorless', probability: 0 }] }
    } : {
      analysis_state: 'STEREO_REQUIRED', unresolved_stereo_elements: 2,
      input_smiles: request.smiles, identifiers: { isomeric_smiles: request.smiles, canonical_smiles: request.smiles }, structure_2d_svg: svg,
      conformer_ensemble: null, stereo_options: stereoOptions, chemistry_screen: screen, display_descriptors: display, predicted_odor_profile: null,
      reference_checks: [], reference_gate: notRunGate
    } });
  });
  const progress = { phase: 'ACCEPTED', attempt: 18, accepted: 5, invalid: 4, duplicates: 2, rejected: 7, reviews: 1, found: 1, unverified: 0, reference_matches: 1, reference_unverified: 0, detail: null };
  const candidate = { isomeric_smiles: 'CCO', canonical_smiles: 'CCO', target_fit: 0.82, target_probabilities: [{ name: 'jasmine', probability: 0.84 }, { name: 'woody', probability: 0.8 }], supporting_descriptors: [{ name: 'floral', probability: 0.76 }], structure_2d_svg: svg, conformer_ensemble: ensemble, chemistry_screen: screen, display_descriptors: display, novelty: { status: 'NOT_FOUND', cids: [], error_code: null }, reference_checks: noMatchChecks, reference_gate: passGate };
  const complete = { shortlist: [candidate, { ...candidate, isomeric_smiles: 'CCN', canonical_smiles: 'CCN', target_fit: 0.78 }, { ...candidate, isomeric_smiles: 'CCC', canonical_smiles: 'CCC', target_fit: 0.71 }], review_queue: [], summary: { attempts: 18, accepted: 5, reviews: 1, invalid: 4, duplicates: 2, rejected: 7, found: 1, unverified: 0, reference_matches: 1, reference_unverified: 0, elapsed_seconds: 1.2, reached_attempt_limit: false, reached_time_limit: false } };
  await page.route('**/api/v1/candidates/stream', (route) => route.fulfill({ status: 200, contentType: 'text/event-stream', body: `event: progress\ndata: ${JSON.stringify(progress)}\n\nevent: complete\ndata: ${JSON.stringify(complete)}\n\n` }));
}

test.beforeEach(async ({ page }) => { await mockApi(page); });

test('Hedione resolves stereo before prediction and navigates five conformers', async ({ page }) => {
  await page.goto('/analysis');
  await expect(page.getByRole('heading', { name: 'Stereo resolution required' })).toBeVisible();
  await expect(page.getByText('Odor prediction is locked until stereochemistry is resolved.')).toBeVisible();
  await page.getByRole('button', { name: 'Use this stereoisomer' }).first().click();
  await expect(page.getByRole('heading', { name: 'Interactive 3D conformer ensemble' })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Select conformer' })).toHaveValue('0');
  await page.getByRole('button', { name: 'Next conformer' }).click();
  await expect(page.getByRole('combobox', { name: 'Select conformer' })).toHaveValue('1');
  await expect(page.getByText(/ΔE 0.40 kcal\/mol/)).toBeVisible();
  await expect(page.getByText('Passed the configured screening profile')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Predicted odor profile' })).toBeVisible();
  await expect(page.locator('.radar-chart')).toBeVisible();
  const odorProfile = page.locator('.odor-profile-panel');
  await expect(odorProfile.getByText('Model output')).toBeVisible();
  await expect(page.locator('.analysis-inspector').getByText('Model output')).toHaveCount(0);
  await odorProfile.getByText('Model output').click();
  await expect(odorProfile.getByText('floral', { exact: true })).toBeVisible();
  await expect(odorProfile.getByText('odorless', { exact: true })).toHaveCount(0);
});

test('invalid SMILES uses the product error without technical stack output', async ({ page }) => {
  await page.unroute('**/api/v1/analysis');
  await page.route('**/api/v1/analysis', (route) => route.fulfill({
    status: 422,
    json: { detail: { code: 'INVALID_SMILES', message: 'This SMILES string could not be parsed. Check the structure and try again.' } }
  }));
  await page.goto('/analysis');
  await page.locator('#smiles-input').fill('not-smiles');
  await page.getByRole('button', { name: 'Analyze molecule' }).click();
  await expect(page.getByRole('alert')).toContainText('This SMILES string could not be parsed. Check the structure and try again.');
  await expect(page.getByRole('alert')).not.toContainText('Traceback');
});

test('candidate consent gates generation and stream creates shortlist', async ({ page }) => {
  await page.goto('/candidates');
  const generate = page.getByRole('button', { name: 'Generate candidates' });
  await expect(generate).toBeDisabled();
  await expect(page.getByText('I understand that candidate Isomeric SMILES will be sent to PubChem/NCBI for identity checks.')).toBeVisible();
  await page.getByRole('checkbox').check();
  await expect(generate).toBeEnabled();
  await generate.click();
  await expect(page.getByRole('heading', { name: 'Shortlisted candidates' })).toBeVisible();
  await expect(page.getByText('Candidate 01')).toBeVisible();
  await expect(page.getByText('No match in PubChem').first()).toBeVisible();
  if (process.env.QA_CAPTURE === '1' && test.info().project.name === 'desktop') {
    await page.waitForTimeout(900);
    await page.screenshot({ path: '../.qa_candidate_mock.png', fullPage: false });
  }
});

test('candidate results survive route navigation within the current app session', async ({ page }) => {
  await page.goto('/candidates');
  await page.getByRole('checkbox').check();
  await page.getByRole('button', { name: 'Generate candidates' }).click();
  await expect(page.getByText('Candidate 01')).toBeVisible();

  await page.locator('a[href="/analysis"]:visible').first().click();
  await expect(page).toHaveURL(/\/analysis$/);
  await page.locator('a[href="/candidates"]:visible').first().click();

  await expect(page).toHaveURL(/\/candidates$/);
  await expect(page.getByText('Candidate 01')).toBeVisible();
  await expect(page.getByRole('checkbox')).toBeChecked();
});

test('theme and responsive navigation remain usable', async ({ page }, testInfo) => {
  await page.goto('/analysis');
  await page.getByRole('button', { name: 'Dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  if (testInfo.project.name === 'mobile') {
    await expect(page.getByRole('navigation', { name: 'Mobile workspace navigation' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Open navigation' })).toBeVisible();
  } else {
    await expect(page.getByText('Projects').first().locator('..')).toHaveAttribute('aria-disabled', 'true');
  }
});

test('analysis has no serious or critical accessibility violations', async ({ page }) => {
  await page.goto('/analysis');
  await page.getByRole('button', { name: 'Use this stereoisomer' }).first().click();
  await expect(page.getByRole('heading', { name: 'Predicted odor profile' })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact ?? ''))).toEqual([]);
});

test('unavailable reference result is explained and not shortlisted', async ({ page }) => {
  const progress = { phase: 'REFERENCE_UNVERIFIED', attempt: 4, accepted: 0, invalid: 0, duplicates: 0, rejected: 0, reviews: 1, found: 0, unverified: 1, reference_matches: 0, reference_unverified: 1, detail: null };
  const complete = { shortlist: [], review_queue: [], summary: { attempts: 4, accepted: 0, reviews: 1, invalid: 0, duplicates: 0, rejected: 0, found: 0, unverified: 1, reference_matches: 0, reference_unverified: 1, elapsed_seconds: 0.3, reached_attempt_limit: false, reached_time_limit: false } };
  await page.unroute('**/api/v1/candidates/stream');
  await page.route('**/api/v1/candidates/stream', (route) => route.fulfill({ status: 200, contentType: 'text/event-stream', body: `event: progress\ndata: ${JSON.stringify(progress)}\n\nevent: complete\ndata: ${JSON.stringify(complete)}\n\n` }));
  await page.goto('/candidates');
  await page.getByRole('checkbox').check();
  await page.getByRole('button', { name: 'Generate candidates' }).click();
  await expect(page.getByText('Reference verification is incomplete. Unverified structures were not shortlisted.')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Shortlisted candidates' })).toHaveCount(0);
});

test('review structures remain in the chemistry review queue', async ({ page }) => {
  const reviewScreen = { ...screen, decision: 'REVIEW', reason_codes: ['HALOGEN_PRESENT'], reasons: ['Halogen present'] };
  const progress = { phase: 'REVIEW', attempt: 2, accepted: 0, invalid: 0, duplicates: 0, rejected: 0, reviews: 1, found: 0, unverified: 0, reference_matches: 0, reference_unverified: 0, detail: null };
  const complete = { shortlist: [], review_queue: [{ isomeric_smiles: 'CCCl', structure_2d_svg: svg, chemistry_screen: reviewScreen, review_category: 'CHEMISTRY', reference_checks: [], reference_gate: notRunGate }], summary: { attempts: 2, accepted: 0, reviews: 1, invalid: 0, duplicates: 0, rejected: 0, found: 0, unverified: 0, reference_matches: 0, reference_unverified: 0, elapsed_seconds: 0.2, reached_attempt_limit: false, reached_time_limit: false } };
  await page.unroute('**/api/v1/candidates/stream');
  await page.route('**/api/v1/candidates/stream', (route) => route.fulfill({ status: 200, contentType: 'text/event-stream', body: `event: progress\ndata: ${JSON.stringify(progress)}\n\nevent: complete\ndata: ${JSON.stringify(complete)}\n\n` }));
  await page.goto('/candidates');
  await page.getByRole('checkbox').check();
  await page.getByRole('button', { name: 'Generate candidates' }).click();
  await page.getByText('Review queue').click();
  await expect(page.getByText('CCCl')).toBeVisible();
  await expect(page.getByText('Halogen present')).toBeVisible();
});

test('Stop run cancels an active candidate stream', async ({ page }) => {
  const progress = { phase: 'SAMPLING', attempt: 1, accepted: 0, invalid: 0, duplicates: 0, rejected: 0, reviews: 0, found: 0, unverified: 0, reference_matches: 0, reference_unverified: 0, detail: null };
  await page.unroute('**/api/v1/candidates/stream');
  await page.route('**/api/v1/candidates/stream', (route) => route.fulfill({ status: 200, contentType: 'text/event-stream', body: `event: progress\ndata: ${JSON.stringify(progress)}\n\n` }));
  await page.goto('/candidates');
  await page.getByRole('checkbox').check();
  await page.getByRole('button', { name: 'Generate candidates' }).click();
  const stop = page.getByRole('button', { name: 'Stop run' }).first();
  await expect(stop).toBeVisible();
  await stop.click();
  await expect(page.getByRole('button', { name: 'Stop run' })).toHaveCount(0);
});

test('data intake preserves unassessed state and reaches batch validation', async ({ page }) => {
  await page.goto('/data/intake');
  await expect(page.getByRole('heading', { name: 'Sensory data intake' })).toBeVisible();
  const intensity = page.getByRole('slider', { name: /Intensity/ });
  await expect(intensity).toBeEnabled();
  await page.getByRole('button', { name: 'Unassessed', exact: true }).click();
  await expect(intensity).toBeDisabled();
  await page.getByRole('tab', { name: 'CSV / XLSX import' }).click();
  await expect(page.getByText('Choose a CSV or XLSX file')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Download template' })).toHaveAttribute('href', '/api/v1/data/templates?format=csv');
});
