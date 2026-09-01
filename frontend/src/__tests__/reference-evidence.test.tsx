import { render, screen } from '@testing-library/react';
import ReferenceEvidencePanel from '../components/ReferenceEvidencePanel';
import type { ReferenceEvidence } from '../types';

const base: ReferenceEvidence = {
  provider: 'PUBCHEM',
  status: 'NO_MATCH',
  match_level: null,
  queried_identifier: 'CCO',
  record_ids: [],
  record_urls: [],
  checked_at: '2026-09-01T00:00:00Z',
  source_version: 'PUG_REST_same_stereo_isotope',
  error_code: null
};

test('reference evidence distinguishes no match from global novelty', () => {
  render(
    <ReferenceEvidencePanel
      checks={[
        base,
        { ...base, provider: 'TGSC', status: 'NOT_CONFIGURED', queried_identifier: null, source_version: null }
      ]}
      gate={{ status: 'PASS', blocking_providers: [], reason_code: null }}
    />
  );

  expect(screen.getByText('No matching record found in the configured reference sources.')).toBeVisible();
  expect(screen.getByText('No match in PubChem')).toBeVisible();
  expect(screen.getByText('Not configured')).toBeVisible();
  expect(screen.queryByText(/globally novel/i)).not.toBeInTheDocument();
});

test('catalog match exposes match level and licensed record link', () => {
  render(
    <ReferenceEvidencePanel
      checks={[{
        ...base,
        provider: 'SCENTREE',
        status: 'MATCH',
        match_level: 'EXACT_CONNECTIVITY',
        record_ids: ['record-1'],
        record_urls: ['https://licensed.example/record-1']
      }]}
      gate={{ status: 'BLOCKED_MATCH', blocking_providers: ['SCENTREE'], reason_code: 'KNOWN_REFERENCE_MATCH' }}
    />
  );

  expect(screen.getByText('Known in fragrance catalog')).toBeVisible();
  expect(screen.getByText('Exact connectivity')).toBeVisible();
  expect(screen.getByRole('link', { name: 'Open SCENTREE record' })).toHaveAttribute(
    'href',
    'https://licensed.example/record-1'
  );
});
