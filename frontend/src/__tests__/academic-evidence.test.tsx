import { render, screen } from '@testing-library/react';
import AcademicEvidencePanel from '../components/AcademicEvidencePanel';
import type { AcademicEvidenceSummary } from '../types';

const summary: AcademicEvidenceSummary = {
  query_isomeric_smiles: 'CCO',
  status: 'EXACT_MATCH',
  normalized_structure: null,
  matches: [{
    evidence_id: 'evidence-1',
    document: {
      paper_id: 'paper-1', title: 'An open odorant study', link: 'https://doi.org/10.1234/example',
      source: 'journal', doi: '10.1234/example', published_date: '2024', content_type: 'full_text',
      text_sha256: 'a'.repeat(64), source_type: 'PRIMARY_STUDY', license_status: 'OA_CONFIRMED', open_access: true
    },
    mention: {
      raw_value: 'CCO', kind: 'SMILES', page: 2, chunk_index: 3, span_start: 10, span_end: 13,
      evidence_excerpt: 'SMILES: CCO was evaluated at a controlled concentration.', confidence: 0.95,
      warnings: [], normalized: null
    },
    status: 'EXACT_MATCH', match_level: 'EXACT_STEREO', odor_descriptors: ['fruity'],
    presence_state: 'PRESENT', intensity: null, source_type: 'PRIMARY_STUDY', review_state: 'ACCEPTED',
    conflict_flags: [], created_at: '2026-09-01T00:00:00Z'
  }],
  conflicts: [],
  provenance: []
};

test('academic evidence panel shows citation provenance and review boundary', () => {
  render(<AcademicEvidencePanel summary={summary} />);
  expect(screen.getByText('Academic evidence')).toBeVisible();
  expect(screen.getByText('Exact structure evidence found')).toBeVisible();
  expect(screen.getByText('An open odorant study')).toBeVisible();
  expect(screen.getByText('Page 2')).toBeVisible();
  expect(screen.getByText(/does not establish safety, novelty/i)).toBeVisible();
});

test('academic evidence remains explicit when analysis has not run', () => {
  render(<AcademicEvidencePanel summary={null} />);
  expect(screen.getByText('Academic evidence is not run until stereochemistry is resolved.')).toBeVisible();
});
