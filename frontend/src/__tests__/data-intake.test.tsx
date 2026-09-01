import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DataIntakePage from '../pages/DataIntakePage';
import type { AppMeta } from '../types';

const meta = {
  label_names: ['fruity', 'green'],
  data_foundation: { available: true, label_semantics: ['PRESENT', 'ABSENT', 'UNASSESSED'], intensity_scale: [0, 10] },
} as unknown as AppMeta;

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ versions: [] }) }));
});

afterEach(() => vi.unstubAllGlobals());

test('manual intake preserves tri-state semantics and exposes batch workflow', async () => {
  const user = userEvent.setup();
  render(<DataIntakePage meta={meta} />);

  const emptySnapshots = await screen.findByText('No immutable snapshots have been created yet.');
  await waitFor(() => expect(emptySnapshots).toBeVisible());
  const intensity = screen.getByRole('slider', { name: /Intensity/ });
  expect(intensity).toBeEnabled();
  await user.click(screen.getByRole('button', { name: 'Unassessed' }));
  expect(intensity).toBeDisabled();

  await user.click(screen.getByRole('tab', { name: 'CSV / XLSX import' }));
  expect(screen.getByText('Choose a CSV or XLSX file')).toBeVisible();
  expect(screen.getByRole('link', { name: 'Download template' })).toHaveAttribute('href', '/api/v1/data/templates?format=csv');
});
