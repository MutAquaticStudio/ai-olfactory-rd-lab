import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import AppShell from '../components/AppShell';
import { ThemeProvider } from '../theme';

test('navigation exposes live routes and disabled coming-soon modules', () => {
  render(<ThemeProvider><MemoryRouter initialEntries={['/analysis']}><AppShell><div>content</div></AppShell></MemoryRouter></ThemeProvider>);
  expect(screen.getAllByRole('link', { name: 'Molecule analysis' }).length).toBeGreaterThan(0);
  expect(screen.getAllByRole('link', { name: 'Candidate design' }).length).toBeGreaterThan(0);
  expect(screen.getAllByRole('link', { name: 'Data intake' }).length).toBeGreaterThan(0);
  expect(screen.getByText('Projects').closest('[aria-disabled="true"]')).toBeInTheDocument();
  expect(screen.getByText('Library').closest('[aria-disabled="true"]')).toBeInTheDocument();
  expect(screen.getByText('Batch runs').closest('[aria-disabled="true"]')).toBeInTheDocument();
});
