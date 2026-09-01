import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from '../theme';
import ThemeControl from '../components/ThemeControl';

test('theme preference is selectable and persisted', async () => {
  localStorage.clear();
  render(<ThemeProvider><ThemeControl /></ThemeProvider>);
  await userEvent.click(screen.getByRole('button', { name: 'Dark theme' }));
  expect(localStorage.getItem('sms-theme')).toBe('dark');
  expect(document.documentElement.dataset.theme).toBe('dark');
  expect(screen.getByRole('button', { name: 'Dark theme' })).toHaveAttribute('aria-pressed', 'true');
});
