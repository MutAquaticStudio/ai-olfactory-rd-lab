import { Laptop, Moon, Sun } from 'lucide-react';
import { copy } from '../copy';
import { useTheme } from '../theme';
import type { ThemePreference } from '../types';

const options: Array<{ value: ThemePreference; label: string; icon: typeof Sun }> = [
  { value: 'system', label: copy.system, icon: Laptop },
  { value: 'light', label: copy.light, icon: Sun },
  { value: 'dark', label: copy.dark, icon: Moon }
];

export default function ThemeControl() {
  const { preference, setPreference } = useTheme();
  return (
    <div className="theme-control" role="group" aria-label={copy.colorTheme}>
      {options.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          className={preference === value ? 'is-selected' : ''}
          aria-pressed={preference === value}
          aria-label={`${label} theme`}
          title={`${label} theme`}
          onClick={() => setPreference(value)}
        >
          <Icon size={16} aria-hidden="true" />
        </button>
      ))}
    </div>
  );
}
