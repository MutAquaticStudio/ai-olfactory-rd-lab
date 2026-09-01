import { Beaker, ClipboardPenLine, Clock3, FlaskConical, Folder, Library, Menu, X } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { copy } from '../copy';
import BrandMark from './BrandMark';
import ThemeControl from './ThemeControl';

const liveRoutes = [
  { to: '/analysis', label: copy.analysis, icon: Beaker },
  { to: '/candidates', label: copy.candidates, icon: FlaskConical },
  { to: '/data/intake', label: copy.dataIntake, icon: ClipboardPenLine }
];

const futureRoutes = [
  { label: copy.projects, icon: Folder },
  { label: copy.library, icon: Library },
  { label: copy.batchRuns, icon: Clock3 }
];

function Navigation({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="primary-nav" aria-label={copy.workspaceNavigation}>
      {liveRoutes.map(({ to, label, icon: Icon }) => (
        <NavLink key={to} to={to} onClick={onNavigate} className={({ isActive }) => isActive ? 'nav-link is-active' : 'nav-link'}>
          <Icon size={20} strokeWidth={1.8} aria-hidden="true" />
          <span>{label}</span>
        </NavLink>
      ))}
      <div className="nav-section-label">{copy.comingSoon}</div>
      {futureRoutes.map(({ label, icon: Icon }) => (
        <div key={label} className="nav-link is-disabled" aria-disabled="true" title={`${label} — ${copy.comingSoon}`}>
          <Icon size={20} strokeWidth={1.7} aria-hidden="true" />
          <span>{label}</span>
        </div>
      ))}
    </nav>
  );
}

export default function AppShell({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <BrandMark />
          <span>{copy.appTitle}</span>
        </div>
        <Navigation />
        <div className="sidebar-footer">
          <span className="avatar">RD</span>
          <span><strong>{copy.rdWorkspace}</strong><small>{copy.version}</small></span>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <button className="mobile-menu-button" type="button" onClick={() => setMenuOpen(true)} aria-label={copy.openNavigation}>
            <Menu aria-hidden="true" />
          </button>
          <div className="mobile-brand"><BrandMark size={28} /><strong>{copy.appTitle}</strong></div>
          <p>{copy.appSubtitle}</p>
          <ThemeControl />
        </header>
        <main className="main-content">{children}</main>
      </div>

      <nav className="bottom-nav" aria-label={copy.mobileNavigation}>
        {liveRoutes.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} className={({ isActive }) => isActive ? 'is-active' : ''}>
            <Icon size={21} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {menuOpen && (
        <div className="mobile-drawer-backdrop" onMouseDown={() => setMenuOpen(false)}>
          <aside className="mobile-drawer" onMouseDown={(event) => event.stopPropagation()} aria-label="Mobile navigation">
            <div className="drawer-header">
              <div className="brand-lockup"><BrandMark size={30} /><span>{copy.appTitle}</span></div>
              <button type="button" onClick={() => setMenuOpen(false)} aria-label={copy.closeNavigation}><X /></button>
            </div>
            <Navigation onNavigate={() => setMenuOpen(false)} />
          </aside>
        </div>
      )}
    </div>
  );
}
