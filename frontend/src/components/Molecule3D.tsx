import { ChevronLeft, ChevronRight, Maximize2, RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { copy } from '../copy';
import type { ConformerEnsemble } from '../types';
import Panel from './Panel';

export default function Molecule3D({ result, compact = false }: { result: ConformerEnsemble; compact?: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [renderError, setRenderError] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selected = result.conformers[selectedIndex];

  useEffect(() => setSelectedIndex(0), [result.conformers[0]?.molblock]);

  useEffect(() => {
    let disposed = false;
    if (!host.current || !result.available || !selected?.molblock) return;
    import('3dmol').then(($3Dmol) => {
      if (disposed || !host.current) return;
      try {
        host.current.innerHTML = '';
        const viewer = $3Dmol.createViewer(host.current, { backgroundColor: '#071015' });
        viewer.addModel(selected.molblock, 'mol');
        viewer.setStyle({}, {
          stick: { radius: 0.16, colorscheme: 'Jmol' },
          sphere: { scale: 0.28, colorscheme: 'Jmol' }
        });
        viewer.zoomTo();
        viewer.zoom(0.88);
        viewer.render();
        viewerRef.current = viewer;
        setRenderError(false);
      } catch {
        setRenderError(true);
      }
    }).catch(() => setRenderError(true));
    return () => {
      disposed = true;
      viewerRef.current?.clear?.();
      viewerRef.current = null;
    };
  }, [result.available, selected?.molblock]);

  const body = !result.available || !selected || renderError ? (
    <div className="conformer-fallback"><span>{copy.conformerUnavailable}</span></div>
  ) : <div ref={host} className="three-d-host" />;

  const navigation = result.available && result.conformers.length > 0 ? (
    <div className="conformer-navigation">
      <button
        type="button"
        onClick={() => setSelectedIndex((index) => Math.max(0, index - 1))}
        disabled={selectedIndex === 0}
        aria-label={copy.previousConformer}
      ><ChevronLeft size={17} /></button>
      <label>
        <span className="sr-only">{copy.conformerSelector}</span>
        <select value={selectedIndex} onChange={(event) => setSelectedIndex(Number(event.target.value))} aria-label={copy.conformerSelector}>
          {result.conformers.map((_, index) => <option value={index} key={index}>Conformer {index + 1} {copy.conformerOf} {result.conformers.length}</option>)}
        </select>
      </label>
      <button
        type="button"
        onClick={() => setSelectedIndex((index) => Math.min(result.conformers.length - 1, index + 1))}
        disabled={selectedIndex === result.conformers.length - 1}
        aria-label={copy.nextConformer}
      ><ChevronRight size={17} /></button>
    </div>
  ) : null;

  const footer = result.available && selected ? (
    <div className="conformer-metadata">
      <strong>{copy.relativeEnergy} ΔE {selected.relative_energy.toFixed(2)} kcal/mol</strong>
      <span>{result.method} · {result.requested_count} / {result.embedded_count} / {result.converged_count} {copy.ensembleCounts}</span>
      <small>{copy.ensembleCaption}</small>
    </div>
  ) : null;

  if (compact) return <div className="compact-3d">{body}{navigation}{footer}</div>;
  return (
    <Panel title={copy.structure3d} className="conformer-panel" actions={
      <div className="panel-actions">
        <button type="button" onClick={() => { viewerRef.current?.zoomTo(); viewerRef.current?.zoom(0.88); viewerRef.current?.render(); }} aria-label={copy.reset3d}><RotateCcw size={17} /></button>
        <button type="button" onClick={() => host.current?.requestFullscreen()} aria-label={copy.fullscreen3d}><Maximize2 size={17} /></button>
      </div>
    }>
      {body}
      {navigation}
      {footer}
    </Panel>
  );
}
