import { Download, Maximize2 } from 'lucide-react';
import { copy } from '../copy';
import Panel from './Panel';

export default function Molecule2D({ svg, compact = false }: { svg: string; compact?: boolean }) {
  const download = () => {
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'molecule-2d.svg';
    link.click();
    URL.revokeObjectURL(link.href);
  };
  if (compact) return <div className="molecule-svg compact" dangerouslySetInnerHTML={{ __html: svg }} />;
  return (
    <Panel title={copy.structure2d} className="structure-panel" actions={
      <div className="panel-actions">
        <button type="button" onClick={download} aria-label={copy.download2d}><Download size={17} /></button>
        <button type="button" onClick={() => document.querySelector('.molecule-svg')?.requestFullscreen()} aria-label={copy.fullscreen2d}><Maximize2 size={17} /></button>
      </div>
    }>
      <div className="molecule-svg" dangerouslySetInnerHTML={{ __html: svg }} />
    </Panel>
  );
}
