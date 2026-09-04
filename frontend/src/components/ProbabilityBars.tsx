import { chemesthesis, copy } from '../copy';
import type { NamedProbability } from '../types';

function displayedPercent(probability: number) {
  return Number((Math.max(0, Math.min(100, probability * 100))).toFixed(1));
}

export default function ProbabilityBars({ items, limit, showChemesthesis = false, hideZero = false, calibrated = true }: {
  items: NamedProbability[];
  limit?: number;
  showChemesthesis?: boolean;
  hideZero?: boolean;
  calibrated?: boolean;
}) {
  const visibleItems = hideZero
    ? items.filter((item) => displayedPercent(item.probability) > 0)
    : items;
  return (
    <div className="probability-bars">
      {visibleItems.slice(0, limit).map((item) => {
        const percent = Math.max(0, Math.min(100, item.probability * 100));
        return (
          <div className="probability-row" key={item.name}>
            <div className="probability-label"><span>{item.name}</span>{showChemesthesis && chemesthesis.has(item.name) && <small>{copy.chemesthesis}</small>}<strong>{calibrated ? `${percent.toFixed(1)}%` : `${item.probability.toFixed(3)} score`}</strong></div>
            <div className="probability-track" role="progressbar" aria-label={item.name} aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${percent}%` }} /></div>
          </div>
        );
      })}
    </div>
  );
}
