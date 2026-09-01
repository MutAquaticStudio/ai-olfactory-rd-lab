import { lazy, Suspense } from 'react';
import { useTheme } from '../theme';
import type { NamedProbability } from '../types';

const Plot = lazy(() => import('./RadarPlot'));

export function makeRadarSeries(facets: NamedProbability[]) {
  const labels = facets.map((item) => item.name);
  const values = facets.map((item) => item.probability * 100);
  return {
    labels,
    theta: labels.length ? [...labels, labels[0]] : [],
    radius: values.length ? [...values, values[0]] : []
  };
}

export default function RadarChart({ facets }: { facets: NamedProbability[] }) {
  const { resolved } = useTheme();
  const textColor = resolved === 'dark' ? '#d8e7e5' : '#284746';
  const gridColor = resolved === 'dark' ? '#355153' : '#d1dfdd';
  const { theta, radius } = makeRadarSeries(facets);
  return (
    <div className="radar-chart" role="img" aria-label="Primary olfactory facets radar chart">
      <Suspense fallback={<div className="chart-loading">Preparing radar chart</div>}>
        <Plot
          data={[{
            type: 'scatterpolar',
            theta,
            r: radius,
            fill: 'toself',
            fillcolor: 'rgba(0, 168, 150, 0.18)',
            line: { color: '#00a896', width: 2 },
            marker: { color: '#008080', size: 5 },
            hovertemplate: '%{theta}: %{r:.1f}%<extra></extra>'
          }]}
          layout={{
            autosize: true,
            margin: { l: 42, r: 42, t: 20, b: 30 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter, ui-sans-serif, system-ui', color: textColor },
            polar: {
              bgcolor: 'rgba(0,0,0,0)',
              radialaxis: { visible: true, range: [0, 100], ticksuffix: '%', gridcolor: gridColor, linecolor: gridColor, tickfont: { size: 9, color: textColor } },
              angularaxis: { gridcolor: gridColor, linecolor: gridColor, tickfont: { size: 11, color: textColor } }
            },
            showlegend: false
          }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </Suspense>
    </div>
  );
}
