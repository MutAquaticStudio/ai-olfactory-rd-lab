import { makeRadarSeries } from '../components/RadarChart';

test('radar series keeps exactly 11 taxonomy axes and closes the polygon', () => {
  const facets = Array.from({ length: 11 }, (_, index) => ({
    name: `Facet ${index + 1}`,
    probability: (index + 1) / 20
  }));
  const series = makeRadarSeries(facets);
  expect(series.labels).toHaveLength(11);
  expect(new Set(series.labels)).toHaveLength(11);
  expect(series.theta).toHaveLength(12);
  expect(series.theta.at(-1)).toBe(series.theta[0]);
  expect(series.radius.at(-1)).toBe(series.radius[0]);
});
