import { render, screen } from '@testing-library/react';
import ProbabilityBars from '../components/ProbabilityBars';

test('hideZero removes values that render as 0.0 percent before applying the limit', () => {
  render(
    <ProbabilityBars
      hideZero
      limit={2}
      items={[
        { name: 'zero', probability: 0 },
        { name: 'rounded zero', probability: 0.0004 },
        { name: 'visible one', probability: 0.0006 },
        { name: 'visible two', probability: 0.2 }
      ]}
    />
  );

  expect(screen.queryByText('zero')).not.toBeInTheDocument();
  expect(screen.queryByText('rounded zero')).not.toBeInTheDocument();
  expect(screen.getByText('visible one')).toBeVisible();
  expect(screen.getByText('visible two')).toBeVisible();
});
