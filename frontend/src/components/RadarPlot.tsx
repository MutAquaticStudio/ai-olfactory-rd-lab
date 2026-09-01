import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js/lib/core';
import scatterpolar from 'plotly.js/lib/scatterpolar';

// Register only the trace required by the 11-facet radar instead of shipping
// the complete Plotly distribution to the browser.
Plotly.register([scatterpolar]);

export default createPlotlyComponent(Plotly);
