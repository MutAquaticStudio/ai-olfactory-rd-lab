import { ChevronDown } from 'lucide-react';
import { copy } from '../copy';
import type { PredictedOdorProfile } from '../types';
import Panel from './Panel';
import ProbabilityBars from './ProbabilityBars';
import RadarChart from './RadarChart';

export default function OdorProfile({ profile }: { profile: PredictedOdorProfile }) {
  return (
    <Panel title={copy.predictedProfile} className="odor-profile-panel">
      <div className="odor-profile-grid">
        <div><h3>{copy.facets}</h3><RadarChart facets={profile.taxonomy.facets} /></div>
        <div><h3>{copy.textures}</h3><ProbabilityBars items={profile.taxonomy.textures} limit={8} /></div>
        <div><h3>{copy.sensations}</h3><ProbabilityBars items={profile.taxonomy.sensations} limit={8} showChemesthesis /></div>
      </div>
      <details className="disclosure profile-model-output">
        <summary>{copy.modelOutput}<ChevronDown size={17} /></summary>
        <div className="disclosure-body"><ProbabilityBars items={profile.model_output} limit={113} hideZero /></div>
      </details>
      <p className="panel-note">{copy.taxonomyAttribution} · {profile.taxonomy.taxonomy_version}. {copy.taxonomyDisclaimer}</p>
    </Panel>
  );
}
