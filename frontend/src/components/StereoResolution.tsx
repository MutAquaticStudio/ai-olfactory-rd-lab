import { LockKeyhole } from 'lucide-react';
import { copy } from '../copy';
import type { StereoInputRequiredAnalysis, StereoRequiredAnalysis } from '../types';
import Panel from './Panel';

type PendingStereo = StereoRequiredAnalysis | StereoInputRequiredAnalysis;

export default function StereoResolution({
  result,
  onSelect
}: {
  result: PendingStereo;
  onSelect: (isomericSmiles: string) => void;
}) {
  return (
    <Panel title={copy.stereoResolution} className="stereo-resolution-panel">
      <div className="stereo-resolution-intro">
        <LockKeyhole size={18} />
        <p>{result.analysis_state === 'STEREO_INPUT_REQUIRED' ? copy.stereoManual : copy.stereoExplanation}</p>
      </div>
      {result.analysis_state === 'STEREO_REQUIRED' ? (
        <div className="stereo-option-list" aria-label={copy.stereoResolution}>
          {result.stereo_options.map((option, index) => (
            <article className="stereo-option" key={option.isomeric_smiles}>
              <div className="stereo-option-svg" dangerouslySetInnerHTML={{ __html: option.structure_2d_svg }} />
              <div>
                <strong>{copy.stereoOption} {index + 1}</strong>
                <span>{option.cip_summary}</span>
                <code>{option.isomeric_smiles}</code>
              </div>
              <button type="button" onClick={() => onSelect(option.isomeric_smiles)}>
                {copy.selectStereoisomer}
              </button>
            </article>
          ))}
        </div>
      ) : null}
      <div className="stereo-locked-states">
        <span><LockKeyhole size={14} />{copy.predictionLocked}</span>
        <span><LockKeyhole size={14} />{copy.conformerLocked}</span>
      </div>
    </Panel>
  );
}
