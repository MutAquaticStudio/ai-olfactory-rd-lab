import { ArrowRight, LoaderCircle } from 'lucide-react';
import { FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import { analyzeMolecule } from '../api';
import { copy } from '../copy';
import type { AnalysisResult } from '../types';
import AnimatedContent from '../vendor/reactbits/AnimatedContent';
import ErrorNotice from '../components/ErrorNotice';
import Inspector from '../components/Inspector';
import Molecule2D from '../components/Molecule2D';
import Molecule3D from '../components/Molecule3D';
import OdorProfile from '../components/OdorProfile';
import StereoResolution from '../components/StereoResolution';

const DEFAULT_SMILES = 'CCCCC1C(CC(=O)C1)CC(=O)OC';

export default function AnalysisPage() {
  const [smiles, setSmiles] = useState(DEFAULT_SMILES);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const activeRequest = useRef<AbortController | null>(null);

  const analyze = useCallback(async (value: string) => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setLoading(true);
    setError(null);
    try {
      setResult(await analyzeMolecule(value, controller.signal));
    } catch (requestError) {
      if (!(requestError instanceof DOMException && requestError.name === 'AbortError')) setError(requestError);
    } finally {
      if (activeRequest.current === controller) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void analyze(DEFAULT_SMILES);
    return () => activeRequest.current?.abort();
  }, [analyze]);

  const submit = (event: FormEvent) => { event.preventDefault(); if (smiles.trim()) void analyze(smiles); };
  const selectStereo = (isomericSmiles: string) => {
    setSmiles(isomericSmiles);
    void analyze(isomericSmiles);
  };
  return (
    <AnimatedContent className="page page-analysis">
      <form className="analysis-command" onSubmit={submit}>
        <label htmlFor="smiles-input">{copy.smiles}</label>
        <div><input id="smiles-input" value={smiles} onChange={(event) => setSmiles(event.target.value)} spellCheck={false} autoComplete="off" /><button type="submit" className="primary-button" disabled={loading}>{loading ? <LoaderCircle className="spin" /> : null}{copy.analyze}<ArrowRight size={17} /></button></div>
      </form>
      {error !== null && <ErrorNotice error={error} />}
      {loading && !result && <div className="loading-workbench"><LoaderCircle className="spin" /><span>{copy.analyzingStructure}</span></div>}
      {result && (
        <div className={loading ? 'result-region is-updating' : 'result-region'} aria-busy={loading}>
          <div className="analysis-workbench">
            <Molecule2D svg={result.structure_2d_svg} />
            {result.analysis_state === 'COMPLETE'
              ? <Molecule3D result={result.conformer_ensemble} />
              : <StereoResolution result={result} onSelect={selectStereo} />}
            <Inspector result={result} />
          </div>
          {result.analysis_state === 'COMPLETE' ? <>
            <OdorProfile profile={result.predicted_odor_profile} />
            <p className="page-disclaimer">{copy.predictionNote}</p>
          </> : null}
        </div>
      )}
    </AnimatedContent>
  );
}
