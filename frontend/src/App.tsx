import { LoaderCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { getMeta } from './api';
import { copy } from './copy';
import AppShell from './components/AppShell';
import ErrorNotice from './components/ErrorNotice';
import AnalysisPage from './pages/AnalysisPage';
import CandidatesPage, { createCandidateWorkspace, EMPTY_CANDIDATE_WORKSPACE, type CandidateWorkspaceState } from './pages/CandidatesPage';
import DataIntakePage from './pages/DataIntakePage';
import type { AppMeta } from './types';

export default function App() {
  const [meta, setMeta] = useState<AppMeta | null>(null);
  const [candidateWorkspace, setCandidateWorkspace] = useState<CandidateWorkspaceState>(EMPTY_CANDIDATE_WORKSPACE);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    const controller = new AbortController();
    getMeta(controller.signal).then((nextMeta) => {
      setMeta(nextMeta);
      setCandidateWorkspace((current) => current.initialized ? current : createCandidateWorkspace(nextMeta));
    }).catch((requestError) => {
      if (!(requestError instanceof DOMException && requestError.name === 'AbortError')) setError(requestError);
    });
    return () => controller.abort();
  }, []);
  return (
    <AppShell>
      {error ? <ErrorNotice error={error} /> : !meta ? <div className="app-loading"><LoaderCircle className="spin" /><span>{copy.loadingResources}</span></div> : (
        <Routes>
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/candidates" element={<CandidatesPage meta={meta} workspace={candidateWorkspace} setWorkspace={setCandidateWorkspace} />} />
          <Route path="/data/intake" element={<DataIntakePage meta={meta} />} />
          <Route path="*" element={<Navigate to="/analysis" replace />} />
        </Routes>
      )}
    </AppShell>
  );
}
