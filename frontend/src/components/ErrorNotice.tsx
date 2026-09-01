import { AlertTriangle, ChevronDown } from 'lucide-react';
import { ApiError } from '../api';
import { copy } from '../copy';

export default function ErrorNotice({ error }: { error: unknown }) {
  const known = error instanceof ApiError;
  const message = known ? error.message : copy.requestFailed;
  const details = known ? `${error.code}${error.technicalDetails ? ` · ${error.technicalDetails}` : ''}` : error instanceof Error ? error.name : copy.unknownError;
  return (
    <div className="error-notice" role="alert">
      <AlertTriangle size={20} aria-hidden="true" />
      <div><strong>{message}</strong><details><summary>{copy.showDetails} <ChevronDown size={14} /></summary><code>{details}</code></details></div>
    </div>
  );
}
