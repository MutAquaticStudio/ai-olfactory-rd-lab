import { BookOpen, ExternalLink, FileSearch, ShieldAlert } from 'lucide-react';
import { copy } from '../copy';
import type { AcademicEvidenceMatch, AcademicEvidenceSummary } from '../types';

function statusLabel(status: AcademicEvidenceSummary['status']) {
  if (status === 'EXACT_MATCH') return copy.academicExactMatch;
  if (status === 'REVIEW_REQUIRED') return copy.academicReviewRequired;
  if (status === 'MENTION_ONLY') return copy.academicMentionOnly;
  return copy.academicNoExact;
}

function StatusIcon({ status }: { status: AcademicEvidenceSummary['status'] }) {
  if (status === 'EXACT_MATCH') return <BookOpen aria-hidden="true" />;
  if (status === 'NO_EXACT_EVIDENCE') return <FileSearch aria-hidden="true" />;
  return <ShieldAlert aria-hidden="true" />;
}

function sourceLabel(item: AcademicEvidenceMatch) {
  return item.document.content_type === 'abstract' ? copy.academicAbstract : copy.academicFullText;
}

export default function AcademicEvidencePanel({ summary }: { summary: AcademicEvidenceSummary | null }) {
  if (!summary) {
    return (
      <section className="academic-evidence academic-evidence-not-run">
        <header><h3>{copy.academicEvidence}</h3></header>
        <p>{copy.academicEvidenceNotRun}</p>
      </section>
    );
  }

  return (
    <section className={`academic-evidence academic-status-${summary.status.toLowerCase()}`}>
      <header>
        <div className="academic-heading">
          <StatusIcon status={summary.status} />
          <div><h3>{copy.academicEvidence}</h3><span>{statusLabel(summary.status)}</span></div>
        </div>
        <span className="academic-badge">{copy.academicLocalIndex}</span>
      </header>
      {summary.status === 'NO_EXACT_EVIDENCE' && <p className="academic-empty">{copy.academicNoMatches}</p>}
      {summary.conflicts.length > 0 && (
        <p className="academic-conflicts">{summary.conflicts.join(' · ')}</p>
      )}
      {summary.matches.length > 0 && (
        <div className="academic-match-list">
          {summary.matches.map((item) => (
            <article className="academic-match" key={item.evidence_id}>
              <div className="academic-match-title">
                <strong>{item.document.title || copy.academicUntitled}</strong>
                {item.document.link && <a href={item.document.link} target="_blank" rel="noreferrer" aria-label={`${copy.academicOpenSource}: ${item.document.title || copy.academicUntitled}`}><ExternalLink aria-hidden="true" /></a>}
              </div>
              <div className="academic-match-meta">
                <span>{sourceLabel(item)}</span>
                {item.source_type && <span>{item.source_type.replaceAll('_', ' ')}</span>}
                {item.document.license_status && <span>{item.document.license_status.replaceAll('_', ' ')}</span>}
                {item.match_level && <span>{item.match_level.replaceAll('_', ' ')}</span>}
                <span>{item.review_state === 'ACCEPTED' ? copy.academicAccepted : copy.academicUnreviewed}</span>
              </div>
              {item.document.doi && <small>{copy.academicDoi(item.document.doi)}</small>}
              {item.mention.page !== null && <small>{copy.academicPage(item.mention.page)}</small>}
              {item.mention.chunk_index !== null && <small>{copy.academicChunk(item.mention.chunk_index)}</small>}
              {item.mention.evidence_excerpt && <p>{item.mention.evidence_excerpt}</p>}
              {item.odor_descriptors.length > 0
                ? <small>{copy.academicDescriptors(item.odor_descriptors)}</small>
                : <small>{copy.academicUnassessed}</small>}
            </article>
          ))}
        </div>
      )}
      <p className="panel-note">{copy.academicBoundary}</p>
    </section>
  );
}
