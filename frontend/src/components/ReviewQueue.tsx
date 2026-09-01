import { ChevronDown } from 'lucide-react';
import { copy } from '../copy';
import type { ReviewCandidate } from '../types';
import ReferenceEvidencePanel from './ReferenceEvidencePanel';

export default function ReviewQueue({ items }: { items: ReviewCandidate[] }) {
  return (
    <details className="review-queue">
      <summary>{copy.reviewQueue}<span>{items.length}</span><ChevronDown /></summary>
      <div className="review-list">
        <p>{copy.reviewCaption}</p>
        {items.map((item, index) => (
          <div className="review-row" key={`${item.isomeric_smiles}-${index}`}>
            {item.structure_2d_svg && <div className="review-structure" dangerouslySetInnerHTML={{ __html: item.structure_2d_svg }} />}
            <div className="review-identity">
              <strong>{item.review_category === 'REFERENCE' ? copy.referenceReview : copy.chemistryReview}</strong>
              <code>{item.isomeric_smiles}</code>
              <span>{item.chemistry_screen.reasons.join(' · ')}</span>
            </div>
            {item.reference_checks.length > 0 && <ReferenceEvidencePanel checks={item.reference_checks} gate={item.reference_gate} compact />}
          </div>
        ))}
      </div>
    </details>
  );
}
