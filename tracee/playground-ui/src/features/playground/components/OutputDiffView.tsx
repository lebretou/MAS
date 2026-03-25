import React from 'react';
import type { DiffLine } from '../../../utils/jsonDiff';

interface Props {
  diff: DiffLine[];
  referenceLabel: string;
  selectedRunLabel: string;
  onClose: () => void;
}

const PREFIX: Record<DiffLine['type'], string> = {
  same: '  ',
  added: '+ ',
  removed: '- ',
};

const OutputDiffView: React.FC<Props> = ({
  diff,
  referenceLabel,
  selectedRunLabel,
  onClose,
}) => {
  const hasChanges = diff.some(l => l.type !== 'same');
  const referenceName = 'anchor';

  return (
    <div className="card diff__card">
      <div className="card__header diff__header">
        <div>
          <h3 className="card__title diff__title">
            Output Diff
          </h3>
          <span className="field__hint">
            {selectedRunLabel} vs {referenceLabel} ({referenceName})
          </span>
        </div>
        <button
          className="icon-btn icon-btn--close diff__close-btn"
          onClick={onClose}
        >
          &times;
        </button>
      </div>
      <div className="card__body diff__body">
        {!hasChanges ? (
          <div className="diff__identical">
            Output is identical to the {referenceName}.
          </div>
        ) : (
          <div className="diff__legend">
            <span><span className="diff__legend-removed">-</span> {referenceLabel}</span>
            <span><span className="diff__legend-added">+</span> this run</span>
          </div>
        )}
        <pre className="diff__pre">
          {diff.map((line, i) => (
            <div
              key={i}
              className={`diff__line diff__line--${line.type}`}
            >
              <span className="diff__prefix">{PREFIX[line.type]}</span>
              {line.text}
            </div>
          ))}
        </pre>
      </div>
    </div>
  );
};

export default OutputDiffView;
