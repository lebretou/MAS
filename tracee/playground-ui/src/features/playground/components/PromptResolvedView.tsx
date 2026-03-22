import React from 'react';

interface Props {
  resolvedPrompt: string;
}

const PromptResolvedView: React.FC<Props> = ({ resolvedPrompt }) => {
  return (
    <div className="card prompt-resolved">
      <div className="card__header prompt-resolved__header">
        <div>
          <div className="card__title diff__title">Resolved prompt</div>
          <span className="field__hint">
            read-only view of the full prompt text that will be executed.
          </span>
        </div>
        <span className="badge badge--neutral">{resolvedPrompt.length} chars</span>
      </div>
      <div className="card__body prompt-resolved__body">
        <textarea
          className="textarea textarea--code prompt-resolved__textarea"
          value={resolvedPrompt}
          readOnly
          aria-label="resolved prompt"
        />
      </div>
    </div>
  );
};

export default PromptResolvedView;
