import React, { useState, useCallback } from 'react';
import iconCopy from '../../../assets/icon-copy.svg';

interface Props {
  resolvedPrompt: string;
}

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

const PromptResolvedView: React.FC<Props> = ({ resolvedPrompt }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(resolvedPrompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [resolvedPrompt]);

  return (
    <div className="card prompt-resolved">
      <div className="card__header prompt-resolved__header">
        <div>
          <div className="card__title diff__title">Resolved prompt</div>
          <span className="field__hint">
            read-only view of the full prompt text that will be executed.
          </span>
        </div>
        <div className="prompt-resolved__actions">
          <button
            type="button"
            className="icon-btn prompt-resolved__copy-btn"
            onClick={handleCopy}
            aria-label="copy resolved prompt"
            title="copy resolved prompt"
          >
            <span
              className="prompt-components__action-icon"
              style={getMaskIconStyle(iconCopy)}
              aria-hidden
            />
            <span className={`prompt-resolved__copied-label ${copied ? 'is-visible' : ''}`}>
              copied
            </span>
          </button>
          <span className="badge badge--neutral">{resolvedPrompt.length} chars</span>
        </div>
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
