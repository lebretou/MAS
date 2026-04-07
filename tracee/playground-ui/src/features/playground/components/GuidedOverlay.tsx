import React from 'react';
import type { GuidedStartRole } from '../../../types/guidedStart';
import { COMPONENT_LABELS } from '../promptEditor';

export type GuidedOverlayStep = 2 | 3 | 4;

interface Props {
  step: GuidedOverlayStep;
  role: GuidedStartRole;
  getAnchor: () => HTMLElement | null;
  onNext: () => void;
  onOpenTools: () => void;
  onOpenSchema: () => void;
  onDone: () => void;
}

const TOTAL_STEPS = 4;

function formatPrevalence(prevalence: number): string {
  return `${Math.round(prevalence * 100)}%`;
}

const GuidedOverlay: React.FC<Props> = ({
  step,
  role,
  getAnchor,
  onNext,
  onOpenTools,
  onOpenSchema,
  onDone,
}) => {
  const overlayRef = React.useRef<HTMLDivElement>(null);
  const [position, setPosition] = React.useState<{ top: number; left: number } | null>(null);

  const updatePosition = React.useCallback(() => {
    const anchor = getAnchor();
    if (!anchor) {
      return;
    }
    const rect = anchor.getBoundingClientRect();
    const overlay = overlayRef.current;
    const overlayHeight = overlay?.offsetHeight ?? 200;

    let top = rect.bottom + 8;
    if (top + overlayHeight > window.innerHeight - 16) {
      top = rect.top - overlayHeight - 8;
    }
    top = Math.max(8, Math.min(top, window.innerHeight - overlayHeight - 16));

    setPosition({ top, left: rect.left });
  }, [getAnchor]);

  React.useLayoutEffect(() => {
    updatePosition();
  }, [updatePosition, step]);

  React.useEffect(() => {
    const frame = requestAnimationFrame(updatePosition);
    return () => cancelAnimationFrame(frame);
  }, [updatePosition, step]);

  React.useEffect(() => {
    window.addEventListener('scroll', updatePosition, true);
    window.addEventListener('resize', updatePosition);
    return () => {
      window.removeEventListener('scroll', updatePosition, true);
      window.removeEventListener('resize', updatePosition);
    };
  }, [updatePosition]);

  if (!position) {
    return null;
  }

  return (
    <div
      ref={overlayRef}
      className="guided-overlay"
      style={{ top: position.top, left: position.left }}
    >
      <div className="guided-overlay__header">
        <span className="guided-overlay__step-indicator">Step {step} of {TOTAL_STEPS}</span>
        <span className="badge badge--neutral">{role.name}</span>
      </div>

      {step === 2 && (
        <div className="guided-overlay__content">
          <div className="guided-overlay__text">
            <span className="guided-overlay__title">Edit your prompt</span>
            <span className="guided-overlay__desc">
              Components for <strong>{role.name}</strong> have been added to the editor. Fill in each section.
            </span>
          </div>
          <div className="guided-overlay__component-list">
            {role.components.map((comp) => (
              <div key={comp.component_type} className="guided-overlay__component-row">
                <span className="type-badge guided-overlay__type-badge">{COMPONENT_LABELS[comp.component_type] ?? comp.component_type}</span>
                <span className="guided-overlay__prevalence">{formatPrevalence(comp.prevalence)}</span>
              </div>
            ))}
          </div>
          <div className="guided-overlay__actions">
            <button type="button" className="btn btn--primary btn--sm" onClick={onNext}>
              Next
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="guided-overlay__content">
          <div className="guided-overlay__text">
            <span className="guided-overlay__title">Use variables</span>
            <span className="guided-overlay__desc">
              Add <code>{'{{variable_name}}'}</code> in your prompt to create dynamic inputs. They'll show up in the Variables panel.
            </span>
          </div>
          <div className="guided-overlay__actions">
            <button type="button" className="btn btn--primary btn--sm" onClick={onNext}>
              Next
            </button>
          </div>
        </div>
      )}

      {step === 4 && (
        <div className="guided-overlay__content">
          <div className="guided-overlay__text">
            <span className="guided-overlay__title">Tools or output schema?</span>
            <span className="guided-overlay__desc">
              Tools let the agent call functions. Output schema enforces structured responses. They can't be used together.
            </span>
          </div>
          <div className="guided-overlay__choice-row">
            <button type="button" className="btn btn--secondary btn--sm" onClick={onOpenTools}>
              Add Tools
            </button>
            <button type="button" className="btn btn--secondary btn--sm" onClick={onOpenSchema}>
              Output Schema
            </button>
          </div>
          <div className="guided-overlay__actions">
            <button type="button" className="btn btn--ghost btn--sm" onClick={onDone}>
              Skip
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default GuidedOverlay;
