import type { GraphNodeData } from "../../../types/node-data";
import { componentColors } from "../constants";

interface Props {
  data: GraphNodeData;
}

export function IntentContent({ data }: Props) {
  const { metadata, promptId, components } = data;
  const enabledComponents = components?.filter((c) => c.enabled) ?? [];
  const disabledComponents = components?.filter((c) => !c.enabled) ?? [];

  return (
    <>
      <div className="agent-node__body">
        {metadata?.model && (
          <div className="agent-node__row">
            <span className="agent-node__key">model</span>
            <span className="agent-node__value">{metadata.model}</span>
          </div>
        )}
        {promptId && (
          <div className="agent-node__row">
            <span className="agent-node__key">prompt</span>
            <span className="agent-node__value prompt-link">{promptId}</span>
          </div>
        )}
      </div>

      {components && components.length > 0 && (
        <div className="agent-node__components">
          <div className="agent-node__components-header">
            <span className="agent-node__components-label">PROMPT COMPONENTS</span>
          </div>
          <div className="agent-node__component-chips">
            {enabledComponents.map((c) => (
              <span
                key={c.type}
                className="agent-node__chip"
                style={{ borderColor: componentColors[c.type] }}
                title={c.content}
              >
                <span className="agent-node__chip-dot" style={{ background: componentColors[c.type] }} />
                {c.type}
              </span>
            ))}
            {disabledComponents.map((c) => (
              <span
                key={c.type}
                className="agent-node__chip agent-node__chip--disabled"
                title={`[disabled] ${c.content}`}
              >
                <span className="agent-node__chip-dot" />
                {c.type}
              </span>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
