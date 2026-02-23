import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { GraphNodeData } from "../../../types/node-data";

type TerminalNodeType = Node<GraphNodeData, "terminal">;

export function TerminalNode({ data, sourcePosition, targetPosition }: NodeProps<TerminalNodeType>) {
  const isStart = data.nodeType === "start";

  return (
    <div className={`terminal-node terminal-node--${data.nodeType}`}>
      <span>{data.label}</span>
      {isStart ? (
        <Handle
          type="source"
          position={sourcePosition ?? Position.Right}
          className="node-handle"
        />
      ) : (
        <Handle
          type="target"
          position={targetPosition ?? Position.Left}
          className="node-handle"
        />
      )}
    </div>
  );
}
