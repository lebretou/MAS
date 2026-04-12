import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { GraphNodeData } from "../../../types/node-data";

type TerminalNodeType = Node<GraphNodeData, "terminal">;

export function TerminalNode({ data, sourcePosition, targetPosition }: NodeProps<TerminalNodeType>) {
  const isStart = data.nodeType === "start";
  const dimmed = data.playback?.frameState === "idle";
  const { ports } = data;

  const sourcePorts = ports?.filter((p) => p.type === "source") ?? [];
  const targetPorts = ports?.filter((p) => p.type === "target") ?? [];

  return (
    <div className={`terminal-node terminal-node--${data.nodeType}${dimmed ? " terminal-node--dimmed" : ""}`}>
      <span>{data.label}</span>
      {isStart ? (
        sourcePorts.length > 0 ? (
          sourcePorts.map((port) => (
            <Handle
              key={port.id}
              id={port.id}
              type="source"
              position={sourcePosition ?? Position.Right}
              className="node-handle"
              style={{ top: port.y }}
            />
          ))
        ) : (
          <Handle
            type="source"
            position={sourcePosition ?? Position.Right}
            className="node-handle"
          />
        )
      ) : (
        targetPorts.length > 0 ? (
          targetPorts.map((port) => (
            <Handle
              key={port.id}
              id={port.id}
              type="target"
              position={targetPosition ?? Position.Left}
              className="node-handle"
              style={{ top: port.y }}
            />
          ))
        ) : (
          <Handle
            type="target"
            position={targetPosition ?? Position.Left}
            className="node-handle"
          />
        )
      )}
    </div>
  );
}
