import { Handle, Position, useReactFlow, type NodeProps, type Node } from "@xyflow/react";
import type { GraphNodeData } from "../../../types/node-data";
import cognitionIcon from "../../../assets/cognition.svg";
import { useSidebar } from "../../../context/SidebarContext";
import { useLayer } from "../../../context/LayerContext";
import { IntentContent } from "./AgentNode.intent";
import { ExecutionContent } from "./AgentNode.execution";

type AgentNodeType = Node<GraphNodeData, "agent">;

export function AgentNode({ id, data, sourcePosition, targetPosition }: NodeProps<AgentNodeType>) {
  const { openSidebar } = useSidebar();
  const { getNode, setCenter, getZoom } = useReactFlow();
  const { layer } = useLayer();
  const { label, metadata, execution, playback, ports } = data;

  const isTraceLayer = layer === "execution" || layer === "cognition";
  const dimmed = isTraceLayer && (playback ? (playback.frameState === "upcoming" || playback.frameState === "idle") : (!execution || !execution.invoked));
  const activeFrame = layer === "execution" && playback?.frameState === "active";

  const sourcePorts = ports?.filter((p) => p.type === "source") ?? [];
  const targetPorts = ports?.filter((p) => p.type === "target") ?? [];

  const handleNodeClick = () => {
    openSidebar(id, data);
    const node = getNode(id);
    if (!node) return;
    const targetZoom = Math.max(getZoom(), 1.2);
    const nodeWidth = node.measured?.width ?? node.width ?? 0;
    const nodeHeight = node.measured?.height ?? node.height ?? 0;
    const centerX = node.position.x + nodeWidth / 2;
    const centerY = node.position.y + nodeHeight / 2;

    // place the node at the horizontal midpoint of the visible canvas,
    // between the left controls panel and the right detail panel (.side-panel: 50vw, right 20px)
    const vw = window.innerWidth;
    const leftPanel = document.querySelector('.left-controls-panel');
    const leftEdge = leftPanel ? leftPanel.getBoundingClientRect().right : 0;
    const rightPanelLeftEdge = vw * 0.5 - 20;
    const visibleCenterX = (leftEdge + rightPanelLeftEdge) / 2;
    const xShift = (vw / 2 - visibleCenterX) / targetZoom;

    setCenter(centerX + xShift, centerY, {
      zoom: targetZoom,
      duration: 450,
    });
  };

  return (
    <div
      className={`agent-node${dimmed ? " agent-node--dimmed" : ""}${activeFrame ? " agent-node--active-frame" : ""}`}
      onClick={handleNodeClick}
    >
      <div className="agent-node__header">
        <div className="agent-node__icon-wrapper">
          <img src={cognitionIcon} alt="" className="agent-node__icon" />
        </div>
        <div className="agent-node__header-text">
          <span className="agent-node__label">{label}</span>
          {(metadata?.hasTools || metadata?.hasRetry) && (
            <div className="agent-node__badges">
              {metadata?.hasTools && <span className="agent-node__badge">tools</span>}
              {metadata?.hasRetry && (
                <span className="agent-node__badge agent-node__badge--retry">retry</span>
              )}
            </div>
          )}
        </div>
      </div>

      {layer === "intent" ? <IntentContent data={data} /> : <ExecutionContent data={data} />}

      {targetPorts.length > 0 ? (
        targetPorts.map((port) => (
          <Handle
            key={port.id}
            id={port.id}
            type="target"
            position={targetPosition ?? Position.Left}
            className="node-handle"
            style={{ top: port.y }}
            onClick={(e) => e.stopPropagation()}
          />
        ))
      ) : (
        <Handle
          type="target"
          position={targetPosition ?? Position.Left}
          className="node-handle"
          onClick={(e) => e.stopPropagation()}
        />
      )}
      {sourcePorts.length > 0 ? (
        sourcePorts.map((port) => (
          <Handle
            key={port.id}
            id={port.id}
            type="source"
            position={sourcePosition ?? Position.Right}
            className="node-handle"
            style={{ top: port.y }}
            onClick={(e) => e.stopPropagation()}
          />
        ))
      ) : (
        <Handle
          type="source"
          position={sourcePosition ?? Position.Right}
          className="node-handle"
          onClick={(e) => e.stopPropagation()}
        />
      )}
    </div>
  );
}
