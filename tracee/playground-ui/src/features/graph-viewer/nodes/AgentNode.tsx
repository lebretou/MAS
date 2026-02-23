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
  const { label, metadata, execution } = data;

  const dimmed = layer === "execution" && (!execution || !execution.invoked);

  const handleNodeClick = () => {
    openSidebar(data);
    const node = getNode(id);
    if (!node) return;
    const targetZoom = Math.max(getZoom(), 1.2);
    const quarterViewportShift = (window.innerWidth * 0.34) / targetZoom;
    const centerX = node.position.x + (node.width ?? 0) / 2;
    const centerY = node.position.y + (node.height ?? 0) / 2;
    const yShift = 80 / targetZoom;

    setCenter(centerX + quarterViewportShift, centerY + yShift, {
      zoom: targetZoom,
      duration: 450,
    });
  };

  return (
    <div className={`agent-node${dimmed ? " agent-node--dimmed" : ""}`} onClick={handleNodeClick}>
      <div className="agent-node__header">
        <div className="agent-node__icon-wrapper">
          <img src={cognitionIcon} alt="" className="agent-node__icon" />
        </div>
        <div className="agent-node__header-text">
          <span className="agent-node__label">{label}</span>
          {metadata?.hasTools && <span className="agent-node__badge">tools</span>}
        </div>
      </div>

      {layer === "intent" ? <IntentContent data={data} /> : <ExecutionContent data={data} />}

      <Handle
        type="target"
        position={targetPosition ?? Position.Left}
        className="node-handle"
        onClick={(e) => e.stopPropagation()}
      />
      <Handle
        type="source"
        position={sourcePosition ?? Position.Right}
        className="node-handle"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
