import { useCallback, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  ConnectionLineType,
  useNodesState,
  useEdgesState,
  type NodeTypes,
  Panel,
} from "@xyflow/react";
import { AgentNode } from "./nodes/AgentNode";
import { TerminalNode } from "./nodes/TerminalNode";
import { AgentDetailPanel } from "./panels/AgentDetailPanel";
import { StateSchemaPanel } from "./panels/StateSchemaPanel";
import { LayerToggle } from "./controls/LayerToggle";
import { GraphInfoPanel } from "./controls/GraphInfoPanel";
import { GraphSelector } from "./controls/GraphSelector";
import { TraceSelector } from "./controls/TraceSelector";
import { useSidebar } from "../../context/SidebarContext";
import { useLayer } from "../../context/LayerContext";
import { useGraph } from "../../hooks/useGraph";
import { useTraceOverlay } from "../../hooks/useTraceOverlay";

const nodeTypes: NodeTypes = {
  agent: AgentNode,
  terminal: TerminalNode,
};

export function GraphViewer() {
  const { selectedNode, closeSidebar } = useSidebar();
  const { selectedTraceId } = useLayer();
  const [selectedGraphId, setSelectedGraphId] = useState<string | null>(null);

  const { nodes: baseNodes, edges: baseEdges, stateSchema, graphId, graphInfo, graphIds, loading, error } =
    useGraph(selectedGraphId);

  const displayNodes = useTraceOverlay(selectedTraceId, baseNodes);

  const [, , onNodesChange] = useNodesState(displayNodes);
  const [, , onEdgesChange] = useEdgesState(baseEdges);

  const reactFlowRef = useRef<{
    fitView: (options?: { duration?: number; padding?: number }) => void;
  } | null>(null);

  const handlePanelDismiss = useCallback(() => {
    closeSidebar();
    reactFlowRef.current?.fitView({ duration: 450, padding: 0.16 });
  }, [closeSidebar]);

  const handleGraphSelect = useCallback((id: string) => {
    setSelectedGraphId(id);
  }, []);

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
        <p style={{ color: "#6b7280" }}>loading graph...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
        <p style={{ color: "#ef4444" }}>{error}</p>
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <ReactFlow
        key={graphId ?? "no-graph"}
        nodes={displayNodes}
        edges={baseEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        connectionLineType={ConnectionLineType.SmoothStep}
        defaultEdgeOptions={{ type: "default" }}
        onInit={(instance) => {
          reactFlowRef.current = instance;
        }}
        proOptions={{ hideAttribution: true }}
        fitView
      >
        <Background gap={24} size={1} />
        <Panel position="top-left" className="left-controls-panel">
          <LayerToggle />
          <GraphInfoPanel graph={graphInfo} />
          <TraceSelector />
        </Panel>
        {graphIds.length > 1 && (
          <GraphSelector selectedGraphId={graphId} onSelect={handleGraphSelect} />
        )}
        {!selectedNode && stateSchema && <StateSchemaPanel schema={stateSchema} />}
      </ReactFlow>
      {selectedNode && <AgentDetailPanel onRequestClose={handlePanelDismiss} />}
    </div>
  );
}
