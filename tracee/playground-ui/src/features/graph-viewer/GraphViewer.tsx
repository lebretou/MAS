import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { FrameScrubber } from "./controls/FrameScrubber";
import { TraceSelector } from "./controls/TraceSelector";
import { GraphSetupGuide } from "./GraphSetupGuide";
import { useSidebar } from "../../context/SidebarContext";
import { useLayer } from "../../context/LayerContext";
import { NO_GRAPHS_REGISTERED_ERROR, useGraph } from "../../hooks/useGraph";
import { useTracePlayback } from "../../hooks/useTracePlayback";

const nodeTypes: NodeTypes = {
  agent: AgentNode,
  terminal: TerminalNode,
};

export function GraphViewer() {
  const { selectedNode, selectedNodeId, syncSelectedNode, closeSidebar } = useSidebar();
  const { selectedTraceId } = useLayer();
  const [selectedGraphId, setSelectedGraphId] = useState<string | null>(null);
  const [activeFrameIndex, setActiveFrameIndex] = useState<number | null>(null);

  const { nodes: baseNodes, edges: baseEdges, stateSchema, graphId, graphInfo, graphIds, loading, error, refetch } =
    useGraph(selectedGraphId);

  const { nodes: displayNodes, frames, activeFrame, error: playbackError } =
    useTracePlayback(selectedTraceId, baseNodes, activeFrameIndex);

  // when scrubber is on a specific frame, clear node selection so only the active-frame highlight shows
  const flowNodes = useMemo(() => {
    if (activeFrameIndex == null) return displayNodes;
    return displayNodes.map((n) => ({ ...n, selected: false }));
  }, [displayNodes, activeFrameIndex]);

  const [, , onNodesChange] = useNodesState(flowNodes);
  const [, , onEdgesChange] = useEdgesState(baseEdges);

  const reactFlowRef = useRef<{
    fitView: (options?: { duration?: number; padding?: number }) => void;
  } | null>(null);

  const handlePanelDismiss = useCallback(() => {
    closeSidebar();
    reactFlowRef.current?.fitView({ duration: 450, padding: 0.16 });
  }, [closeSidebar]);

  const handleGraphSelect = useCallback((id: string) => {
    closeSidebar();
    setSelectedGraphId(id);
  }, [closeSidebar]);

  useEffect(() => {
    setActiveFrameIndex(null);
  }, [selectedTraceId]);

  useEffect(() => {
    if (!selectedTraceId) {
      setActiveFrameIndex(null);
      return;
    }
    setActiveFrameIndex((current) => {
      if (current == null) return null;
      return Math.min(current, frames.length - 1);
    });
  }, [selectedTraceId, frames]);

  useEffect(() => {
    if (!selectedNodeId) return;
    const liveNode = displayNodes.find((node) => node.id === selectedNodeId);
    if (liveNode) {
      syncSelectedNode(liveNode.data);
      return;
    }
    closeSidebar();
  }, [displayNodes, selectedNodeId, syncSelectedNode, closeSidebar]);

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
        <p style={{ color: "#6b7280" }}>loading graph...</p>
      </div>
    );
  }

  if (error) {
    if (error === NO_GRAPHS_REGISTERED_ERROR) {
      return <GraphSetupGuide onRefresh={() => void refetch()} />;
    }

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
        nodes={flowNodes}
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
          <TraceSelector nodes={baseNodes} edges={baseEdges} />
        </Panel>
        {graphIds.length > 1 && (
          <GraphSelector selectedGraphId={graphId} onSelect={handleGraphSelect} />
        )}
        {playbackError && (
          <Panel position="bottom-right" className="trace-playback-error">
            failed to load trace playback
          </Panel>
        )}
        {!selectedNode && stateSchema && <StateSchemaPanel schema={stateSchema} activeFrame={activeFrame} />}
        {!selectedNode && selectedTraceId && frames.length > 0 && (
          <Panel position="bottom-center">
            <FrameScrubber
              frames={frames}
              activeFrameIndex={activeFrameIndex}
              onChange={setActiveFrameIndex}
            />
          </Panel>
        )}
      </ReactFlow>
      {selectedNode && <AgentDetailPanel onRequestClose={handlePanelDismiss} />}
    </div>
  );
}
