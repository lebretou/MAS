import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  ConnectionLineType,
  type Edge,
  type Node,
  type NodeTypes,
  Panel,
  type ReactFlowInstance,
} from "@xyflow/react";
import { AgentNode } from "./nodes/AgentNode";
import { TerminalNode } from "./nodes/TerminalNode";
import { AgentDetailPanel } from "./panels/AgentDetailPanel";
import { ExecutionInspector } from "./panels/ExecutionInspector";
import { CognitionInspector } from "./panels/CognitionInspector";
import { LayerToggle } from "./controls/LayerToggle";
import { GraphInfoPanel } from "./controls/GraphInfoPanel";
import { GraphSelector } from "./controls/GraphSelector";
import { TraceSelector } from "./controls/TraceSelector";
import { GraphSetupGuide } from "./GraphSetupGuide";
import { useSidebar } from "../../context/SidebarContext";
import { useLayer } from "../../context/LayerContext";
import { NO_GRAPHS_REGISTERED_ERROR, useGraph } from "../../hooks/useGraph";
import { useTracePlayback } from "../../hooks/useTracePlayback";
import { useCognitionOverlay } from "../../hooks/useCognitionOverlay";
import cognitionIcon from "../../assets/cognition.svg";
import type { GraphEdgeData, GraphNodeData, TraceOutlineItem } from "../../types/node-data";

const nodeTypes: NodeTypes = {
  agent: AgentNode,
  terminal: TerminalNode,
};

export function GraphViewer() {
  const { selectedNode, selectedNodeId, syncSelectedNode, closeSidebar, openSidebar } = useSidebar();
  const { layer, selectedTraceId, setSelectedTraceId } = useLayer();
  const [selectedGraphId, setSelectedGraphId] = useState<string | null>(null);
  const [activeFrameIndex, setActiveFrameIndex] = useState<number | null>(null);

  const { nodes: baseNodes, edges: baseEdges, stateSchema, graphId, graphInfo, graphIds, loading, error, refetch } =
    useGraph(selectedGraphId);

  const {
    nodes: displayNodes,
    edges: playbackEdges,
    frames,
    outline,
    activeFrame,
    loading: playbackLoading,
    error: playbackError,
  } = useTracePlayback(selectedTraceId, baseNodes, baseEdges, activeFrameIndex);

  const isCognitionLayer = layer === "cognition";
  const {
    nodes: cognitionNodes,
    edges: cognitionEdges,
    cognition,
    loading: cognitionLoading,
    analyzing: cognitionAnalyzing,
    analyze: runAnalysis,
  } = useCognitionOverlay(selectedTraceId, displayNodes, playbackEdges, isCognitionLayer);

  const mergedNodes = isCognitionLayer ? cognitionNodes : displayNodes;
  const mergedEdges = isCognitionLayer ? cognitionEdges : playbackEdges;

  const flowNodes = useMemo(() => {
    if (activeFrameIndex == null) return mergedNodes;
    return mergedNodes.map((n) => ({ ...n, selected: false }));
  }, [mergedNodes, activeFrameIndex]);

  const reactFlowRef = useRef<ReactFlowInstance<Node<GraphNodeData>, Edge<GraphEdgeData>> | null>(null);

  const handlePanelDismiss = useCallback(() => {
    closeSidebar();
    reactFlowRef.current?.fitView({ duration: 450, padding: 0.16 });
  }, [closeSidebar]);

  const handleGraphSelect = useCallback((id: string) => {
    closeSidebar();
    setSelectedTraceId(null);
    setSelectedGraphId(id);
  }, [closeSidebar, setSelectedTraceId]);

  const focusNodeInCanvas = useCallback((nodeId: string) => {
    const flow = reactFlowRef.current;
    if (!flow) return;
    const node = flow.getNode(nodeId);
    if (!node) return;

    const targetZoom = Math.max(flow.getZoom(), 1.2);
    const nodeWidth = node.measured?.width ?? node.width ?? 0;
    const nodeHeight = node.measured?.height ?? node.height ?? 0;
    const centerX = node.position.x + nodeWidth / 2;
    const centerY = node.position.y + nodeHeight / 2;
    const vw = window.innerWidth;
    const leftPanel = document.querySelector(".left-controls-panel");
    const leftEdge = leftPanel ? leftPanel.getBoundingClientRect().right : 0;
    const rightPanelLeftEdge = vw * 0.5 - 20;
    const visibleCenterX = (leftEdge + rightPanelLeftEdge) / 2;
    const xShift = (vw / 2 - visibleCenterX) / targetZoom;

    flow.setCenter(centerX + xShift, centerY, {
      zoom: targetZoom,
      duration: 450,
    });
  }, []);

  const handleTraceOutlineSelect = useCallback((item: TraceOutlineItem) => {
    if (!item.nodeId) return;
    const node = displayNodes.find((candidate) => candidate.id === item.nodeId);
    if (!node) return;
    focusNodeInCanvas(item.nodeId);
    openSidebar(item.nodeId, node.data, undefined, item.operationId ?? null);
  }, [displayNodes, focusNodeInCanvas, openSidebar]);

  useEffect(() => {
    setActiveFrameIndex(null);
  }, [selectedTraceId, layer]);

  useEffect(() => {
    if (!selectedTraceId) {
      setActiveFrameIndex(null);
      return;
    }
    setActiveFrameIndex((current) => {
      if (current == null) return null;
      if (frames.length === 0) return null;
      return Math.min(current, frames.length - 1);
    });
  }, [selectedTraceId, frames]);

  useEffect(() => {
    if (!selectedNodeId) return;
    const liveNode = mergedNodes.find((node) => node.id === selectedNodeId);
    if (liveNode) {
      syncSelectedNode(liveNode.data);
      return;
    }
    closeSidebar();
  }, [mergedNodes, selectedNodeId, syncSelectedNode, closeSidebar]);

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
        edges={mergedEdges}
        nodeTypes={nodeTypes}
        connectionLineType={ConnectionLineType.Bezier}
        defaultEdgeOptions={{ type: "default" }}
        onInit={(instance) => {
          reactFlowRef.current = instance;
        }}
        proOptions={{ hideAttribution: true }}
        fitView
      >
        <Background gap={24} size={1} />
        <Panel position="top-left" className="left-controls-panel">
          <div className="layer-toggle-row">
            <LayerToggle />
            {isCognitionLayer && selectedTraceId && (
              <button
                type="button"
                className="cognition-run-btn"
                onClick={runAnalysis}
                disabled={cognitionAnalyzing}
                aria-label="Run cognition analysis"
              >
                <img src={cognitionIcon} alt="" className="cognition-run-btn__icon" />
                {cognitionAnalyzing ? "analyzing..." : "analyze"}
              </button>
            )}
          </div>
          <GraphInfoPanel graph={graphInfo} />
          <TraceSelector
            nodes={baseNodes}
            edges={baseEdges}
            graphId={graphId}
            outline={outline}
            outlineLoading={playbackLoading}
            onOutlineSelect={handleTraceOutlineSelect}
          />
        </Panel>
        {graphIds.length > 1 && (
          <GraphSelector selectedGraphId={graphId} onSelect={handleGraphSelect} />
        )}
        {playbackError && (
          <Panel position="bottom-right" className="trace-playback-error">
            failed to load trace playback
          </Panel>
        )}
        {!selectedNode && !isCognitionLayer && stateSchema && selectedTraceId && frames.length > 0 && (
          <ExecutionInspector
            frames={frames}
            activeFrameIndex={activeFrameIndex}
            onFrameChange={setActiveFrameIndex}
            schema={stateSchema}
            activeFrame={activeFrame}
          />
        )}
        {!selectedNode && isCognitionLayer && selectedTraceId && (
          <CognitionInspector
            cognition={cognition}
            loading={cognitionLoading}
            analyzing={cognitionAnalyzing}
            onAnalyze={runAnalysis}
          />
        )}
      </ReactFlow>
      {selectedNode && <AgentDetailPanel onRequestClose={handlePanelDismiss} />}
    </div>
  );
}
