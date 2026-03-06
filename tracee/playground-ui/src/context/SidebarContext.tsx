import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";
import type { GraphNodeData } from "../types/node-data";

interface SidebarContextType {
  selectedNodeId: string | null;
  selectedNode: GraphNodeData | null;
  openSidebar: (nodeId: string, nodeData: GraphNodeData) => void;
  syncSelectedNode: (nodeData: GraphNodeData) => void;
  closeSidebar: () => void;
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined);

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(null);

  const openSidebar = useCallback((nodeId: string, nodeData: GraphNodeData) => {
    setSelectedNodeId(nodeId);
    setSelectedNode(nodeData);
  }, []);

  const syncSelectedNode = useCallback((nodeData: GraphNodeData) => {
    setSelectedNode(nodeData);
  }, []);

  const closeSidebar = useCallback(() => {
    setSelectedNodeId(null);
    setSelectedNode(null);
  }, []);

  return (
    <SidebarContext.Provider value={{ selectedNodeId, selectedNode, openSidebar, syncSelectedNode, closeSidebar }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function useSidebar() {
  const context = useContext(SidebarContext);
  if (context === undefined) {
    throw new Error("useSidebar must be used within a SidebarProvider");
  }
  return context;
}
