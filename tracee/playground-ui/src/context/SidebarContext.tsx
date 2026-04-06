import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";
import type { GraphNodeData } from "../types/node-data";

export interface ChipExpansion {
  type: string;
  value: string;
}

interface SidebarContextType {
  selectedNodeId: string | null;
  selectedNode: GraphNodeData | null;
  chipExpansion: ChipExpansion | null;
  openSidebar: (nodeId: string, nodeData: GraphNodeData, expansion?: ChipExpansion) => void;
  syncSelectedNode: (nodeData: GraphNodeData) => void;
  closeSidebar: () => void;
  clearChipExpansion: () => void;
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined);

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(null);
  const [chipExpansion, setChipExpansion] = useState<ChipExpansion | null>(null);

  const openSidebar = useCallback((nodeId: string, nodeData: GraphNodeData, expansion?: ChipExpansion) => {
    setSelectedNodeId(nodeId);
    setSelectedNode(nodeData);
    setChipExpansion(expansion ?? null);
  }, []);

  const syncSelectedNode = useCallback((nodeData: GraphNodeData) => {
    setSelectedNode(nodeData);
  }, []);

  const closeSidebar = useCallback(() => {
    setSelectedNodeId(null);
    setSelectedNode(null);
    setChipExpansion(null);
  }, []);

  const clearChipExpansion = useCallback(() => {
    setChipExpansion(null);
  }, []);

  return (
    <SidebarContext.Provider value={{ selectedNodeId, selectedNode, chipExpansion, openSidebar, syncSelectedNode, closeSidebar, clearChipExpansion }}>
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
