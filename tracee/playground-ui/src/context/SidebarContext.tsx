import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";
import type { GraphNodeData } from "../types/node-data";

interface SidebarContextType {
  selectedNode: GraphNodeData | null;
  openSidebar: (nodeData: GraphNodeData) => void;
  closeSidebar: () => void;
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined);

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(null);

  const openSidebar = (nodeData: GraphNodeData) => {
    setSelectedNode(nodeData);
  };

  const closeSidebar = () => {
    setSelectedNode(null);
  };

  return (
    <SidebarContext.Provider value={{ selectedNode, openSidebar, closeSidebar }}>
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
