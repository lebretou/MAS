import { createContext, useContext, useState, useCallback } from "react";
import type { ReactNode } from "react";

type Layer = "intent" | "execution";

interface LayerContextType {
  layer: Layer;
  setLayer: (layer: Layer) => void;
  selectedTraceId: string | null;
  setSelectedTraceId: (id: string | null) => void;
}

const LayerContext = createContext<LayerContextType | undefined>(undefined);

export function LayerProvider({ children }: { children: ReactNode }) {
  const [layer, setLayerState] = useState<Layer>("intent");
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);

  const setLayer = useCallback((next: Layer) => {
    setLayerState(next);
    if (next === "intent") {
      setSelectedTraceId(null);
    }
  }, []);

  return (
    <LayerContext.Provider value={{ layer, setLayer, selectedTraceId, setSelectedTraceId }}>
      {children}
    </LayerContext.Provider>
  );
}

export function useLayer() {
  const context = useContext(LayerContext);
  if (context === undefined) {
    throw new Error("useLayer must be used within a LayerProvider");
  }
  return context;
}
