import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SidebarProvider } from "./context/SidebarContext";
import { LayerProvider } from "./context/LayerContext";
import { AppShell } from "./components/AppShell";
import { GraphViewer } from "./features/graph-viewer/GraphViewer";
import { PlaygroundPage } from "./features/playground/PlaygroundPage";

export default function App() {
  return (
    <LayerProvider>
      <SidebarProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<AppShell />}>
              <Route index element={<GraphViewer />} />
              <Route path="playground" element={<PlaygroundPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </SidebarProvider>
    </LayerProvider>
  );
}
