import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SidebarProvider } from "./context/SidebarContext";
import { LayerProvider } from "./context/LayerContext";
import { AppShell } from "./features/app-shell/AppShell";
import { GraphViewer } from "./features/graph-viewer/GraphViewer";
import { PlaygroundPage } from "./features/playground/PlaygroundPage";
import { PromptsPage } from "./features/playground/PromptsPage";

export default function App() {
  return (
    <LayerProvider>
      <SidebarProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<AppShell />}>
              <Route index element={<GraphViewer />} />
              <Route path="playground" element={<PlaygroundPage />} />
              <Route path="prompts" element={<PromptsPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </SidebarProvider>
    </LayerProvider>
  );
}
