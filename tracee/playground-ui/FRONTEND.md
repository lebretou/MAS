# Tracee Playground UI -- Frontend Guide

Tracee is a developer tool for building and debugging multi-agent systems. The **playground-ui** is the web frontend that visualizes agent graphs and lets you test prompts against live models. This guide explains the project structure, key concepts, and how to make changes.

## Tech Stack

| Tool | What it does |
|------|--------------|
| **Vite** | Build tool and development server. Think of it as the thing that compiles your code and serves it during development. |
| **React 19** | UI library. The entire interface is built from React components -- functions that return HTML-like markup (called JSX). |
| **TypeScript 5.9** | JavaScript with type annotations. Catches bugs at compile time instead of at runtime. |
| **React Flow** (`@xyflow/react`) | Library for rendering interactive node-and-edge diagrams. We use it to draw the agent graph. |
| **dagre** (`@dagrejs/dagre`) | Graph layout algorithm. Given a set of nodes and edges, it computes x/y positions so the graph looks tidy. |
| **axios** | HTTP client for making API calls to the backend. |
| **react-router-dom** | Client-side routing. Lets the app show different pages (Graph Viewer, Playground) without full page reloads. |

## Prerequisites

- **Node.js** (v18 or later) and **npm** installed
- The FastAPI backend running on port 8000 (see the main Tracee README for setup)

## Running the Project

```bash
cd tracee/playground-ui

# Install dependencies (first time, or after package.json changes)
npm install

# Start the dev server
npm run dev
```

This starts Vite on `http://localhost:5173`. The Vite config (`vite.config.ts`) includes a proxy rule that forwards any request starting with `/api` to `http://localhost:8000`, so the frontend talks to the FastAPI backend seamlessly.

**The backend must be running first.** If it is not, API calls will fail and the UI will show errors.

### Other Commands

| Command | Purpose |
|---------|---------|
| `npm run dev` | Start the dev server with hot reload |
| `npm run build` | Production build (runs TypeScript type-check, then Vite bundling) |
| `npx tsc -b` | Type-check without producing build output -- useful for catching errors quickly |
| `npm run preview` | Serve the production build locally for testing |

## Project Structure

```
src/
├── main.tsx              # Entry point. Mounts the React app into the HTML page.
├── App.tsx               # Root component. Sets up providers (global state) and routes.
├── index.css             # All styles for the application (single CSS file).
├── vite-env.d.ts         # TypeScript declarations so you can import .svg and .css files.
│
├── api/                  # Functions that call the backend. One file per domain.
├── types/                # TypeScript interfaces that mirror the backend data models.
├── hooks/                # Custom React hooks that fetch and transform data.
├── context/              # React Contexts that hold global UI state.
├── components/           # Shared UI components used across pages.
├── features/             # Feature modules. Each subfolder is a self-contained section of the app.
└── assets/               # SVG icons used in the UI.
```

### `src/api/` -- Backend Communication

Each file exports functions that call one group of backend endpoints:

| File | Functions | Backend endpoints |
|------|-----------|-------------------|
| `client.ts` | (shared axios instance) | Base URL: `/api` |
| `graphs.ts` | `fetchGraphs()`, `fetchGraph(id)` | `GET /api/graphs`, `GET /api/graphs/{id}` |
| `agents.ts` | `fetchAgents()`, `fetchAgent(id)` | `GET /api/agents`, `GET /api/agents/{id}` |
| `prompts.ts` | `fetchPrompts()`, `fetchLatestVersion(id)`, `createPrompt()`, `createVersion()` | `/api/prompts/*` |
| `traces.ts` | `fetchTraces()`, `fetchTraceEvents(id)`, `fetchTraceSummary(id)` | `/api/traces/*` |
| `playground.ts` | `createRun()`, `fetchRuns()`, `fetchRun(id)` | `/api/playground/*` |

All of these use the shared axios client in `client.ts`, which automatically prepends `/api` to every request path.

### `src/types/` -- Data Models

TypeScript interfaces that describe the shape of data coming from the backend. When the backend changes a response format, update the matching type here. Key files:

- `graph.ts` -- `GraphTopology`, `GraphNode`, `GraphEdge`
- `agent.ts` -- `AgentRegistryEntry`
- `prompt.ts` -- `Prompt`, `PromptVersion`, `PromptComponent`, `PromptComponentType`
- `trace.ts` -- `TraceEvent`, `TraceMetadata`
- `playground.ts` -- `PlaygroundRun`, `PlaygroundRunCreate`, `PlaygroundRunResponse`
- `node-data.ts` -- `GraphNodeData` (the data attached to each ReactFlow node), `ExecutionData`
- `schema.ts` -- `JsonSchema` (recursive type representing a JSON Schema object)

### `src/hooks/` -- Data Fetching Hooks

React hooks are functions that let components "hook into" React features like state and side effects. The two custom hooks here handle all data fetching for the graph viewer:

- **`useGraph(graphId?)`** -- Fetches the graph topology from the API, then fans out to fetch prompt components for every agent node. Returns fully hydrated ReactFlow nodes and edges, plus the workflow state schema.
- **`useTraceOverlay(traceId, baseNodes)`** -- When a trace is selected, fetches its events and groups them by LangGraph node. Returns a new array of nodes where each agent node has an `execution` field containing status, latency, and LLM input/output.

### `src/context/` -- Global State

React Context is a way to share state across many components without passing it through every level of the component tree. This project has two contexts:

- **`LayerContext`** -- Tracks which layer is active (`"intent"` or `"execution"`) and which trace ID is selected. When you switch to the intent layer, the trace selection is automatically cleared.
- **`SidebarContext`** -- Tracks which agent node is currently selected for the detail side panel. Opening a node sets it; closing the panel clears it.

### `src/features/` -- Feature Modules

Each folder is a self-contained part of the application:

**`features/graph-viewer/`** -- The main graph visualization canvas.

| File / Folder | Purpose |
|----------------|---------|
| `GraphViewer.tsx` | Top-level component. Wires together hooks, contexts, and ReactFlow. |
| `layout.ts` | Runs the dagre algorithm to compute node positions (left-to-right). |
| `constants.ts` | Node dimensions and prompt component color palette. |
| `nodes/AgentNode.tsx` | Custom ReactFlow node. Delegates rendering to intent or execution variant. |
| `nodes/AgentNode.intent.tsx` | Intent mode: shows prompt chips, model name, link to prompt. |
| `nodes/AgentNode.execution.tsx` | Execution mode: shows status badge, latency, token count. |
| `nodes/TerminalNode.tsx` | Renders the START and END nodes. |
| `panels/AgentDetailPanel.tsx` | Slide-out side panel when you click an agent node. |
| `panels/IntentDetails.tsx` | Panel content for intent layer: config grid, prompt components, output schema. |
| `panels/ExecutionDetails.tsx` | Panel content for execution layer: status, latency, raw LLM I/O. |
| `panels/StateSchemaPanel.tsx` | Floating panel showing the workflow state fields. |
| `panels/SchemaTable.tsx` | Reusable table that renders a JSON Schema as rows. |
| `controls/LayerToggle.tsx` | Toggle button to switch between Intent and Execution layers. |
| `controls/GraphSelector.tsx` | Dropdown to pick which graph to view (shown when multiple graphs exist). |
| `controls/TraceSelector.tsx` | Dropdown to pick which trace run to overlay. |

**`features/playground/`** -- The prompt testing page.

| File | Purpose |
|------|---------|
| `PlaygroundPage.tsx` | Form to type a prompt, pick a model, execute a run, and see the output. |

## Key Concepts

### The Two-Layer Model

The graph viewer has two modes of looking at the same graph, controlled by the toggle in the top-right corner:

**Intent Layer** (default) -- The "design-time" view. Shows every node and edge in the graph as the developer defined it. Agent nodes display their prompt components, model configuration, and a link to the prompt editor. Data comes from `GET /api/graphs/{id}` (for the topology) and `GET /api/prompts/{id}/latest` (for each agent's prompt components).

**Execution Layer** -- The "runtime" view. Overlays data from a specific trace run onto the graph. Nodes that were not invoked during the run appear greyed out. Nodes that did run show a status badge (success or error), latency in milliseconds, and expandable LLM input/output. Data comes from `GET /api/traces/{id}`.

Switching to Intent mode automatically clears the trace selection. Switching to Execution mode lets you pick a trace from the dropdown.

### Data Flow

Here is how data travels from the backend into the UI:

```
Backend API
    |
    v
src/api/ functions (axios calls)
    |
    v
src/hooks/useGraph        --> fetches topology + prompt components
src/hooks/useTraceOverlay --> fetches trace events, computes per-node execution data
    |
    v
GraphViewer.tsx           --> passes hydrated nodes + edges to ReactFlow
    |
    v
AgentNode / TerminalNode  --> reads LayerContext to decide which sub-component to render
    |
    v
AgentDetailPanel          --> reads SidebarContext to know which node is selected
```

1. `useGraph` calls `fetchGraphs()` to get the list of available graphs, then `fetchGraph(id)` for the topology, then `fetchLatestVersion(promptId)` for each agent node. It runs the dagre layout and returns positioned nodes and edges.
2. `useTraceOverlay` receives the base nodes from `useGraph`. When a trace ID is selected, it calls `fetchTraceEvents(traceId)`, groups events by LangGraph node, and merges execution data onto each node.
3. `GraphViewer` passes the final nodes to the `<ReactFlow>` component, which renders them on a pannable, zoomable canvas.
4. Each `AgentNode` reads `LayerContext` to decide whether to render the intent variant or the execution variant.

### React Flow Basics

React Flow is a library that renders interactive node-and-edge diagrams. You give it an array of node objects and an array of edge objects, and it draws them on a canvas that supports panning, zooming, and selection.

**Custom nodes** -- By default, React Flow renders plain rectangles. This project registers two custom node types (`agent` and `terminal`) so we can render richer content inside each node. The mapping is defined in `GraphViewer.tsx`:

```typescript
const nodeTypes: NodeTypes = {
  agent: AgentNode,
  terminal: TerminalNode,
};
```

**Dagre layout** -- React Flow does not position nodes for you. The `layout.ts` file uses dagre to compute positions automatically. It creates a directed graph, sets each node's width and height, then calls `dagre.layout(g)` to get x/y coordinates. The result is a clean left-to-right (LR) layout.

### CSS Approach

All styles live in a single file: `src/index.css`. The project uses **BEM-like** class naming, which stands for Block-Element-Modifier. The convention looks like this:

```css
/* Block: the top-level component */
.agent-node { }

/* Element: a part inside the block (double underscore) */
.agent-node__header { }
.agent-node__chip { }

/* Modifier: a variation of the block or element (double dash) */
.agent-node__chip--role { }
.agent-node__chip--active { }
```

The benefit of BEM is that class names are descriptive and unlikely to collide. You can look at a class name like `.side-panel__meta-card` and immediately know it belongs to the side panel component.

There are no CSS modules, no Tailwind, and no CSS-in-JS. If you need to add styles, add them to `index.css` following the same naming pattern.

## How to Add a New Feature

### Adding a new API call

1. Open (or create) the appropriate file in `src/api/`.
2. Import the shared client: `import client from "./client";`
3. Write an async function that calls the endpoint and returns typed data:

```typescript
import client from "./client";
import type { MyType } from "../types/myType";

export async function fetchMyThing(id: string): Promise<MyType> {
  const { data } = await client.get<MyType>(`/my-endpoint/${id}`);
  return data;
}
```

### Adding a new TypeScript type

Create or edit a file in `src/types/`. Define an `interface` or `type` that matches the JSON the backend returns:

```typescript
export interface MyType {
  id: string;
  name: string;
  created_at: string;
}
```

### Adding a new page

1. Create a new component file in `src/features/your-feature/YourPage.tsx`.
2. Add a route in `src/App.tsx` inside the existing `<Route path="/" element={<AppShell />}>` block:

```tsx
<Route path="your-page" element={<YourPage />} />
```

3. Optionally add a navigation link in `src/components/AppShell.tsx`.

### Adding a new panel or control to the graph viewer

- **New control** (toolbar button, dropdown): add a component in `src/features/graph-viewer/controls/`, then render it inside the `<ReactFlow>` block in `GraphViewer.tsx`.
- **New panel** (side panel, floating panel): add a component in `src/features/graph-viewer/panels/`, then render it alongside the existing panels in `GraphViewer.tsx`.
- **New node type**: create a component in `src/features/graph-viewer/nodes/`, register it in the `nodeTypes` map in `GraphViewer.tsx`, and add its dimensions to `constants.ts`.

## Routing

The app has two routes, defined in `src/App.tsx`:

| Path | Component | Description |
|------|-----------|-------------|
| `/` | `GraphViewer` | The main graph visualization page |
| `/playground` | `PlaygroundPage` | The prompt testing page |

Both are wrapped in `AppShell`, which provides the top navigation bar. The router uses `react-router-dom` v7 with the `BrowserRouter` component.

## Environment and Configuration

The frontend itself has no environment variables. All configuration is in `vite.config.ts`:

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

If the backend runs on a different port, change the `target` value here.

## Troubleshooting

**"loading graph..." never finishes** -- The backend is not running or not reachable on port 8000. Start it with `uvicorn server.app:app --reload --port 8000` from the `tracee/` directory.

**Type errors after pulling new code** -- Run `npm install` to pick up any new dependencies, then `npx tsc -b` to see all type errors.

**Styles not updating** -- Vite hot-reloads CSS changes automatically. If styles seem stale, hard-refresh the browser (`Cmd+Shift+R` on macOS, `Ctrl+Shift+R` on Windows/Linux).

**Graph layout looks wrong after adding nodes** -- Check that the new node type has dimensions defined in `constants.ts`. Dagre needs width and height to position nodes correctly.
