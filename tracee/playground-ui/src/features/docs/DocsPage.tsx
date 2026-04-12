import { useState, useEffect, useRef } from "react";
import "./docs.css";
import intentLayerImg from "./img/intent-layer.png";
import executionLayerImg from "./img/execution-layer.png";
import cognitionLayerImg from "./img/cognition-layer.png";

const sections = [
  { id: "overview", label: "Overview" },
  { id: "concepts", label: "Core Concepts" },
  { id: "setup", label: "Setup Guide" },
  { id: "graph", label: "Graph" },
  { id: "playground", label: "Playground" },
  { id: "prompts", label: "Prompts" },
  { id: "workflows", label: "Workflows" },
  { id: "troubleshooting", label: "Troubleshooting" },
];

const installSnippet = `pip install tracee`;

const installServerSnippet = `pip install 'tracee[server]'`;

const startServerSnippet = `tracee serve`;

const startServerOptionsSnippet = `tracee serve --port 8000 --host 0.0.0.0`;

const envSnippet = `# .env (in the directory where you run tracee serve)
OPENAI_API_KEY=sk-...
CORS_ORIGINS=http://localhost:5173,http://localhost:3000`;

const integrationSnippet = `import tracee

# compile your LangGraph workflow as usual
app = workflow.compile()

# register the graph topology with the Tracee server
tracee.init(
    app,
    graph_id="my-workflow",
    name="My Workflow",
    description="Orchestrates planning and execution agents",
    server_url="http://localhost:8000",
)

# wrap any invocation in tracee.trace() to record it
with tracee.trace():
    result = app.invoke(initial_state)`;

const registrationSnippet = `import tracee

app = workflow.compile()

tracee.init(
    app,
    graph_id="your-graph-id",
    name="Your workflow name",
    description="What this graph does",
    server_url="http://localhost:8000",
)

with tracee.trace():
    app.invoke(initial_state)`;

const metadataSnippet = `workflow.add_node("planner", create_planner_agent, metadata={
    "prompt_id": "planner-prompt",
    "model": "gpt-4o-mini",
    "has_tools": True,
})`;

function ScreenshotPlaceholder({ title, guidance, src }: { title: string; guidance: string; src?: string }) {
  return (
    <figure className="docs__screenshot">
      {src ? (
        <div className="docs__screenshot-frame docs__screenshot-frame--image">
          <img src={src} alt={title} className="docs__screenshot-img" loading="lazy" />
        </div>
      ) : (
        <div className="docs__screenshot-frame">
          <span className="docs__screenshot-icon">&#128247;</span>
          <p className="docs__screenshot-title">{title}</p>
        </div>
      )}
      <figcaption className="docs__screenshot-caption">{guidance}</figcaption>
    </figure>
  );
}

const pythonKeywords = new Set([
  "import", "from", "as", "def", "class", "return", "if", "elif", "else",
  "for", "while", "with", "try", "except", "finally", "raise", "yield",
  "and", "or", "not", "in", "is", "None", "True", "False", "pass", "break",
  "continue", "lambda", "del", "global", "nonlocal", "assert", "async", "await",
]);

function highlightPython(code: string): React.ReactNode[] {
  return code.split("\n").map((line, lineIdx) => {
    const parts: React.ReactNode[] = [];
    let i = 0;

    while (i < line.length) {
      // comments
      if (line[i] === "#") {
        parts.push(<span key={`${lineIdx}-${i}`} className="hl-comment">{line.slice(i)}</span>);
        i = line.length;
        continue;
      }

      // strings (double or single quoted, including triple)
      if (line[i] === '"' || line[i] === "'") {
        const quote = line[i];
        const triple = line.slice(i, i + 3) === quote.repeat(3);
        const end = triple ? quote.repeat(3) : quote;
        const start = i;
        i += triple ? 3 : 1;
        while (i < line.length) {
          if (line[i] === "\\" && i + 1 < line.length) { i += 2; continue; }
          if (line.slice(i, i + end.length) === end) { i += end.length; break; }
          i++;
        }
        parts.push(<span key={`${lineIdx}-${start}`} className="hl-string">{line.slice(start, i)}</span>);
        continue;
      }

      // words (keywords, builtins, identifiers)
      if (/[a-zA-Z_]/.test(line[i])) {
        const start = i;
        while (i < line.length && /[a-zA-Z0-9_]/.test(line[i])) i++;
        const word = line.slice(start, i);
        if (pythonKeywords.has(word)) {
          parts.push(<span key={`${lineIdx}-${start}`} className="hl-keyword">{word}</span>);
        } else {
          parts.push(word);
        }
        continue;
      }

      // numbers
      if (/[0-9]/.test(line[i])) {
        const start = i;
        while (i < line.length && /[0-9.]/.test(line[i])) i++;
        parts.push(<span key={`${lineIdx}-${start}`} className="hl-number">{line.slice(start, i)}</span>);
        continue;
      }

      // punctuation and whitespace — accumulate plain text
      const start = i;
      while (i < line.length && !/[a-zA-Z_0-9#"']/.test(line[i])) i++;
      parts.push(line.slice(start, i));
    }

    return lineIdx < code.split("\n").length - 1
      ? <span key={lineIdx}>{parts}{"\n"}</span>
      : <span key={lineIdx}>{parts}</span>;
  });
}

function CodeBlock({ code, label }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="docs__code-wrapper">
      {label && <p className="docs__code-label">{label}</p>}
      <div className="docs__code-container">
        <button type="button" className="docs__code-copy" onClick={handleCopy} aria-label="Copy code">
          {copied ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></svg>
          )}
        </button>
        <pre className="docs__code"><code>{highlightPython(code)}</code></pre>
      </div>
    </div>
  );
}

function Callout({ type, title, children }: { type: "info" | "tip" | "warning"; title?: string; children: React.ReactNode }) {
  const icons: Record<string, string> = { info: "\u2139\uFE0F", tip: "\uD83D\uDCA1", warning: "\u26A0\uFE0F" };
  return (
    <aside className={`docs__callout docs__callout--${type}`}>
      <span className="docs__callout-icon">{icons[type]}</span>
      <div className="docs__callout-body">
        {title && <p className="docs__callout-title">{title}</p>}
        <div className="docs__callout-text">{children}</div>
      </div>
    </aside>
  );
}

export function DocsPage() {
  const [activeId, setActiveId] = useState("overview");
  const scrollRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) setActiveId(visible[0].target.id);
      },
      { root, rootMargin: "-80px 0px -60% 0px", threshold: 0 },
    );

    sections.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="docs">
      <nav className="docs__rail" aria-label="Documentation sections">
        <div className="docs__rail-inner">
          <p className="docs__rail-heading">Documentation</p>
          <ul className="docs__rail-list">
            {sections.map(({ id, label }) => (
              <li key={id}>
                <a
                  href={`#${id}`}
                  className={`docs__rail-link${activeId === id ? " is-active" : ""}`}
                  onClick={(e) => { e.preventDefault(); scrollTo(id); }}
                >
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      <article className="docs__main" ref={scrollRef}>
        {/* ── Overview ──────────────────────────────────── */}
        <section id="overview" className="docs__section">
          <header className="docs__hero">
            <p className="docs__eyebrow">tracee documentation</p>
            <h1 className="docs__title">Build, inspect, and iterate on agentic workflows</h1>
            <p className="docs__lede">
              Tracee is an observability and prompt-engineering toolkit for multi-agent systems built on LangGraph.
              It gives you a visual graph of your workflow topology, a playground for prompt experimentation,
              and a versioned prompt library — all in one interface.
            </p>
          </header>

          <div className="docs__overview-grid">
            <div className="docs__overview-card">
              <h3 className="docs__overview-card-title">Graph</h3>
              <p className="docs__overview-card-desc">
                Visualize agent topology, inspect execution traces frame-by-frame, and run cognition analysis on completed runs.
              </p>
            </div>
            <div className="docs__overview-card">
              <h3 className="docs__overview-card-title">Playground</h3>
              <p className="docs__overview-card-desc">
                Author prompts with structured components, run experiments against live models, and compare outputs side by side.
              </p>
            </div>
            <div className="docs__overview-card">
              <h3 className="docs__overview-card-title">Prompts</h3>
              <p className="docs__overview-card-desc">
                Browse your saved prompt library, compare versions with a visual diff, and load any version into the playground.
              </p>
            </div>
          </div>
        </section>

        {/* ── Core Concepts ─────────────────────────────── */}
        <section id="concepts" className="docs__section">
          <h2 className="docs__section-title">Core Concepts</h2>
          <p className="docs__prose">
            These are the key abstractions you will encounter throughout Tracee. Understanding them makes the rest of the documentation easier to follow.
          </p>

          <dl className="docs__definition-list">
            <div className="docs__definition">
              <dt className="docs__term">Graph</dt>
              <dd className="docs__desc">
                A directed topology of agents and terminal states that mirrors the compiled LangGraph workflow.
                Registering a graph publishes its structure to the Tracee server so it can be visualized.
              </dd>
            </div>
            <div className="docs__definition">
              <dt className="docs__term">Trace</dt>
              <dd className="docs__desc">
                A recorded execution of a graph. Each trace captures the sequence of state transitions, LLM calls,
                tool invocations, and final outputs produced by a single <code>app.invoke()</code> run.
              </dd>
            </div>
            <div className="docs__definition">
              <dt className="docs__term">Prompt</dt>
              <dd className="docs__desc">
                A structured template composed of one or more components (system, user, assistant messages),
                optional tool definitions, variables, and an output schema. Prompts are the primary unit of authoring in the Playground.
              </dd>
            </div>
            <div className="docs__definition">
              <dt className="docs__term">Version</dt>
              <dd className="docs__desc">
                An immutable snapshot of a prompt. Every save in the Playground creates a new version, and the Prompts library
                lets you browse, compare, and revert across the full version history.
              </dd>
            </div>
            <div className="docs__definition">
              <dt className="docs__term">Layer</dt>
              <dd className="docs__desc">
                The Graph page supports three viewing layers — <strong>Intent</strong> (static topology),
                <strong>Execution</strong> (runtime trace playback), and <strong>Cognition</strong> (AI-powered analysis).
                Each layer reveals progressively deeper insight into what your agents are doing.
              </dd>
            </div>
            <div className="docs__definition">
              <dt className="docs__term">Cognition Analysis</dt>
              <dd className="docs__desc">
                An AI-generated summary that examines a completed trace for decision quality, tool usage patterns,
                and potential improvements. Available on the Cognition layer after a trace is loaded.
              </dd>
            </div>
          </dl>
        </section>

        {/* ── Setup Guide ────────────────────────────────── */}
        <section id="setup" className="docs__section">
          <h2 className="docs__section-title">Setup Guide</h2>
          <p className="docs__prose">
            This section walks through installing Tracee, starting the server, and connecting your
            LangGraph application so traces and graphs appear in the UI.
          </p>

          <h3 className="docs__subsection-title">Prerequisites</h3>
          <ul className="docs__list">
            <li>Python 3.11 or later</li>
            <li>A LangGraph workflow you want to instrument (<code>langgraph</code> installed in your project)</li>
            <li>An OpenAI API key if you plan to use the Playground or Cognition analysis features</li>
          </ul>

          <h3 className="docs__subsection-title">1. Install the package</h3>
          <p className="docs__prose">
            The core SDK is lightweight and only depends on <code>httpx</code>, <code>langchain-core</code>, and <code>pydantic</code>.
            Install it with pip or uv:
          </p>
          <CodeBlock code={installSnippet} label="core SDK" />
          <p className="docs__prose">
            To also run the Tracee server and UI locally, install with the server extras. This adds
            FastAPI, Uvicorn, and the LLM client libraries:
          </p>
          <CodeBlock code={installServerSnippet} label="with server + UI" />

          <h3 className="docs__subsection-title">2. Start the server</h3>
          <p className="docs__prose">
            The Tracee CLI provides a single command to start the server. The built-in UI is served
            automatically — no separate frontend build step required.
          </p>
          <CodeBlock code={startServerSnippet} label="start the server" />
          <p className="docs__prose">
            By default, the server listens on <code>http://0.0.0.0:8000</code>. You can override the
            port and host:
          </p>
          <CodeBlock code={startServerOptionsSnippet} label="custom host and port" />
          <p className="docs__prose">
            Open <code>http://localhost:8000</code> in your browser. You should see the Tracee UI with
            the Graph page. It will be empty until you register a workflow.
          </p>

          <h3 className="docs__subsection-title">3. Configure environment</h3>
          <p className="docs__prose">
            The server loads a <code>.env</code> file from the working directory on startup.
            At minimum, set your OpenAI key if you want to use the Playground or Cognition features:
          </p>
          <CodeBlock code={envSnippet} label=".env file" />

          <Callout type="info" title="Environment variables reference">
            <p>
              <code>OPENAI_API_KEY</code> — required for Playground runs and Cognition analysis.<br />
              <code>CORS_ORIGINS</code> — comma-separated allowed origins (defaults to <code>*</code>).<br />
              <code>TRACE_DB_PATH</code> — override the SQLite database location.<br />
              <code>TRACEE_COGNITION_MODEL</code> — LLM model for cognition analysis (defaults to <code>gpt-4o-mini</code>).
            </p>
          </Callout>

          <h3 className="docs__subsection-title">4. Instrument your LangGraph app</h3>
          <p className="docs__prose">
            In your application code, import <code>tracee</code>, register the compiled graph, and wrap
            invocations with <code>tracee.trace()</code>. This is typically three lines of code added
            to your existing workflow:
          </p>
          <CodeBlock code={integrationSnippet} label="full integration example" />

          <Callout type="tip" title="What each call does">
            <p>
              <code>tracee.init()</code> publishes the graph topology to the server and patches <code>invoke</code> /
              <code>ainvoke</code> to automatically attach tracing callbacks.<br />
              <code>tracee.trace()</code> is a context manager that records the full execution — state transitions,
              LLM calls, tool invocations, and outputs — and streams them to the server.
            </p>
          </Callout>

          <h3 className="docs__subsection-title">5. Verify the connection</h3>
          <p className="docs__prose">
            After running your instrumented app at least once:
          </p>
          <ol className="docs__steps">
            <li className="docs__step">
              <div className="docs__step-index">1</div>
              <div className="docs__step-copy">
                <h4 className="docs__step-title">Check the Graph page</h4>
                <p className="docs__step-body">
                  Open the Tracee UI and switch to the Graph tab. Your workflow topology should appear
                  with agent nodes and edges.
                </p>
              </div>
            </li>
            <li className="docs__step">
              <div className="docs__step-index">2</div>
              <div className="docs__step-copy">
                <h4 className="docs__step-title">Switch to Execution layer</h4>
                <p className="docs__step-body">
                  Toggle the layer to Execution and select your trace from the dropdown. You should see
                  the frame scrubber and be able to replay the execution step by step.
                </p>
              </div>
            </li>
            <li className="docs__step">
              <div className="docs__step-index">3</div>
              <div className="docs__step-copy">
                <h4 className="docs__step-title">Try the Playground</h4>
                <p className="docs__step-body">
                  Navigate to the Playground tab and create a simple prompt to confirm the server can
                  reach the LLM API. If it runs successfully, your setup is complete.
                </p>
              </div>
            </li>
          </ol>
        </section>

        {/* ── Graph ─────────────────────────────────────── */}
        <section id="graph" className="docs__section">
          <h2 className="docs__section-title">Graph</h2>
          <p className="docs__prose">
            The Graph page is the primary entry point for Tracee. It renders your workflow as an interactive node-and-edge
            diagram where each node represents an agent or terminal state. The three layers let you move from
            static structure to live execution replay to AI-powered analysis.
          </p>

          <h3 className="docs__subsection-title">Registering a graph</h3>
          <p className="docs__prose">
            Before anything appears on the Graph page, you need to register your compiled LangGraph workflow with the
            Tracee server. This is a one-time call that publishes the topology.
          </p>
          <CodeBlock code={registrationSnippet} label="graph registration" />

          <Callout type="tip" title="Enrich agent cards with metadata">
            <p>
              Include <code>prompt_id</code>, <code>model</code>, and <code>has_tools</code> in your <code>add_node</code> metadata
              to surface richer information on each agent card without opening the detail panel.
            </p>
          </Callout>
          <CodeBlock code={metadataSnippet} label="optional node metadata" />

          <h3 className="docs__subsection-title">Intent layer</h3>
          <p className="docs__prose">
            The default view. Shows the static graph topology — every agent, terminal node, and edge — without
            requiring any traced runs. Use this to verify that registration captured the correct structure.
          </p>
          <ScreenshotPlaceholder
            title="Graph canvas — Intent layer"
            guidance="Intent layer: static topology with agents and edges. No trace required."
            src={intentLayerImg}
          />

          <h3 className="docs__subsection-title">Execution layer</h3>
          <p className="docs__prose">
            Switch to the Execution layer and select a trace to replay it frame by frame.
            Active nodes highlight as state flows through the graph, and the Execution Inspector
            panel shows the state diff and schema at each frame.
          </p>
          <ScreenshotPlaceholder
            title="Execution layer with trace playback"
            guidance="Execution layer: trace selected, frame scrubber, and inspector for state at the current step."
            src={executionLayerImg}
          />

          <h3 className="docs__subsection-title">Cognition layer</h3>
          <p className="docs__prose">
            The Cognition layer adds AI-powered analysis on top of a completed trace. After selecting a trace,
            click <strong>Analyze</strong> to generate a summary of decision quality, tool usage, and improvement suggestions.
            Click any agent node to see per-node cognition details.
          </p>
          <ScreenshotPlaceholder
            title="Cognition analysis results"
            guidance="Cognition layer: analysis summary and per-node insights after a trace is loaded."
            src={cognitionLayerImg}
          />

          <h3 className="docs__subsection-title">Node detail panel</h3>
          <p className="docs__prose">
            Click any agent node to open its detail panel on the right side. The panel adapts to the current layer:
            intent shows metadata and connections, execution shows the state at that frame, and cognition shows
            AI analysis scoped to that specific agent.
          </p>
          <ScreenshotPlaceholder
            title="Agent detail panel (execution layer)"
            guidance="Click an agent node while on the execution layer and capture the detail panel showing state and operation timeline."
          />

          <h3 className="docs__subsection-title">Multi-graph switching</h3>
          <p className="docs__prose">
            If you have registered more than one workflow, a graph selector appears in the top-left controls.
            Switch between graphs without leaving the page — each graph retains its own selected trace and layer.
          </p>
        </section>

        {/* ── Playground ────────────────────────────────── */}
        <section id="playground" className="docs__section">
          <h2 className="docs__section-title">Playground</h2>
          <p className="docs__prose">
            The Playground is where you author, run, and compare prompt experiments against live models.
            It is split into two modes: <strong>Author</strong> for building the prompt, and <strong>Analysis</strong>
            for reviewing results after a run.
          </p>

          <h3 className="docs__subsection-title">Authoring prompts</h3>
          <p className="docs__prose">
            Prompts are composed of structured components — system instructions, user messages, and assistant
            pre-fills. Each component has its own editor. You can also attach tool definitions, declare variables,
            and define an output schema that the model should conform to.
          </p>
          <ScreenshotPlaceholder
            title="Playground — Author mode"
            guidance="Show the authoring workspace with at least two prompt components, a tool definition, and the output schema builder visible."
          />

          <Callout type="info" title="Guided start">
            <p>
              If you open an empty playground, the guided overlay walks you through creating your first component
              and choosing a model. You can dismiss it at any time.
            </p>
          </Callout>

          <h3 className="docs__subsection-title">Running experiments</h3>
          <p className="docs__prose">
            Click <strong>Run</strong> to send the resolved prompt to the selected model. The playground supports
            single runs and grouped experiment batches. After the run completes, the page switches to Analysis mode automatically.
          </p>

          <h3 className="docs__subsection-title">Comparing results</h3>
          <p className="docs__prose">
            Analysis mode presents run outputs side by side. If you set an <strong>anchor</strong> — an expected
            reference output — the comparison highlights deviations. You can promote any run output as the new anchor.
          </p>
          <ScreenshotPlaceholder
            title="Playground — Analysis mode with comparison"
            guidance="Show two run results side by side with an anchor set. Include the deviation indicators."
          />

          <h3 className="docs__subsection-title">Saving and versioning</h3>
          <p className="docs__prose">
            Every time you save a prompt in the Playground, a new version is created in the Prompts library.
            You can continue iterating in the Playground, and the full history is browseable from the Prompts tab.
          </p>
        </section>

        {/* ── Prompts ───────────────────────────────────── */}
        <section id="prompts" className="docs__section">
          <h2 className="docs__section-title">Prompts</h2>
          <p className="docs__prose">
            The Prompts page is a versioned library of every prompt you have saved. It provides three views
            on each version — <strong>Components</strong>, <strong>Resolved</strong>, and <strong>Diff</strong> — and
            lets you load any version directly into the Playground.
          </p>

          <h3 className="docs__subsection-title">Prompt list and search</h3>
          <p className="docs__prose">
            The left panel lists all saved prompts with search and sort controls (by name, version count, or
            last updated). Select a prompt to open its version tree.
          </p>
          <ScreenshotPlaceholder
            title="Prompts — Library list"
            guidance="Show the left prompt list with several entries, search bar visible, and one prompt selected."
          />

          <h3 className="docs__subsection-title">Version tree</h3>
          <p className="docs__prose">
            The center panel displays the version history for the selected prompt. Select a version to inspect,
            or toggle a second version for comparison mode.
          </p>
          <ScreenshotPlaceholder
            title="Version tree with comparison toggle"
            guidance="Show a prompt with 3+ versions, one active, one selected as compare target."
          />

          <h3 className="docs__subsection-title">Components vs Resolved vs Diff</h3>
          <p className="docs__prose">
            The detail workspace on the right has a segmented control to switch between views:
          </p>
          <ul className="docs__list">
            <li><strong>Components</strong> — read-only cards for each prompt component, with tool and variable chips and the output schema table.</li>
            <li><strong>Resolved</strong> — the fully interpolated prompt text as it would be sent to the model.</li>
            <li><strong>Diff</strong> — a side-by-side diff of the resolved text between two versions (only visible when a comparison target is selected).</li>
          </ul>
          <ScreenshotPlaceholder
            title="Diff view between two versions"
            guidance="Select two versions, switch to the Diff tab, and capture the side-by-side diff output."
          />

          <h3 className="docs__subsection-title">Loading into Playground</h3>
          <p className="docs__prose">
            Click <strong>Load in Playground</strong> from any version detail to open that exact version in the Playground
            authoring workspace. The URL carries the prompt and version IDs so you can also share deep links.
          </p>
        </section>

        {/* ── Workflows ─────────────────────────────────── */}
        <section id="workflows" className="docs__section">
          <h2 className="docs__section-title">Workflows</h2>
          <p className="docs__prose">
            Tracee's three pages are designed to work together as a continuous feedback loop.
            Here is a typical end-to-end workflow for iterating on an agentic system.
          </p>

          <ol className="docs__steps">
            <li className="docs__step">
              <div className="docs__step-index">1</div>
              <div className="docs__step-copy">
                <h4 className="docs__step-title">Design in Prompts</h4>
                <p className="docs__step-body">
                  Start by browsing existing prompts or creating a new one. Define your system instructions,
                  user message template, tools, and output schema.
                </p>
              </div>
            </li>
            <li className="docs__step">
              <div className="docs__step-index">2</div>
              <div className="docs__step-copy">
                <h4 className="docs__step-title">Experiment in Playground</h4>
                <p className="docs__step-body">
                  Load the prompt into the Playground, run experiments with different variable values or model
                  configurations, and compare outputs against your anchor.
                </p>
              </div>
            </li>
            <li className="docs__step">
              <div className="docs__step-index">3</div>
              <div className="docs__step-copy">
                <h4 className="docs__step-title">Observe in Graph</h4>
                <p className="docs__step-body">
                  Once your prompt is deployed in a LangGraph workflow, use the Graph page to trace executions,
                  replay state transitions, and run cognition analysis on completed runs.
                </p>
              </div>
            </li>
            <li className="docs__step">
              <div className="docs__step-index">4</div>
              <div className="docs__step-copy">
                <h4 className="docs__step-title">Iterate</h4>
                <p className="docs__step-body">
                  Use insights from execution traces and cognition analysis to refine your prompts.
                  Save a new version, re-run experiments, and verify improvements on the Graph page.
                </p>
              </div>
            </li>
          </ol>
        </section>

        {/* ── Troubleshooting ──────────────────────────── */}
        <section id="troubleshooting" className="docs__section">
          <h2 className="docs__section-title">Troubleshooting</h2>

          <div className="docs__faq">
            <details className="docs__faq-item">
              <summary className="docs__faq-question">The Graph page is empty</summary>
              <div className="docs__faq-answer">
                <p>
                  The graph only appears after you register a compiled workflow with <code>tracee.init()</code>.
                  Make sure:
                </p>
                <ul className="docs__list">
                  <li>The Tracee server is running and reachable at the configured <code>server_url</code>.</li>
                  <li>You called <code>tracee.init(app, ...)</code> after <code>workflow.compile()</code>.</li>
                  <li>The registration call completed without errors — check your terminal output.</li>
                </ul>
                <p>
                  After fixing, click <strong>check again</strong> on the setup guide or refresh the page.
                </p>
              </div>
            </details>

            <details className="docs__faq-item">
              <summary className="docs__faq-question">Execution layer shows no traces</summary>
              <div className="docs__faq-answer">
                <p>
                  Traces appear only after at least one execution is recorded via <code>tracee.trace()</code>.
                  Run your workflow inside the trace context manager:
                </p>
                <CodeBlock code={`with tracee.trace():\n    app.invoke(initial_state)`} />
                <p>
                  Then switch to the Execution layer and select the trace from the dropdown.
                </p>
              </div>
            </details>

            <details className="docs__faq-item">
              <summary className="docs__faq-question">Diff tab does not appear in Prompts</summary>
              <div className="docs__faq-answer">
                <p>
                  The Diff view only becomes available when you select a comparison target.
                  In the version tree, toggle the compare checkbox on a second version. The segmented control
                  will then show the Diff option.
                </p>
              </div>
            </details>

            <details className="docs__faq-item">
              <summary className="docs__faq-question">A prompt version is missing from the library</summary>
              <div className="docs__faq-answer">
                <p>
                  Versions are created when you explicitly save from the Playground. If a version seems missing:
                </p>
                <ul className="docs__list">
                  <li>Confirm the save completed — look for the success indicator in the Playground.</li>
                  <li>Check if you are viewing the correct prompt — the list supports search and sort.</li>
                  <li>Deleted prompts and their versions are permanently removed. There is no undo for deletion.</li>
                </ul>
              </div>
            </details>

            <details className="docs__faq-item">
              <summary className="docs__faq-question">Cognition analysis is not available</summary>
              <div className="docs__faq-answer">
                <p>
                  Cognition analysis requires a completed trace on the Cognition layer. Make sure:
                </p>
                <ul className="docs__list">
                  <li>You have switched to the <strong>Cognition</strong> layer using the layer toggle.</li>
                  <li>A trace is selected from the trace dropdown.</li>
                  <li>The trace has finished executing — in-progress traces cannot be analyzed.</li>
                </ul>
              </div>
            </details>
          </div>
        </section>
      </article>
    </div>
  );
}
