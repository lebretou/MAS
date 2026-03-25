const registrationSnippet = [
  "import tracee",
  "",
  "app = workflow.compile()",
  "",
  "tracee.init(",
  '    app,',
  '    graph_id="your-graph-id",',
  '    name="Your workflow name",',
  '    description="What this graph does",',
  '    server_url="http://localhost:8000",',
  ")",
  "",
  "with tracee.trace():",
  "    app.invoke(initial_state)",
].join("\n");

const metadataSnippet = [
  'workflow.add_node("planner", create_planner_agent, metadata={',
  '    "prompt_id": "planner-prompt",',
  '    "model": "gpt-4o-mini",',
  '    "has_tools": True,',
  "})",
].join("\n");

const setupSteps = [
  {
    title: "point your app at the tracee server",
    body: "Use the same base URL your UI is reading from, usually http://localhost:8000 during local development.",
  },
  {
    title: "register the compiled graph once",
    body: "Call tracee.init(...) right after workflow.compile(). That publishes the topology to /api/graphs and makes the graph page discoverable.",
  },
  {
    title: "run executions inside tracee.trace()",
    body: "The graph page shows structure from registration, and the execution layer lights up after traced runs are sent to the server.",
  },
  {
    title: "add node metadata for a richer graph",
    body: "Include prompt ids, model names, and tool flags in add_node(..., metadata={...}) so agent cards have useful context.",
  },
];

interface GraphSetupGuideProps {
  onRefresh: () => void;
}

export function GraphSetupGuide({ onRefresh }: GraphSetupGuideProps) {
  return (
    <section className="graph-setup-guide" aria-labelledby="graph-setup-guide-title">
      <div className="graph-setup-guide__hero">
        <p className="graph-setup-guide__eyebrow">graph setup</p>
        <h1 id="graph-setup-guide-title" className="graph-setup-guide__title">
          Register your workflow to unlock the graph view
        </h1>
        <p className="graph-setup-guide__lede">
          This page only renders workflows that have been registered from your codebase. Add one setup call
          when you compile your graph, run a traced execution, then check again here.
        </p>
        <div className="graph-setup-guide__actions">
          <button type="button" className="graph-setup-guide__button" onClick={onRefresh}>
            check again
          </button>
          <p className="graph-setup-guide__hint">
            Tracee reads registered topologies from <code>/api/graphs</code>.
          </p>
        </div>
      </div>

      <div className="graph-setup-guide__grid">
        <article className="graph-setup-guide__card">
          <header className="graph-setup-guide__card-header">
            <p className="graph-setup-guide__card-label">checklist</p>
            <h2 className="graph-setup-guide__card-title">What to wire into your app</h2>
          </header>
          <ol className="graph-setup-guide__steps">
            {setupSteps.map((step, index) => (
              <li key={step.title} className="graph-setup-guide__step">
                <div className="graph-setup-guide__step-index">{index + 1}</div>
                <div className="graph-setup-guide__step-copy">
                  <h3 className="graph-setup-guide__step-title">{step.title}</h3>
                  <p className="graph-setup-guide__step-body">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </article>

        <article className="graph-setup-guide__card">
          <header className="graph-setup-guide__card-header">
            <p className="graph-setup-guide__card-label">recommended flow</p>
            <h2 className="graph-setup-guide__card-title">Minimal registration example</h2>
          </header>
          <pre className="graph-setup-guide__code">
            <code>{registrationSnippet}</code>
          </pre>
          <div className="graph-setup-guide__note">
            <p className="graph-setup-guide__note-title">optional metadata</p>
            <p className="graph-setup-guide__note-body">
              Add metadata to each node if you want prompt ids and model details to appear in the graph panels.
            </p>
            <pre className="graph-setup-guide__code graph-setup-guide__code--compact">
              <code>{metadataSnippet}</code>
            </pre>
          </div>
        </article>
      </div>
    </section>
  );
}
