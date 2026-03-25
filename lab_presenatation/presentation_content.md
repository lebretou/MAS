# LAb Presentation

## Motivation

This project started with a semi-structured interview with developers who have experience in developing LLM-based multi-agent system.

todo: I think here we would need definition of AI agent and what multi-agent system is. 

The questions we wanted to know about are:

1. What do you use your multi-agent system for? (domain)
2. What framework did you use for your MAS?
  a. What kind of agent roles there are?
   b. What are some of the prompt components that are always there?
3. What are the challenges you had while developing your MAS? Specifically, what was the part that took the longest time?
4. What are the interactions or UI you wish to have that can support or speed up your developing process?

## Semi-structured Interview

### Demographics and domain

15 developers, 9 industry project, 1 personal, 5 school projects
total: 20 distinct projects

primary domain	count	examples
analytics / extraction / evaluation	6	financial PDF-to-structured-data extraction, crisis fact-checking, design-chart structured output, root-cause analysis on logs, tax/text analytics, confidence calibration via debate
research / report / content workflows	4	literature review assistant, research report generation, research/content workflow, content drafting-reviewing
task automation / support / productivity	4	Adobe task-planning assistant, Amazon shopping/product support, generic task automation, task coordination/scheduling
healthcare / medical	2	healthcare decision support, medical imaging + diagnosis support
safety / simulation / benchmarking	2	persona-agent social simulation, attacker-defender jailbreak/alignment system
robotics / physical systems	1	multi-drone search-and-rescue
exploratory hackathon / prototyping	1	recent hackathon exploring MCP + Google ADK

### Coding process

We then had two rounds of coding with two authors and identified common themes from what developers said

### What we found

I think here we would also need to mention the distinction between MAS and agentic workflows. What we need to say is that we just ended with more responses that fits "agentic workflow" instead of MAS. 

#### two-stage process

We identified that developers often go through a two-stage process when developing test individual agent first -> then test the entire workflow.

#### trial-and-error and edge cases

Because of the demographics of our interviewees (which leans towards industry), developers care about the consistency of outputs and stability of the system. Therefore a lot of their time are spent running individual agents or the entire workflow, waiting for edge cases to come up and refine their prompt to rule out the edge cases. Therefore developers are constantly making small changes/tweaks to the prompts and observing behavior.

##### hard to establish the causal relationship between input (prompt) and output

another thing that made this difficult is that because of the stochastic behavior of LLM, it is hard to exactly control the behavior of models. and it is hard to establish the causal relationship between what input would incur what behavior. 

#### Prompt complexity and growth

Participants have mentioned that towards the end of the development the prompts can grow into large chunks of text. This length of the prompt also scale with the complexity of the project, number of collaborators involved, company-specific guidelines to follow, etc. Some developers also don't know where to start, since writing prompts are technically not their expertise. 

#### Stochastic system and difference from traditional programming

Since LLM agents are like the "backbone" of these systems, these systems are naturally stochastic which means that they can suffer from hallucination. This means two things:

1. As these systems are interconnected with the input and output of models, the completion of a workflow depend on the quality of outputs from every agent on the chain. Failed output from one agent can mean the failure of the entire workflow
2. unlike traditional programming where you have error logs that tell you where the error occur and a rather complete fault taxonomy, debugging in agentic workflow is much harder since error can happen anywhere on the chain, and the most of the time developers have to go through tedious logs in order to find the faulty agent.

### Component Analysis

Why doing a component analysis? 
On a higher level, we want to better understand the content and the composition of prompts from existing projects. 

1. To see if we can identify common agent roles
2. If we can identify common agent role, then we want to see if system prompts for agentic workflow share common structure.

If above are both true then we can have 

1. kickstart templates to help developers writing system prompt from zero
2. structure system prompts in components/sections to construct a more organized mental model

#### About the analysis

Process:
Selection criteria 
LLM-based MAS - implements >= 3 agents (separate models)
Open-source 
Contains explicit prompt text in the repo
Built on top of at least one of the common MAS agentic frameworks (e.g. LangChain, LangGraph, AutoGen, etc.) or implements a custom multi-agent orchestration but clearly describes multiple LLM agents/roles
Last commit within the past 12-18 months
more than 100 stars or more than 10 forks 

Selection Results: 
27 repos with 268 pieces of prompts 

Process

1. We first identified whether the prompt contains components using a codebook with LLM
2. we turned each prompt into an embedding and clustered similar prompts to identify recurring role families
3. we used an llm to generate a short human-readable label/summary for each cluster
4. we compared component distributions within each cluster to see which prompt sections are characteristic of each role

## Developer Workflow Model

From our findings, MAS prompt development can be described as a two-level iterative refinement loop under uncertainty. Every phase of both loops currently lacks adequate tooling — developers rely on print statements, raw logs, and mental bookkeeping.

```
┌─────────────────────────────────────────────────────────────┐
│              Outer Loop: Workflow-Level                      │
│                                                             │
│   ┌──────────┐    ┌─────────┐    ┌───────────┐    ┌──────┐ │
│   │ Compose  │───>│   Run   │───>│   Trace   │───>│Locate│ │
│   │ Agents   │    │Workflow │    │ Execution │    │Fault │ │
│   └──────────┘    └─────────┘    └───────────┘    └──┬───┘ │
│        ^                                             │      │
│        │              on success                     │      │
│        └─────────────────────────────────────────────┘      │
│                                                      │      │
│                                          on failure  │      │
│                                                      ▼      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │         Inner Loop: Individual agent level             ││
│  │                                                         ││
│  │  ┌────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ ││
│  │  │ Author │──>│ Execute │──>│ Observe │──>│Evaluate │ ││
│  │  │ Prompt │   │ Agent   │   │ Outputs │   │ Quality │ ││
│  │  └───┬────┘   └─────────┘   └─────────┘   └────┬────┘ ││
│  │      ^                                          │      ││
│  │      │              ┌────────┐                  │      ││
│  │      └──────────────│ Refine │<─────────────────┘      ││
│  │                     │ Prompt │                          ││
│  │                     └────────┘                          ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```


| Finding                                            | Where it hurts in the loop                                 |
| -------------------------------------------------- | ---------------------------------------------------------- |
| Cold start / don't know where to start             | **Author** — blank page problem                            |
| Prompt complexity & growth                         | **Author** — prompt becomes unmanageable                   |
| Trial-and-error / edge cases                       | **Execute → Observe** — must run many times                |
| Hard to establish causal input→output relationship | **Evaluate** — stochastic outputs make attribution hard    |
| Cascading failures across agent chain              | **Trace → Locate fault** in outer loop                     |
| No error taxonomy / tedious logs                   | **Trace → Locate fault** — no structured debugging support |


## Design Goals

**DG1: Lower the authoring barrier (Author phase)**
Provide templates and AI-assisted scaffolding so developers don't face a blank page. Grounded in the component analysis — the role clusters and their characteristic component profiles become starting templates.

**DG2: Make prompts tractable as they grow (Author phase)**
Decompose monolithic prompt text into named, typed sections (role, task, constraints, I/O rules, examples, etc.) so developers maintain a structured mental model even as prompts scale.

**DG3: Surface output patterns across stochastic runs (Observe + Evaluate phases)**
Let developers run an agent multiple times and see consistency/variance at a glance — not just read individual outputs. Support schema validation, field-level distribution analysis, and embedding-based similarity visualization.

**DG4: Support systematic prompt iteration (Refine phase)**
Provide version control and structured comparison so developers can track what changed in a prompt and what effect it had on outputs, replacing ad-hoc trial-and-error with deliberate experimentation.

**DG5: Enable workflow-level debugging (Outer loop: Trace + Locate fault)**
Visualize the agent topology, overlay execution traces, and let developers scrub through execution frames to locate the faulty agent — replacing the current practice of reading through raw logs.

---

## Slide Deck Plan

### Slide 1 — Title

**Content:** Title slide — project name, your name, lab name, date
**Speaker notes:** "Today I'll walk through our work on understanding how developers build LLM-based multi-agent systems, what makes it hard, and a tool we built to address those pain points."

### Slide 2 — What are we talking about?

**Content:** Brief definitions with a small visual (e.g. a simple 3-node agent diagram):

- AI Agent: an LLM-powered component with a role, tools, and a system prompt that can take actions
- Multi-agent system / agentic workflow: multiple agents chained or orchestrated together to accomplish a complex task

**Speaker notes:** "Quick level-set. When I say agent, I mean an LLM wrapped with a system prompt and possibly tools. A multi-agent system chains several of these together. In practice, most of what our participants built leans closer to 'agentic workflows' — sequential or supervised pipelines — rather than fully autonomous multi-agent coordination."

### Slide 3 — Research questions

**Content:** Two questions:

1. What are the challenges developers face when building multi-agent systems?
2. Can we build tooling that addresses these challenges?

**Speaker notes:** "Our overarching questions. We started with formative interviews, then built a tool informed by what we learned."

### Slide 4 — Interview method

**Content:**

- Semi-structured interviews, 15 developers, 20 distinct projects
- Questions: domain, frameworks, agent roles, prompt components, challenges, wished-for tooling

**Speaker notes:** "We interviewed 15 developers with hands-on MAS experience across 20 projects. A mix of industry, personal, and school projects. The interview covered what they build, how they build it, what hurts, and what they wish they had."

### Slide 5 — Demographics

**Content:** The domain table (condensed — just domain + count, maybe a horizontal bar chart):


| Domain                                   | Count |
| ---------------------------------------- | ----- |
| Analytics / extraction / evaluation      | 6     |
| Research / report / content workflows    | 4     |
| Task automation / support / productivity | 4     |
| Healthcare / medical                     | 2     |
| Safety / simulation / benchmarking       | 2     |
| Robotics / physical systems              | 1     |
| Exploratory / prototyping                | 1     |


**Speaker notes:** "The spread across domains is important — the pain points we found are not domain-specific. Whether someone is building a financial extraction pipeline or a medical support system, they're hitting the same development challenges. Two rounds of open coding with two authors converged on four themes."

### Slide 6 — Finding 1: Two-stage development process

**Content:** Simple visual:

```
Stage 1: Test individual agent → Stage 2: Test entire workflow
         (inner loop)                    (outer loop)
```

**Speaker notes:** "The most structural finding. Every developer described a two-stage process: get each agent working in isolation first, then wire them together and test the full pipeline. These are fundamentally different activities with different needs — the first is about prompt quality, the second is about system debugging."

### Slide 7 — Finding 2: Trial-and-error under uncertainty

**Content:** Key bullets:

- Constant small tweaks to prompts, then re-run
- Hard to establish causal relationship between prompt change and output change
- Stochastic outputs mean a single run proves nothing

**Speaker notes:** "The inner loop is deeply frustrating. Developers make a small change, run the agent, look at the output, and can't tell if the difference is because of their change or just LLM variance. There's no structured way to compare across runs or versions."

### Slide 8 — Finding 3: Prompt complexity & cold start

**Content:** Two sub-points:

- Prompts grow into large unstructured text blobs over time
- Developers don't know where to start — prompt writing is not their expertise

**Speaker notes:** "Prompts start small but accumulate edge-case patches, company guidelines, output format instructions. By the end they're thousands of tokens of unstructured text. And at the beginning, some developers stare at a blank textbox not knowing what sections a good agent prompt should even have."

### Slide 9 — Finding 4: Workflow debugging is broken

**Content:** Comparison visual:


|               | Traditional software        | Agentic workflow              |
| ------------- | --------------------------- | ----------------------------- |
| Errors        | Deterministic, stack traces | Stochastic, no clear taxonomy |
| Debugging     | Jump to error location      | Read through entire log chain |
| Failure scope | Usually localized           | One agent fails → cascade     |


**Speaker notes:** "Unlike traditional programming where you get a stack trace pointing to line 42, a bad output from agent 3 in a 6-agent pipeline could manifest as a completely unrelated failure in agent 6. Developers told us they spend hours reading through raw logs trying to find which agent went wrong."

### Slide 10 — The developer workflow model

**Content:** The two-level loop diagram:

```
┌─────────────────────────────────────────────────────────┐
│          Outer Loop: Workflow-Level                      │
│  Compose → Run Workflow → Trace → Locate Fault          │
│       ↑                           │                     │
│       └───── on success ──────────┘                     │
│                              on failure ↓               │
│  ┌────────────────────────────────────────────────────┐ │
│  │     Inner Loop: Single-Agent Refinement            │ │
│  │  Author → Execute → Observe → Evaluate → Refine   │ │
│  │    ↑                                    │          │ │
│  │    └────────────────────────────────────┘          │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Speaker notes:** "Synthesizing the findings, we arrive at this model. MAS development is a two-level loop. The inner loop is single-agent prompt refinement — author, execute, observe, evaluate, refine, repeat. The outer loop is workflow-level — compose agents, run the pipeline, trace execution, locate faults. Every phase of both loops currently lacks adequate tooling. This model gives us a frame to derive design goals."

### Slide 11 — Mapping findings to the loop

**Content:** The mapping table:


| Finding                             | Loop phase                   |
| ----------------------------------- | ---------------------------- |
| Cold start / blank page             | Author                       |
| Prompt complexity growth            | Author                       |
| Trial-and-error, stochastic outputs | Execute → Observe → Evaluate |
| Causal attribution difficulty       | Evaluate                     |
| Cascading failures                  | Outer: Trace → Locate        |
| No error taxonomy / tedious logs    | Outer: Trace → Locate        |


**Speaker notes:** "Each finding maps to a specific phase. This tells us exactly where tooling interventions are needed. But before we could design solutions for the authoring phase, we needed to understand: what does a well-structured prompt actually look like?"

### Slide 12 — Component analysis: motivation

**Content:**

- "Before we could help developers *author* prompts, we needed to understand prompt *structure*"
- Two questions:
  1. Do system prompts share common structural components?
  2. Are there recurring agent roles with characteristic component profiles?
- If yes → templates for cold start, structured editing for complexity

**Speaker notes:** "This is where the component analysis comes in. It directly feeds into two design goals: helping developers start from a template, and giving them a structured editor instead of a blank text box. We analyzed open-source MAS repos to answer these questions."

### Slide 13 — Component analysis: method & data

**Content:**

- 27 open-source repos, 268 system prompts
- Selection: ≥3 agents, explicit prompts, active repos, common frameworks
- Process:
  1. LLM-based component coding with codebook
  2. Embed prompts → cluster into role families
  3. LLM-generated cluster labels
  4. Component distribution per cluster

**Speaker notes:** "We collected 268 prompts from 27 repos. We used an LLM with a codebook to identify which components each prompt contains — things like role description, task, constraints, I/O rules, examples, safety instructions. Then we embedded the prompts and clustered them to find natural role families."

### Slide 14 — Component analysis: results

**Content:** Two visuals side by side:

1. Cluster overview — top clusters (Task Orchestration: 84, Trading Decision: 52, Phase-Based Workflow: 44, Data Science: 38, Expert Agent: 9, Multi-Agent Dev Roles: 9...)
2. Component × cluster heatmap from the notebook (the seaborn heatmap showing which components are enriched in which clusters)

**Speaker notes:** "We found 12 role clusters. The big ones — task orchestration, domain-specific analysis, phase-based workflow — account for most prompts. The heatmap shows which components are characteristic of each role. For example, task orchestration prompts almost always have role, task, and constraints, while data science agents emphasize I/O rules and examples. These profiles become the basis for our templates."

### Slide 15 — Design goals

**Content:** Five DGs mapped to loop phase:


| #   | Design Goal                                    | Loop Phase            |
| --- | ---------------------------------------------- | --------------------- |
| DG1 | Lower the authoring barrier with templates     | Author                |
| DG2 | Structure prompts into named components        | Author                |
| DG3 | Surface output patterns across stochastic runs | Observe + Evaluate    |
| DG4 | Support systematic prompt iteration            | Refine                |
| DG5 | Enable workflow-level debugging                | Outer: Trace + Locate |


**Speaker notes:** "Five design goals, each targeting a specific pain point in the loop. DG1 and DG2 are directly informed by the component analysis — we know what components exist and which are common per role, so we can offer structured templates. DG3-4 address the trial-and-error problem. DG5 tackles the outer loop."

### Slide 16 — Tracee: overview

**Content:** System diagram or annotated screenshot of the app shell showing three surfaces:

- Playground (prompt authoring + execution + analysis) → DG1-4
- Graph Viewer (workflow visualization + trace debugging) → DG5
- Prompt Library (versioning + browsing) → DG4

**Speaker notes:** "We built Tracee, a developer tool for MAS prompt engineering and debugging. It has three surfaces. The playground is where you author prompts, run them, and analyze outputs. The graph viewer shows your agent topology and lets you replay execution traces. The prompt library manages versions."

### Slide 17 — DG1: Guided start

**Content:** [screenshot of GuidedPromptStart panel — archetype catalog cards and staged question flow]

- Label: "DG1: Lower the authoring barrier"
- Callouts: archetype selection from cluster analysis → staged questions → AI-assisted scaffold

**Speaker notes:** "For DG1, when a developer opens a new prompt, they can launch the guided start. It shows archetypes derived from our cluster analysis — task orchestrator, data science agent, etc. The developer picks one, answers a few scoping questions, and gets a structured prompt draft with the right components pre-filled. This replaces the blank page."

### Slide 18 — DG2: Component-based prompt editor

**Content:** [screenshot of PromptForm with component-based editor — multiple collapsible sections like Role, Task, Constraints, I/O Rules, each as a separate editable block with colored type badges]

- Label: "DG2: Structure prompts into components"
- Callouts: typed sections (role, task, constraints, outputs, examples, safety...), toggle-able, color-coded, with PromptStructureOutline sidebar

**Speaker notes:** "For DG2, instead of a single text area, prompts are decomposed into typed components — the same types from our codebook. Each section is independently editable, can be toggled on/off, and maps to a message role. The outline panel on the side shows the structure at a glance. This keeps prompts organized even as they grow to thousands of tokens."

### Slide 19 — DG3: Multi-run analysis

**Content:** [screenshot of ResultsComparison with SimilarityScatterplot — embedding-based 2D projection of multiple run outputs as colored dots, plus field selector toolbar]

- Label: "DG3: Surface output patterns"
- Callouts: N parallel runs → embedding scatterplot (tight cluster = consistent, spread = unstable), field-level drill-down, schema validation badges

**Speaker notes:** "For DG3, the developer runs the same prompt multiple times in parallel. Instead of reading each output individually, they see a scatterplot where each dot is a run's output projected by embedding similarity. Tight clusters mean consistent behavior. Outliers are immediately visible. They can also drill into specific JSON fields to see value distributions, and schema validation flags structural deviations."

### Slide 20 — DG3 continued: Anchor-based comparison

**Content:** [screenshot of RunDetailView showing OutputDiffView — diff against anchor output, plus JsonTreeView]

- Label: "DG3: Anchor comparison"
- Callouts: promote a "good" run as anchor → all subsequent runs diff against it → structural diff view

**Speaker notes:** "Developers can mark a good output as the 'anchor.' Every subsequent run is automatically compared against it, both as a JSON diff and as a similarity score. This turns the vague question 'is this output okay?' into a concrete 'how does this differ from what I wanted?'"

### Slide 21 — DG4: Version tree & diff

**Content:** [screenshot of PromptVersionTree showing branching version history with lane layout, colored component badges, and diff summary annotations + PromptDiffWorkspace showing side-by-side component comparison]

- Label: "DG4: Systematic iteration"
- Callouts: branching version history, per-version diff summary, load/compare any two versions, run history per version

**Speaker notes:** "For DG4, every prompt save creates a version. The version tree shows the full history with branching — you can see what was changed at each step. The diff workspace compares any two versions component-by-component. Combined with the run history per version, developers can answer: 'I changed the constraints section in version 3 — did that actually improve consistency?'"

### Slide 22 — DG5: Graph viewer + trace playback

**Content:** [screenshot of GraphViewer in execution layer — React Flow canvas with agent nodes, LayerToggle (intent/execution), TraceSelector panel, and FrameScrubber timeline at bottom]

- Label: "DG5: Workflow-level debugging"
- Callouts: intent layer (static topology) vs execution layer (live trace overlay), frame scrubber, node highlighting during playback

**Speaker notes:** "For DG5, the graph viewer shows the agent topology. The intent layer is the static design — what agents exist and how they connect. Switch to the execution layer, select a trace, and you see the actual execution overlaid on the graph. The frame scrubber lets you step through the execution frame by frame. Each agent node lights up with its LLM calls, tool usage, and latency. Instead of reading log files, you visually walk through what happened."

### Slide 23 — DG5 continued: Agent detail

**Content:** [screenshot of AgentDetailPanel in execution mode — per-agent execution data: LLM calls, tool calls, inputs/outputs, latency]

- Label: "DG5: Locate the faulty agent"
- Callouts: click any agent node → see its execution detail, LLM prompts/responses, tool calls, timing

**Speaker notes:** "When you spot something wrong in the trace, click the agent node to inspect it. You see its LLM calls, tool invocations, and the actual inputs and outputs. This is the 'locate fault → drop into inner loop' transition from our model — once you identify the problem agent, you can go to the playground and iterate on its prompt."

### Slide 24 — Summary

**Content:** Clean 2-column mapping:


| Design Goal                  | Tracee Feature                        |
| ---------------------------- | ------------------------------------- |
| DG1: Lower authoring barrier | Guided start + cluster templates      |
| DG2: Structure prompts       | Component-based editor                |
| DG3: Surface output patterns | Multi-run + scatterplot + anchor diff |
| DG4: Systematic iteration    | Version tree + diff workspace         |
| DG5: Workflow debugging      | Graph viewer + trace playback         |


**Speaker notes:** "To recap the mapping. Each design goal derived from our interview findings is addressed by a specific feature surface in Tracee."

### Slide 25 — What's next / limitations

**Content:** Possible items:

- User evaluation (usability study with MAS developers)
- Support for more LLM providers beyond OpenAI
- Tighter integration with frameworks (LangGraph, AutoGen)
- Connecting the two loops — from graph viewer fault → playground for that agent's prompt

**Speaker notes:** "Next steps. We want to run a user study with MAS developers to evaluate whether the tool actually speeds up their workflow. On the engineering side, we want to better connect the two loops — when you find a faulty agent in the graph viewer, one click should open its prompt in the playground."