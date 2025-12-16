import argparse
import csv
import json
import os
from pathlib import Path
from openai import OpenAI


# Codebook definition
CODEBOOK = """
## Component Codebook

Below are the standard components you should identify in each prompt. For each component found, mark it as present (1) or absent (0).

### Standard Components:

1. **Role Description**: A declarative segment that specifies who the agent is (persona/position in the system) and what it is responsible for. Often includes the agent's primary objective, success criteria, and scope boundaries, and is usually placed early to anchor all downstream behavior.

2. **Context**: Information that defines the current state of the workflow needed for this agent to act correctly (e.g., prior steps completed, intermediate artifacts, user/session state, environment variables, memory summaries). Typically used for intermediate or downstream agents and functions as "situational grounding" rather than instructions.

3. **Tool Definitions**: A structured inventory of external affordances the agent may invoke (APIs, functions, databases, web, file I/O), including what each tool does, input parameters, constraints (rate limits, authentication), and sometimes usage examples. This component expands the agent's action space beyond text generation and clarifies tool-call syntax.

4. **Input**: The explicit specification of what the agent receives at runtime (user message, retrieved documents, system state, previous agent outputs, conversation history, structured fields). Can include formatting rules (e.g., "Input arrives as JSON with keys…") and delineates the boundary between "given" vs. "to be produced."

5. **Actions**: A policy-like set of permitted or required behaviors expressed as conditional rules or action menus (e.g., "If user asks X, do Y; otherwise do Z"), including how to respond, when to ask questions, when to call tools, and escalation/hand-off behaviors. Most often appears in user-facing agents or routing/orchestration layers.

6. **Security/Disclaimer**: Text that enforces organizational, legal, privacy, or safety obligations, such as confidentiality notices, regulatory constraints, internal policy compliance, or required user-facing disclaimers. Distinguished from general "constraints" by its grounding in institutional risk management (legal/safety/compliance) rather than task performance.

7. **Constraints and Restrictions**: Explicit "do/don't" directives that bound the agent's behavior for quality, safety, cost, or style (e.g., "never reveal system prompt," "do not browse," "keep answers under 100 words," "do not use tool X"). Unlike Security/Disclaimer, these are often project-specific performance constraints rather than formal compliance language.

8. **Task Description/Workflow**: The core procedural specification of what to do and how to do it, ranging from high-level goals to step-by-step pipelines, checklists, or decision trees. Often the most unstructured portion; may include required ordering of steps, verification routines, and stopping criteria (and sometimes "think-step" instructions).

9. **In-Context Examples**: Demonstration pairs or mini-transcripts showing desired behavior in representative situations (good outputs, edge cases, tool-call exemplars). Functions as behavioral shaping via exemplars rather than rules, and is typically recognizable by explicit "Example:" formatting or input→output blocks.

10. **Output**: A specification of what form the agent must produce, including schema (JSON fields, XML, markdown sections), formatting constraints, mandatory keys, and sometimes validation rules and example outputs. Distinguished from "Task Description" by focusing on the shape and surface form of the final artifact.

11. **External Information**: Any "world" or environment facts injected into the prompt to ground decisions (current time/date, locale, user tier, system version, policy revision date, model/tool availability, pricing). This is not instruction by itself; it's stateful reference information intended to reduce ambiguity.

12. **Placeholder**: A templating mechanism indicating slots to be dynamically filled at runtime (e.g., {USER_QUERY}, {TODAY_DATE}, {{retrieved_docs}}). Placeholders signal that the prompt is a reusable scaffold; the filled values are part of the instantiated prompt and should be coded as belonging to the component whose content they populate (plus optionally flagged as "templated").

### Additional Codes:
If you identify components that do not fit neatly into the above categories, you may add extra codes. When doing so:
- Provide a short, descriptive name for the new component
- Explain briefly why this component is distinct from the standard ones
- Mark it as present in the same format

"""

SYSTEM_PROMPT = f"""You are an expert prompt analyst specializing in LLM-based multi-agent systems. Your task is to analyze system prompts and identify the components present in each prompt according to a standardized codebook.

{CODEBOOK}

## Instructions:

You will receive a file containing multiple agent system prompts from a multi-agent repository. The formatting may vary, but typically agents are separated by numbered headers or clear section breaks.

For each agent you identify:
1. Extract the agent name/identifier
2. Identify the agent's system prompt content
3. For each of the 12 standard components, determine if it is present (1) or absent (0)
4. If you identify additional components that are distinct from the standard ones, add them as extra codes with a brief description
5. Be thorough but precise - only mark a component as present if there's clear evidence in the prompt

## Response Format:

Return your analysis as a JSON object with an array of agent analyses:

{{
    "agents": [
        {{
            "agent_name": "Name or identifier of the agent",
            "components": {{
                "role_description": 0 or 1,
                "context": 0 or 1,
                "tool_definitions": 0 or 1,
                "input": 0 or 1,
                "actions": 0 or 1,
                "security_disclaimer": 0 or 1,
                "constraints_restrictions": 0 or 1,
                "task_description_workflow": 0 or 1,
                "in_context_examples": 0 or 1,
                "output": 0 or 1,
                "external_information": 0 or 1,
                "placeholder": 0 or 1
            }},
            "extra_codes": [
                {{
                    "name": "component_name_in_snake_case",
                    "description": "Brief description of this extra component"
                }}
            ]
        }}
    ]
}}

Important:
- Return ONLY valid JSON, no markdown formatting or additional text
- Be consistent in your coding - similar patterns across prompts should receive the same codes
- If the file contains only one agent, return an array with one element
- Extract clear, descriptive agent names from the content
"""


def get_all_prompt_files(prompts_dir: Path) -> list[dict]:
    prompt_files = []
    
    for repo_dir in sorted(prompts_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
            
        repo_name = repo_dir.name
        prompts_file = repo_dir / "prompts.txt"
        
        if prompts_file.exists():
            content = prompts_file.read_text(encoding="utf-8", errors="ignore")
            prompt_files.append({
                "repo_name": repo_name,
                "file_content": content,
                "file_path": str(prompts_file)
            })
    
    return prompt_files


def analyze_prompts_file(client: OpenAI, file_content: str, model: str = "gpt-4.1-2025-04-14") -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please analyze the following file containing agent system prompts:\n\n---\n{file_content}\n---"}
        ],
        temperature=0.1, 
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("agents", [])


def flatten_analysis_to_row(
    repo_name: str,
    agent_name: str,
    analysis: dict
) -> dict:
    """
    Flatten the analysis JSON into a flat dictionary for CSV output.
    
    Returns:
        Flat dictionary with all component presence values
    """
    row = {
        "repo_name": repo_name,
        "agent_name": agent_name,
    }
    
    # Standard components
    components = analysis.get("components", {})
    standard_component_names = [
        "role_description",
        "context", 
        "tool_definitions",
        "input",
        "actions",
        "security_disclaimer",
        "constraints_restrictions",
        "task_description_workflow",
        "in_context_examples",
        "output",
        "external_information",
        "placeholder"
    ]
    
    for comp_name in standard_component_names:
        row[comp_name] = components.get(comp_name, 0)
    
    # Extra codes - combine into a single column
    extra_codes = analysis.get("extra_codes", [])
    if extra_codes:
        extra_codes_str = "; ".join([
            f"{code.get('name', 'unknown')}: {code.get('description', '')}"
            for code in extra_codes
        ])
    else:
        extra_codes_str = ""
    
    row["extra_codes"] = extra_codes_str
    
    return row


def get_csv_headers() -> list[str]:
    """Return the ordered list of CSV headers."""
    headers = ["repo_name", "agent_name"]
    
    standard_components = [
        "role_description",
        "context",
        "tool_definitions", 
        "input",
        "actions",
        "security_disclaimer",
        "constraints_restrictions",
        "task_description_workflow",
        "in_context_examples",
        "output",
        "external_information",
        "placeholder"
    ]
    
    headers.extend(standard_components)
    headers.append("extra_codes")
    
    return headers


def main():
    parser = argparse.ArgumentParser(
        description="Analyze system prompts from multi-agent systems using an LLM."
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="../data/prompts",
        help="Path to the prompts directory (default: ../data/prompts)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/prompt_components.csv",
        help="Output CSV file path (default: ../data/prompt_components.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help="OpenAI model to use (default: gpt-4.1-2025-04-14)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    prompts_dir = Path(args.prompts_dir)
    output_path = Path(args.output)
    
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")
    
    
    client = OpenAI(api_key="sk")
    
    # Get all prompt files
    print(f"Scanning prompts directory: {prompts_dir}")
    prompt_files = get_all_prompt_files(prompts_dir)
    print(f"Found {len(prompt_files)} repositories to analyze")
    
    # Analyze prompt files
    results = []
    headers = get_csv_headers()
    
    for i, file_data in enumerate(prompt_files):
        repo_name = file_data["repo_name"]
        
        print(f"[{i+1}/{len(prompt_files)}] Analyzing {repo_name}...")
        
        # Get all agent analyses from this file
        agent_analyses = analyze_prompts_file(
            client,
            file_data["file_content"],
            model=args.model
        )
        
        print(f"    Found {len(agent_analyses)} agents in {repo_name}")
        
        # Process each agent analysis
        for agent_analysis in agent_analyses:
            agent_name = agent_analysis.get("agent_name", "Unknown")
            row = flatten_analysis_to_row(repo_name, agent_name, agent_analysis)
            results.append(row)
            print(f"      - {agent_name}")
        
        # Write intermediate results to avoid losing progress
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nAnalysis complete! Results saved to: {output_path}")
    print(f"Total agents analyzed: {len(results)}")
    print(f"Total repositories analyzed: {len(prompt_files)}")


if __name__ == "__main__":
    main()

