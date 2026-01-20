from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.state.schema import AnalysisState
from backend.tools.execution_tools import execute_code_safely
import os


CODING_SYSTEM_PROMPT = """You are a coding agent in a data analysis system. Your role is to:

1. Write executable Python code based on the analysis plan and instructions
2. Ensure the code is safe, efficient, and follows best practices
3. Use only allowed libraries and operations

**Available Libraries:**
- pandas (imported as pd)
- numpy (imported as np)
- matplotlib.pyplot (imported as plt)
- seaborn (imported as sns)
- sklearn (scikit-learn)
- scipy

**Dataset Access:**
- The dataset is available as variable: `df` or `dataset`
- Both refer to the same pandas DataFrame

**Code Requirements:**
- Use ONLY the allowed libraries above
- Save all plots using plt.savefig('plot_name.png') before creating new figures
- Use plt.figure() to create new figures if making multiple plots
- Include print() statements for key results and statistics
- Do NOT use: open(), file operations, imports other than allowed libraries
- Validate that columns exist before using them

**Code Structure:**
Your response should be ONLY the Python code, without any markdown formatting or explanation.
Do not include ```python or ``` markers.
Just provide the raw executable code.

Example structure:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# analysis code here
result = df['column'].mean()
print(f"Mean: {result}")

# visualization
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y')
plt.savefig('scatter_plot.png')
plt.close()
```

You have access to tools to validate your code and check dataset columns.
"""


def create_coding_agent(state: AnalysisState) -> AnalysisState:
    """Coding agent that generates executable Python code.
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated state with generated code and execution results
    """
    # get dataset
    dataset = state["dataset"]
    callbacks = state.get("callbacks", [])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=callbacks,
        metadata={"agent": "coding", "has_tools": False}
    )
    
    # execute agent to generate code
    try:
        system_message = SystemMessage(content=CODING_SYSTEM_PROMPT)
        user_message = HumanMessage(content=f"""Analysis Plan and Instructions:
{state['coding_prompt']}

Dataset columns available: {list(dataset.columns)}

Please generate the Python code to accomplish this analysis.
Return ONLY the code, no markdown formatting.""")
        
        response = llm.invoke([system_message, user_message], config={"callbacks": callbacks})
        code = response.content
        
        # clean up code if it has markdown formatting
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        state["generated_code"] = code
        state["messages"].append(AIMessage(content=f"Code generated successfully. {len(code)} characters."))
        
        # execute the code
        output_dir = os.path.join(os.path.dirname(__file__), "../../outputs")
        execution_result = execute_code_safely(code, dataset, output_dir)
        
        state["execution_result"] = execution_result
        state["next_agent"] = "summary"
        
        if execution_result["success"]:
            state["messages"].append(AIMessage(content="Code executed successfully."))
        else:
            error_info = execution_result.get("error", {})
            state["messages"].append(AIMessage(content=f"Code execution failed: {error_info.get('message', 'Unknown error')}"))
        
    except Exception as e:
        error_msg = f"Error in coding agent: {str(e)}"
        state["generated_code"] = ""
        state["execution_result"] = {"success": False, "error": {"message": str(e)}}
        state["next_agent"] = "summary"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
