# Step 1: Define tools and model

from langchain.tools import tool
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="MiniMax-M3",
    # temperature=,
    # max_tokens=,
    # timeout=,
    # max_retries=,
    # ...
)



# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Step 2: Define state

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Step 3: Define model node
from langchain.messages import SystemMessage


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Step 4: Define tool node

from langchain.messages import ToolMessage


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Step 5: Define logic to determine whether to end

from typing import Literal
from langgraph.graph import StateGraph, START, END


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Step 6: Build agent

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


# Display the agent system
#from IPython.display import Image, display
# Show the agent in Jupyter Notebook
#display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
# Show the agent in terminal
print("==="*10 + "Agent graph" + "==="*10)
agent.get_graph(xray=True).print_ascii()


from pathlib import Path

png = agent.get_graph(xray=True).draw_mermaid_png()
path = Path("ex1_agent_graph.png")
path.write_bytes(png)
print(f"Saved to {path.resolve()}")

#import subprocess
#subprocess.run(["xdg-open", str(path)])

# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 32 and 41. And then multiply the result by 2. And then divide the result by 4. Give me step-by-step thinking process.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

"""
<IPython.core.display.Image object>
==============================Agent graph==============================
          +-----------+             
          | __start__ |             
          +-----------+             
                 *                  
                 *                  
                 *                  
           +----------+             
           | llm_call |             
           +----------+             
          ...         ***           
         .               *          
       ..                 **        
+---------+           +-----------+ 
| __end__ |           | tool_node | 
+---------+           +-----------+ 
================================ Human Message =================================

Add 32 and 41. And then multiply the result by 2. And then divide the result by 4. Give me step-by-step thinking process.
================================== Ai Message ==================================

[{'text': "I'll solve this step-by-step using the available tools.\n\n**Step 1: Add 32 and 41**", 'type': 'text'}, {'id': 'call_function_5w2j48s9ivo4_1', 'input': {'a': 32, 'b': 41}, 'name': 'add', 'type': 'tool_use'}]
Tool Calls:
  add (call_function_5w2j48s9ivo4_1)
 Call ID: call_function_5w2j48s9ivo4_1
  Args:
    a: 32
    b: 41
================================= Tool Message =================================

73
================================== Ai Message ==================================

[{'text': '32 + 41 = **73**\n\n**Step 2: Multiply the result (73) by 2**', 'type': 'text'}, {'id': 'call_function_vjfo82e1sxsg_1', 'input': {'a': 73, 'b': 2}, 'name': 'multiply', 'type': 'tool_use'}]
Tool Calls:
  multiply (call_function_vjfo82e1sxsg_1)
 Call ID: call_function_vjfo82e1sxsg_1
  Args:
    a: 73
    b: 2
================================= Tool Message =================================

146
================================== Ai Message ==================================

[{'text': '73 × 2 = **146**\n\n**Step 3: Divide the result (146) by 4**', 'type': 'text'}, {'id': 'call_function_mthaoornixl5_1', 'input': {'a': 146, 'b': 4}, 'name': 'divide', 'type': 'tool_use'}]
Tool Calls:
  divide (call_function_mthaoornixl5_1)
 Call ID: call_function_mthaoornixl5_1
  Args:
    a: 146
    b: 4
================================= Tool Message =================================

36.5
================================== Ai Message ==================================

146 ÷ 4 = **36.5**

## Summary of the Step-by-Step Process:
1. **Addition:** 32 + 41 = 73
2. **Multiplication:** 73 × 2 = 146
3. **Division:** 146 ÷ 4 = 36.5

**Final Answer: 36.5**
"""
