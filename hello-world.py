from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print("==="*10 + "result" + "==="*10)
print(result)
print("==="*10 + "result['messages']" + "==="*10)
print(result["messages"])
print("==="*10 + "result['messages'][0]" + "==="*10)
print(result["messages"][0])
print("==="*10 + "result['messages'][1]" + "==="*10)
print(result["messages"][1])

"""
==============================result==============================
{'messages': [HumanMessage(content='hi!', additional_kwargs={}, response_metadata={}, id='7cab4a8c-4884-49ba-b095-5bb07de23f07'), AIMessage(content='hello world', additional_kwargs={}, response_metadata={}, id='430dab98-1b48-4dbc-91e3-e8506e5f78b5', tool_calls=[], invalid_tool_calls=[])]}
==============================result['messages']==============================
[HumanMessage(content='hi!', additional_kwargs={}, response_metadata={}, id='7cab4a8c-4884-49ba-b095-5bb07de23f07'), AIMessage(content='hello world', additional_kwargs={}, response_metadata={}, id='430dab98-1b48-4dbc-91e3-e8506e5f78b5', tool_calls=[], invalid_tool_calls=[])]
==============================result['messages'][0]==============================
content='hi!' additional_kwargs={} response_metadata={} id='7cab4a8c-4884-49ba-b095-5bb07de23f07'
==============================result['messages'][1]==============================
content='hello world' additional_kwargs={} response_metadata={} id='430dab98-1b48-4dbc-91e3-e8506e5f78b5' tool_calls=[] invalid_tool_calls=[]
"""
