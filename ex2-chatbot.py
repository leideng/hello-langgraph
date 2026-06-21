"""Simple multi-turn Q&A chatbot (one session at a time)."""

from langchain_anthropic import ChatAnthropic
from langchain.messages import SystemMessage, HumanMessage

model = ChatAnthropic(model="MiniMax-M3")
SYSTEM = SystemMessage(content="You are a helpful assistant.")

NEW_SESSION = {"/new", "/reset", "new"}
QUIT = {"/quit", "/exit", "quit", "exit"}


def main():
    messages = [SYSTEM]
    print("Chatbot ready. Commands: /new (new session), /quit (exit)")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in QUIT:
            print("Bye!")
            break
        if user_input.lower() in NEW_SESSION:
            messages = [SYSTEM]
            print("New session started.")
            continue

        messages.append(HumanMessage(content=user_input))
        response = model.invoke(messages)
        messages.append(response)
        print(f"\nAssistant: {response.content}")


if __name__ == "__main__":
    main()


"""
Chatbot ready. Commands: /new (new session), /quit (exit)

You: who are you

Assistant: I'm MiniMax-M3, an AI assistant developed by MiniMax. I'm here to help you with questions, tasks, conversations, and a wide range of topics. How can I assist you today?

You: introdue intel for me; how's current techinque advantages over TSMC

Assistant: # Intel Corporation Overview

**Intel** is one of the world's largest semiconductor companies, founded in 1968 and headquartered in Santa Clara, California. Historically, Intel dominated the CPU market through its x86 architecture and in-house manufacturing (IDM model), but in recent years it has faced intense competition and undergone significant transformation.
...
"""