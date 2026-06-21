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
