import os
import subprocess

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

CASE_FILE = "case.md"

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Save interesting findings in case_notes. Act, don't explain."


# Tool definition
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }},
    {
        "name": "case_notes",
        "description": "Save a detective 'clue' to the case file.",
        "input_schema": {
            "type": "object",
            "properties": {"clue": {"type": "string"}},
            "required": ["clue"],
        },
    },
]

# Tool execution


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def save_clue(clue: str) -> str:
    print(f"\033[35m🔍 clue: {clue}\033[0m")  # magenta, like run_bash's yellow
    with open(CASE_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {clue}\n")
    return f"Clue saved to {CASE_FILE}"

# agent loop


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "bash":
                    print(
                        f"using tool {block.name}, command is {block.input['command']}.")
                    output = run_bash(block.input['command'])
                elif block.name == "case_notes":
                    print(
                        f"using tool {block.name}, saving clue {block.input['clue']}.")
                    output = save_clue(block.input['clue'])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": results})


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("s01: Agent Loop with clues")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
