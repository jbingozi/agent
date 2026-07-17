import argparse
import sys

from agent.react_agent import ReactAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent 智能客服启动入口")
    parser.add_argument("query", nargs="?", help="单轮输入问题")
    parser.add_argument("--session-id", dest="session_id", default=None, help="会话 ID")
    parser.add_argument("--user-id", dest="user_id", default=None, help="用户 ID")
    return parser


def run_once(agent: ReactAgent, query: str, session_id: str | None, user_id: str | None) -> None:
    for chunk in agent.execute_stream(query, session_id=session_id, user_id=user_id):
        print(chunk, end="", flush=True)


def interactive_loop(agent: ReactAgent) -> None:
    print("Enter /exit to quit.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"/exit", "exit", "quit"}:
            break

        run_once(agent, query, None, None)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    agent = ReactAgent()

    if args.query:
        run_once(agent, args.query, args.session_id, args.user_id)
        return 0

    if sys.stdin.isatty():
        interactive_loop(agent)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
