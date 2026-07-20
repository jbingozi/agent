import argparse
import os
from utilss.config_handler import load_env_file
from utilss.path_tool import get_abs_path


ENV_KEYS = ["MILVUS_URI", "MILVUS_TOKEN", "AMAP_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"]


def mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 14:
        return value[:3] + "..."
    return value[:8] + "..." + value[-6:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent 智能客服启动入口")
    parser.add_argument("query", nargs="?", help="单轮输入问题")
    parser.add_argument("--session-id", dest="session_id", default=None, help="会话 ID")
    parser.add_argument("--user-id", dest="user_id", default=None, help="用户 ID")
    parser.add_argument("--check-env", action="store_true", help="Print the .env path and masked env values")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI web server")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    return parser


def check_env() -> None:
    env_path = get_abs_path(".env")
    load_env_file(env_path)
    print(f"ENV_FILE={env_path}")
    print(f"ENV_FILE_EXISTS={os.path.exists(env_path)}")
    for key in ENV_KEYS:
        print(f"{key}={mask_secret(os.getenv(key, ''))}")


def run_once(agent, query: str, session_id: str | None, user_id: str | None) -> None:
    for chunk in agent.execute_stream(query, session_id=session_id, user_id=user_id):
        print(chunk, end="", flush=True)


def interactive_loop(agent) -> None:
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
    load_env_file(get_abs_path(".env"))

    if args.check_env:
        check_env()
        return 0

    if args.serve:
        import uvicorn

        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
        return 0

    from agent.react_agent import ReactAgent

    agent = ReactAgent()

    if args.query:
        run_once(agent, args.query, args.session_id, args.user_id)
        return 0

    interactive_loop(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
