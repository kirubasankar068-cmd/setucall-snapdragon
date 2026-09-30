"""
SetuCall Executable Entry Point Script.

Usage:
  Interactive Mode: python setucall.py --mode=host | join
  Command Mode:
    setucall connect <peer-ip>
    setucall call
    setucall msg "Hello Snapdragon!"
    setucall quit
"""

import sys
import argparse
import asyncio
from cli import SetuCallCLI


def main():
    parser = argparse.ArgumentParser(
        description="SetuCall CLI: Offline P2P Video/Audio Calling & Encrypted Text Chat"
    )
    parser.add_argument(
        "cmd_args",
        nargs="*",
        help="Command and arguments: connect <ip>, call, msg <text>, quit",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["host", "join"],
        default="host",
        help="Operating mode: 'host' or 'join'",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Target host IP address for connect command",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Signaling port (default: 8080)",
    )
    parser.add_argument(
        "--npu-mode",
        type=str,
        choices=["npu", "cpu", "auto"],
        default="auto",
        help="Execution provider choice",
    )
    parser.add_argument(
        "--dummy-media",
        action="store_true",
        default=True,
        help="Use synthetic test media",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=True,
        help="Disable OpenCV GUI popups",
    )

    args = parser.parse_args()

    # Determine command passed via positional arguments (e.g. setucall msg "Hello")
    if args.cmd_args:
        first_arg = args.cmd_args[0].lower()
        if first_arg == "connect":
            args.mode = "join"

    cli = SetuCallCLI(
        mode=args.mode,
        port=args.port,
        dummy_media=args.dummy_media,
        no_gui=args.no_gui,
        npu_mode=args.npu_mode,
    )

    if args.cmd_args:
        # Single-command scriptable mode
        cmd_str = " ".join(args.cmd_args)
        async def run_single_command():
            await cli.initialize()
            await cli.execute_command(cmd_str)
            await cli.do_quit()
        asyncio.run(run_single_command())
    else:
        # Interactive REPL shell mode
        asyncio.run(cli.run_interactive_repl())


if __name__ == "__main__":
    main()
