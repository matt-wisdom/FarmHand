#!/usr/bin/env python3
"""
================================================================================
🌱 FarmHand AI - Unified Turn-Key Runner
================================================================================
Automated all-in-one launcher:
1. Installs/verifies dependencies from backend/requirements.txt
2. Verifies and downloads required GGUF weights via download_model.py
3. Starts the FastAPI backend server (uvicorn main:app on port 8000)
4. Waits for healthcheck and automatically launches the web UI in browser
5. Handles clean shutdown on Ctrl+C
================================================================================
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
REQUIREMENTS_FILE = BACKEND_DIR / "requirements.txt"
DOWNLOAD_SCRIPT = ROOT_DIR / "download_model.py"


def print_banner() -> None:
    print(
        """
================================================================================
 🌱 FarmHand AI - Unified Turn-Key Runner
 Offline-First Multilingual Agricultural Intelligence System
================================================================================
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FarmHand AI: Automated All-in-One Launcher"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server (default: 8000)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip dependency installation step",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download check step",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open web browser on startup",
    )
    return parser.parse_args()


def install_dependencies() -> None:
    if not REQUIREMENTS_FILE.exists():
        print(f"[-] Warning: Requirements file not found at {REQUIREMENTS_FILE}")
        return

    print(f"[+] [1/4] Checking and installing dependencies from {REQUIREMENTS_FILE}...")

    # Use uv if installed for lightning-fast installation
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
        print("[✓] Dependencies installed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(
            f"[-] ERROR: Dependency installation failed with code {e.returncode}.",
            file=sys.stderr,
        )
        sys.exit(e.returncode)


def verify_or_download_model() -> None:
    print("[+] [2/4] Verifying model weights and assets...")
    if not DOWNLOAD_SCRIPT.exists():
        print(f"[-] Warning: Download script not found at {DOWNLOAD_SCRIPT}")
        return

    cmd = [sys.executable, str(DOWNLOAD_SCRIPT)]
    try:
        subprocess.run(cmd, check=True)  # noqa: S603
        print("[✓] Model assets verified.\n")
    except subprocess.CalledProcessError as e:
        print(
            f"[-] ERROR: Model asset download failed with code {e.returncode}.",
            file=sys.stderr,
        )
        sys.exit(e.returncode)


def wait_for_server(url: str, timeout: float = 45.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(  # noqa: S310
                url, headers={"User-Agent": "FarmHand-HealthCheck/1.0"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:  # noqa: S310
                if resp.status in (200, 307, 308):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    print_banner()
    args = parse_args()

    # Step 1: Dependencies
    if not args.skip_install:
        install_dependencies()
    else:
        print("[*] Skipping dependency installation (--skip-install).")

    # Step 2: Model download
    if not args.skip_download:
        verify_or_download_model()
    else:
        print("[*] Skipping model download check (--skip-download).")

    # Step 3: Launch FastAPI server
    print(
        f"[+] [3/4] Starting FarmHand AI backend server on http://{args.host}:{args.port}..."
    )
    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--app-dir",
        str(BACKEND_DIR),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    server_process = subprocess.Popen(server_cmd)  # noqa: S603

    def signal_handler(sig, frame):
        print("\n[*] Stopping FarmHand AI server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("[✓] Server stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Step 4: Wait for readiness & open browser
    health_url = f"http://{args.host}:{args.port}/api/health"
    app_url = (
        f"http://localhost:{args.port}"
        if args.host in ("127.0.0.1", "0.0.0.0")  # noqa: S104
        else f"http://{args.host}:{args.port}"
    )

    print(f"[+] [4/4] Waiting for backend readiness at {health_url}...")
    if wait_for_server(health_url, timeout=30.0):
        print(f"\n[✓] FarmHand AI is LIVE at {app_url}")
        if not args.no_browser:
            print(f"[+] Opening {app_url} in your default web browser...")
            webbrowser.open(app_url)
        else:
            print(f"[*] Access the web application at: {app_url}")
    else:
        print(
            f"[-] Warning: Server took longer than expected to report healthy. You can still access it at {app_url}"
        )

    print(
        "\n[i] FarmHand AI is running. Press Ctrl+C at any time to stop the server.\n"
    )

    try:
        server_process.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
