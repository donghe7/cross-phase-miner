"""Real HTTP delivery across server restart, with a lost response and stable report ID."""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from server.fleet import ServerClient
from server.outbox import ArrivalOutbox

ROOT = Path(__file__).resolve().parents[2]


def main():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="cp-delivery-") as folder:
        db = str(Path(folder) / "server.sqlite3")
        journal = str(Path(folder) / "outbox.sqlite3")
        client = ServerClient(f"http://127.0.0.1:{port}", timeout=1)
        env = {
            k: v for k, v in os.environ.items() if not k.startswith("CP_") and k != "DATABASE_URL"
        }
        processes = []
        outboxes = []

        def start():
            process = subprocess.Popen(
                [
                    os.environ.get("CP_SERVER_PYTHON", sys.executable),
                    "-m",
                    "server.app",
                    "--port",
                    str(port),
                    "--speed",
                    "1",
                    "--intersections",
                    "12",
                    "--zone",
                    "8",
                    "--hubs",
                    "2",
                    "--db",
                    db,
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
            )
            processes.append(process)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Test server exited")
                try:
                    return process, client.get("/v1/clock")["sim_now"]
                except OSError:
                    time.sleep(0.05)
            raise TimeoutError("Test server readiness")

        def stop(process):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise

        class LoseResponse:
            def post(self, path, payload):
                client.post(path, payload)
                raise OSError("simulated HTTP response loss after server commit")

        try:
            server, now = start()
            report = dict(
                record_id="http-recovery",
                robot_id="robot_001",
                intersection_id="seongsu_station",
                arrival_time=now - 120,
                depart_time=now,
                waited=120,
                mode="scout",
                transitions=[
                    dict(
                        timestamp=now,
                        from_color="RED",
                        to_color="GREEN",
                        episode_start=now - 100,
                        exact_red=True,
                    )
                ],
            )
            first = ArrivalOutbox(journal, "http-test", LoseResponse(), False)
            outboxes.append(first)
            first.enqueue(report)
            first.deliver_once()
            assert first.pending_count() == 1
            stop(server)
            first.close()
            outboxes.remove(first)
            # Restart both endpoints; the robot still holds exactly the same record ID.
            second = ArrivalOutbox(journal, "http-test", client, False)
            outboxes.append(second)
            assert second.pending_count() == 1
            server, _ = start()
            second.deliver_once()
            assert second.pending_count() == 0
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                detail = client.get("/v1/intersections/seongsu_station")
                if detail["learning"]["samples"] == 1:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("Durable learning did not resume")
            assert detail["learning"]["visits"]["visits"] == 1
            print(
                "PASS: HTTP committed ack, lost response, robot/server restart, one visit and one learned sample"
            )
        finally:
            for outbox in outboxes:
                outbox.close()
            for process in processes:
                stop(process)


if __name__ == "__main__":
    main()
