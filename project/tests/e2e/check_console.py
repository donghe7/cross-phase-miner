import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("CP_SERVER_PYTHON", sys.executable)
with socket.socket() as port_probe:
    port_probe.bind(("127.0.0.1", 0))
    PORT = port_probe.getsockname()[1]
BASE = f"http://127.0.0.1:{PORT}"


def api(path, data=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def main():
    (ROOT / "test-results").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cp-fleet-ui-") as folder:
        server = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "server.app",
                "--port",
                str(PORT),
                "--speed",
                "1",
                "--db",
                folder + "/test.sqlite3",
            ],
            cwd=ROOT,
            env={
                k: v
                for k, v in os.environ.items()
                if not k.startswith("CP_") and k != "DATABASE_URL"
            },
            stdout=subprocess.DEVNULL,
        )
        try:
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError("Test server exited before readiness")
                try:
                    now = api("/v1/clock")["sim_now"]
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Test server readiness timed out")
            for i in range(1, 21):
                api(
                    "/v1/observations",
                    dict(
                        robot_id=f"robot_{i:03}",
                        intersection_id="seongsu_station",
                        mode="scout",
                        action="WAIT",
                        observations=[dict(t=now, color="RED", conf=0.95)],
                    ),
                )
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1600, "height": 1200})
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(BASE)
                page.wait_for_function(
                    "state.snapshot?.robots.length===20 && document.querySelector('#robot-picker').children.length===20"
                )
                assert page.locator("#fleet tr:visible").count() == 5
                assert "1 / 4 页" in page.locator("#fleet-page").inner_text()
                page.locator("#fleet-next").click()
                assert "6–10 / 20" in page.locator("#fleet-page").inner_text()
                page.get_by_role("button", name="定位 robot_020", exact=True).click()
                assert "第 4 / 4 页" in page.locator("#fleet-page").inner_text()
                assert page.locator("#fleet-next").is_disabled()
                assert page.get_by_role("button", name="跟随 robot_020", exact=True).is_visible()
                page.wait_for_function(
                    "document.querySelector('#track-content')?.textContent.includes('R020')"
                )
                sizes = page.evaluate("""() => {
                  const bounds=()=>({table:document.querySelector('.table-scroll').getBoundingClientRect().height,
                    tracking:document.querySelector('#tracking').getBoundingClientRect().height,
                    columns:[...document.querySelectorAll('.fleet-table th')].map(e=>e.getBoundingClientRect().width),
                    rows:[...document.querySelectorAll('#fleet tr:not([hidden])')].map(e=>e.getBoundingClientRect().height)});
                  const before=bounds(); const r=state.snapshot.robots.find(r=>r.robot_id==='robot_020');
                  Object.assign(r,{action:'TRAVEL',intersection_id:null,origin_id:'seongsu_station',destination_id:'seongsuiero',
                    travel_started_at:state.snapshot.sim_now,travel_arrives_at:state.snapshot.sim_now+100,observation:null});
                  state.byId.get('seongsuiero').name='Very long intersection name '.repeat(12); paint();
                  const during=bounds(); r.action='WAIT'; r.intersection_id='seongsuiero'; paint();
                  return [before,during,bounds()];
                }""")
                assert sizes[0] == sizes[1] == sizes[2], sizes
                page.screenshot(path=str(ROOT / "test-results/fleet-desktop.png"))
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                assert page.locator("#robot-picker button").count() == 20
                assert page.locator(".fleet-mobile-note").is_visible()
                assert page.evaluate("""() => {
                    const frame=document.querySelector('.table-scroll');
                    const last=[...document.querySelectorAll('#fleet tr:not([hidden])')].at(-1);
                    return last.getBoundingClientRect().bottom <= frame.getBoundingClientRect().bottom;
                }""")
                page.locator(".fleet-panel").screenshot(
                    path=str(ROOT / "test-results/fleet-mobile.png")
                )
                page.locator("#fleet-prev").click()
                assert "第 3 / 4 页" in page.locator("#fleet-page").inner_text()
                page.get_by_role("button", name="定位 robot_001", exact=True).click()
                assert "第 1 / 4 页" in page.locator("#fleet-page").inner_text()
                assert page.locator("#speed-options button").count() == 5
                assert not errors, errors
                print(
                    "PASS: all 20 robot shortcuts, explicit 4-page navigation, follow jumps to robot page, stable table/column/row/tracking dimensions across long routes and state changes, mobile horizontal scrolling without clipped fifth row, five speeds, no JS errors"
                )
                print(json.dumps(sizes[0]))
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=8)


if __name__ == "__main__":
    main()
