import base64
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "refinement"))
import minimax_h3_image as h3img  # noqa: E402


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), rgb).save(buf, format="PNG")
    return buf.getvalue()


def _data_uri(width: int, height: int, rgb: tuple[int, int, int]) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(width, height, rgb)).decode("ascii")


class MiniMaxH3ImageTests(unittest.TestCase):
    def test_url_normalization_and_size_helpers(self):
        self.assertEqual(h3img.normalize_minimax_base_url("https://api.minimaxi.com"), "https://api.minimaxi.com/v1")
        self.assertEqual(h3img.normalize_minimax_base_url("https://api.minimax.io/v1/"), "https://api.minimax.io/v1")
        with self.assertRaises(ValueError):
            h3img.normalize_minimax_base_url("ftp://example.com")
        with self.assertRaises(ValueError):
            h3img.normalize_minimax_base_url("https://user:pass@api.minimax.io/v1")

        self.assertEqual(h3img.compute_target_size(1280, 720, "auto"), "auto")
        self.assertEqual(h3img.compute_target_size(1280, 720, "match"), "1280x704")
        self.assertEqual(h3img.compute_target_size(1536, 1024, "2K"), "2048x1376")
        self.assertEqual(h3img.compute_target_size(1024, 1536, "4K"), "2720x4096")

        w, h = h3img.compute_comfy_dimensions(1536, 1024, megapixels=4.0)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertAlmostEqual(w / h, 1536 / 1024, delta=0.05)

    def test_prompts_for_all_presets(self):
        for preset in ("faithful", "detailed", "match_3d_light"):
            p = h3img.build_style_redraw_prompt(preset, custom_prompt="sunny noon", protect_equipment=True)
            self.assertIn("<Picture 1>", p)
            self.assertIn("<Picture 2>", p)
            self.assertIn("sunny noon", p)
            self.assertIn("unobstructed", p)

    def test_minimax_api_b64_and_url_fallback_and_errors(self):
        captured: dict = {}
        out_png = _png_bytes(160, 128, (25, 140, 210))
        mode = {"kind": "b64"}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                captured["auth"] = self.headers.get("Authorization")
                captured["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
                if mode["kind"] == "b64":
                    self._send_json(
                        200,
                        {
                            "request_id": "req_123",
                            "data": [
                                {
                                    "b64_json": base64.b64encode(out_png).decode("ascii"),
                                    "revised_prompt": "expanded prompt text",
                                    "size": "160x128",
                                }
                            ],
                        },
                    )
                elif mode["kind"] == "url":
                    port = self.server.server_address[1]
                    self._send_json(
                        200,
                        {
                            "request_id": "req_url",
                            "data": [{"url": f"http://127.0.0.1:{port}/dl.png?sig=ok", "size": "160x128"}],
                        },
                    )
                elif mode["kind"] == "base_resp_err":
                    self._send_json(200, {"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}})
                else:
                    self._send_json(401, {"error": {"message": "unauthorized"}})

            def do_GET(self):
                if self.path.startswith("/dl.png"):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(out_png)))
                    self.end_headers()
                    self.wfile.write(out_png)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_json(self, code: int, payload: dict):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}/v1"
            shot = Image.new("RGB", (160, 128), (100, 100, 100))
            ref = Image.new("RGB", (192, 128), (200, 180, 150))

            img_bytes, meta = h3img.call_minimax_api(
                shot, ref, "test prompt", api_key="secret_tok", base_url=base_url, size="match"
            )
            self.assertEqual(img_bytes, out_png)
            self.assertEqual(captured["auth"], "Bearer secret_tok")
            self.assertEqual(captured["body"]["model"], "MiniMax-H3-Image")
            self.assertEqual(len(captured["body"]["images"]), 2)
            self.assertTrue(captured["body"]["images"][0]["image_url"].startswith("data:image/"))
            self.assertEqual(meta["revised_prompt"], "expanded prompt text")

            mode["kind"] = "url"
            img_bytes2, _ = h3img.call_minimax_api(shot, ref, "test prompt", api_key="secret_tok", base_url=base_url)
            self.assertEqual(img_bytes2, out_png)

            mode["kind"] = "base_resp_err"
            with self.assertRaises(RuntimeError) as ctx:
                h3img.call_minimax_api(shot, ref, "p", api_key="secret_tok", base_url=base_url)
            self.assertIn("invalid api key", str(ctx.exception))

            mode["kind"] = "http_err"
            with self.assertRaises(RuntimeError) as ctx2:
                h3img.call_minimax_api(shot, ref, "p", api_key="secret_tok", base_url=base_url)
            self.assertIn("unauthorized", str(ctx2.exception))
        finally:
            server.shutdown()
            server.server_close()

    def test_comfyui_graph_and_mocked_execution(self):
        graph = h3img.build_comfyui_minimax_h3_graph(
            "pic1.png", "pic2.png", "repaint prompt", width=1536, height=1024, ref_width=1536, ref_height=1024, seed=7
        )
        self.assertEqual(graph["30"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["32"]["class_type"], "MiniMaxH3ImageConditioning")
        self.assertEqual(graph["32"]["inputs"]["images.image_1"], ["48", 0])
        self.assertEqual(graph["32"]["inputs"]["images.image_2"], ["49", 0])
        self.assertEqual(graph["38"]["inputs"]["steps"], 6)

        repainted_png = _png_bytes(384, 256, (180, 120, 70))
        state = {"has_node": True, "uploaded": 0, "submitted_prompt": None}

        class ComfyHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path == "/system_stats":
                    self._send_json(200, {"system": {"comfyui_version": "0.46.0"}})
                elif self.path == "/object_info/MiniMaxH3ImageConditioning":
                    if state["has_node"]:
                        self._send_json(200, {"MiniMaxH3ImageConditioning": {"input": {}}})
                    else:
                        self._send_json(200, {})
                elif self.path == "/history/job-99":
                    self._send_json(
                        200,
                        {
                            "job-99": {
                                "status": {"status_str": "success", "completed": True},
                                "outputs": {"9": {"images": [{"filename": "out.png", "type": "output"}]}},
                            }
                        },
                    )
                elif self.path.startswith("/view"):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(repainted_png)))
                    self.end_headers()
                    self.wfile.write(repainted_png)
                else:
                    self._send_json(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if self.path == "/upload/image":
                    state["uploaded"] += 1
                    self._send_json(200, {"name": f"in_{state['uploaded']}.png", "subfolder": ""})
                elif self.path == "/prompt":
                    state["submitted_prompt"] = json.loads(raw.decode("utf-8"))["prompt"]
                    self._send_json(200, {"prompt_id": "job-99"})
                else:
                    self._send_json(404, {})

            def _send_json(self, code: int, payload: dict):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ComfyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            ping_ok = h3img.ping_comfyui(url)
            self.assertTrue(ping_ok["ok"])
            self.assertEqual(ping_ok["comfyui_version"], "0.46.0")
            self.assertTrue(ping_ok["has_minimax_h3_image"])

            shot = Image.new("RGB", (384, 256), (90, 90, 90))
            ref = Image.new("RGB", (384, 256), (120, 100, 80))
            with tempfile.TemporaryDirectory() as tmp:
                out_bytes, meta = h3img.call_comfyui_minimax_h3(
                    shot, ref, "comfy prompt", comfy_url=url, megapixels=1.0, task_id="t1", directory=Path(tmp)
                )
                self.assertEqual(out_bytes, repainted_png)
                self.assertEqual(state["uploaded"], 2)
                self.assertEqual(meta["provider"], "comfyui")
                self.assertEqual(json.loads((Path(tmp) / "comfy-task.json").read_text("utf-8"))["prompt_id"], "job-99")

            state["has_node"] = False
            with self.assertRaises(RuntimeError) as ctx:
                h3img.call_comfyui_minimax_h3(shot, ref, "p", comfy_url=url)
            self.assertIn("MiniMaxH3ImageConditioning", str(ctx.exception))
        finally:
            server.shutdown()
            server.server_close()

    def test_run_task_equipment_protection_and_full_frame(self):
        # Mock MiniMax server that records which picture 1 color it received and returns solid blue.
        observed: list[tuple[int, int, int]] = []
        repainted_bytes = _png_bytes(320, 256, (20, 90, 200))

        class ApiHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                img1_uri = body["images"][0]["image_url"]
                b64 = img1_uri.split(",", 1)[1]
                with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
                    observed.append(im.getpixel((5, 5)))
                raw = json.dumps({"data": [{"b64_json": base64.b64encode(repainted_bytes).decode("ascii")}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            api_base = f"http://127.0.0.1:{server.server_port}/v1"
            # Source has red equipment box (50..110, 40..90) over grey background (120).
            src_arr = np.full((128, 160, 3), 120, dtype=np.uint8)
            src_arr[40:90, 50:110] = (230, 40, 30)
            src_buf = io.BytesIO()
            Image.fromarray(src_arr).save(src_buf, format="PNG")
            src_uri = "data:image/png;base64," + base64.b64encode(src_buf.getvalue()).decode("ascii")

            env_uri = _data_uri(160, 128, (110, 110, 110))
            mask_arr = np.zeros((128, 160, 3), dtype=np.uint8)
            mask_arr[40:90, 50:110] = 255
            mask_buf = io.BytesIO()
            Image.fromarray(mask_arr).save(mask_buf, format="PNG")
            mask_uri = "data:image/png;base64," + base64.b64encode(mask_buf.getvalue()).decode("ascii")

            with patch.dict(os.environ, {"MINIMAX_RUNTIME_API_KEY": "runtime-secret"}):
                with tempfile.TemporaryDirectory() as tmpdir:
                    d1 = Path(tmpdir) / "redraw_protected"
                    d1.mkdir()
                    (d1 / "input.json").write_text(
                        json.dumps(
                            {
                                "provider": "minimax_api",
                                "api_base_url": api_base,
                                "source_image": src_uri,
                                "environment_image": env_uri,
                                "protection_mask": mask_uri,
                                "reference_image": _data_uri(160, 128, (50, 60, 70)),
                                "protect_equipment": True,
                                "style_preset": "faithful",
                            }
                        ),
                        encoding="utf-8",
                    )
                    meta1 = h3img.run_task(d1)
                    self.assertTrue(meta1["equipment_protected"])
                    self.assertGreater(meta1["protection_coverage_pct"], 5.0)
                    # Model received environment image (110,110,110), not source.
                    self.assertEqual(observed[0], (110, 110, 110))
                    with Image.open(d1 / "result.png") as final_img:
                        self.assertEqual(final_img.size, (160, 128))
                        # Inside protected mask -> exact source red pixel
                        self.assertEqual(final_img.getpixel((80, 65)), (230, 40, 30))
                        # Outside protected mask -> repainted blue pixel
                        self.assertEqual(final_img.getpixel((10, 10)), (20, 90, 200))
                    meta_text = (d1 / "result.meta.json").read_text("utf-8")
                    self.assertNotIn("runtime-secret", meta_text)

                # Now verify empty mask works cleanly at full model resolution without raising.
                with tempfile.TemporaryDirectory() as tmpdir2:
                    d2 = Path(tmpdir2) / "redraw_full"
                    d2.mkdir()
                    (d2 / "input.json").write_text(
                        json.dumps(
                            {
                                "provider": "minimax_api",
                                "api_base_url": api_base,
                                "source_image": src_uri,
                                "environment_image": env_uri,
                                "protection_mask": _data_uri(160, 128, (0, 0, 0)),
                                "reference_image": _data_uri(160, 128, (50, 60, 70)),
                                "protect_equipment": True,
                            }
                        ),
                        encoding="utf-8",
                    )
                    meta2 = h3img.run_task(d2)
                    self.assertFalse(meta2["equipment_protected"])
                    self.assertEqual(meta2["output_size"], [320, 256])
        finally:
            server.shutdown()
            server.server_close()

    def test_comfy_gateway_status_wait_and_hot_takeover(self):
        state = {"gw": "reconnecting", "upstream": None, "calls": []}

        class GwHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                state["calls"].append(self.path)
                if self.path == "/_gateway/status":
                    self._json(
                        200,
                        {
                            "state": state["gw"],
                            "recovery_stage": "tier2_wake" if state["gw"] != "ready" else "connected",
                            "upstream_url": state["upstream"],
                            "last_error": "boom" if state["gw"] == "error" else "",
                            "active_requests": 0,
                            "cooldown_remaining_sec": 0.0,
                        },
                    )
                elif self.path == "/system_stats":
                    self._json(200, {"system": {"comfyui_version": "0.48.0"}})
                elif self.path == "/object_info/MiniMaxH3ImageConditioning":
                    self._json(200, {"MiniMaxH3ImageConditioning": {"input": {}}})
                elif self.path == "/object_info/MiniMaxH3ImageToVideo":
                    self._json(200, {"MiniMaxH3ImageToVideo": {"input": {}}})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                if self.path == "/_gateway/set-upstream":
                    ln = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(ln).decode("utf-8"))
                    up = payload.get("url") or payload.get("upstream_url")
                    state["upstream"] = up
                    state["gw"] = "ready"
                    self._json(200, {"ok": True, "upstream_url": up})
                else:
                    self._json(404, {"error": "not found"})

            def _json(self, code: int, body: dict):
                raw = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), GwHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            gw_url = f"http://127.0.0.1:{srv.server_port}"
            p_recon = h3img.ping_comfyui(gw_url)
            self.assertFalse(p_recon["ok"])
            self.assertEqual(p_recon["gateway_state"], "reconnecting")
            self.assertNotIn("/system_stats", state["calls"])

            res = h3img.set_comfy_gateway_upstream(gw_url, "https://live-tunnel.trycloudflare.com")
            self.assertTrue(res["ok"])
            self.assertEqual(state["upstream"], "https://live-tunnel.trycloudflare.com")

            ready_gw = h3img.wait_for_comfy_gateway_ready(gw_url, wait_timeout_sec=5.0)
            self.assertEqual(ready_gw["state"], "ready")

            p_ok = h3img.ping_comfyui(gw_url)
            self.assertTrue(p_ok["ok"])
            self.assertTrue(p_ok["has_minimax_h3_image"])
            self.assertTrue(p_ok["has_minimax_h3_video"])
            self.assertEqual(p_ok["gateway_state"], "ready")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
