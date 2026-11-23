import http.server
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/scene_pipeline"))
import minimax_h3_orbit as h3  # noqa: E402


class MiniMaxH3OrbitTests(unittest.TestCase):
    def test_snap_length_and_dimension(self):
        self.assertEqual(h3.snap_length(5), 5)
        self.assertEqual(h3.snap_length(120), 124)
        self.assertEqual(h3.snap_length(124), 124)
        self.assertEqual(h3.snap_length(125), 141)
        self.assertEqual(h3.snap_length(243), 243)
        self.assertEqual(h3.snap_length(250), 260)
        for v in (5, 124, 243, 362):
            self.assertEqual((h3.snap_length(v) - 5) % 17, 0)
        self.assertEqual(h3.snap_dimension(1330), 1344)
        self.assertEqual(h3.snap_dimension(768), 768)
        self.assertEqual(h3.snap_dimension(16), 32)

    def test_clean_watermark_modifies_only_corner(self):
        arr = np.full((100, 200, 3), 120, dtype=np.uint8)
        arr[90:98, 175:198] = 250
        src = Image.fromarray(arr)
        cleaned = h3.clean_watermark(src)
        self.assertEqual(cleaned.size, src.size)
        out = np.asarray(cleaned)
        np.testing.assert_array_equal(out[:88, :], arr[:88, :])
        np.testing.assert_array_equal(out[:, :172], arr[:, :172])
        self.assertLess(out[92:96, 180:194].mean(), 200)

    def test_build_workflow_modes(self):
        loop = h3.build_workflow(first_image_name="a.png", mode="loop360", length=240, turbo=True)
        h3_inputs = loop["131"]["inputs"]
        self.assertEqual(h3_inputs["first_frame"], ["101", 0])
        self.assertEqual(h3_inputs["last_frame"], ["101", 0])
        self.assertEqual(h3_inputs["length"], 243)
        self.assertIn("134", loop)
        self.assertEqual(loop["126"]["inputs"]["model"], ["134", 0])
        self.assertEqual(loop["124"]["inputs"]["steps"], 8)

        arc = h3.build_workflow(
            first_image_name="a.png",
            last_image_name="b.png",
            mode="arc",
            turbo=False,
            steps=20,
        )
        self.assertNotIn("134", arc)
        self.assertEqual(arc["131"]["inputs"]["first_frame"], ["101", 0])
        self.assertEqual(arc["131"]["inputs"]["last_frame"], ["102", 0])
        self.assertEqual(arc["126"]["inputs"]["model"], ["127", 0])
        self.assertEqual(arc["124"]["inputs"]["steps"], 20)

        single_arc = h3.build_workflow(first_image_name="a.png", mode="arc")
        self.assertNotIn("last_frame", single_arc["131"]["inputs"])

    def test_end_to_end_mocked_comfy_server(self):
        captured = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if self.path == "/upload/image":
                    captured["upload_bytes"] = len(body)
                    self._json(200, {"name": "uploaded_first.png", "subfolder": ""})
                elif self.path == "/prompt":
                    captured["prompt"] = json.loads(body.decode("utf-8"))["prompt"]
                    self._json(200, {"prompt_id": "job-123"})
                else:
                    self._json(404, {"error": "not found"})

            def do_GET(self):
                if self.path == "/history/job-123":
                    self._json(
                        200,
                        {
                            "job-123": {
                                "status": {"status_str": "success", "completed": True},
                                "outputs": {
                                    "92": {
                                        "images": [
                                            {
                                                "filename": "orbit_00001_.mp4",
                                                "subfolder": "video",
                                                "type": "output",
                                            }
                                        ]
                                    }
                                },
                            }
                        },
                    )
                elif self.path.startswith("/view?"):
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    payload = b"ftypmp42mockvideodata"
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self._json(404, {})

            def _json(self, code, data):
                raw = json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                src_img = tmp_path / "ref.png"
                Image.new("RGB", (64, 64), (100, 140, 90)).save(src_img)
                out_dir = tmp_path / "out"
                args = h3.parse_args(
                    [
                        "--comfy-url",
                        f"http://127.0.0.1:{port}",
                        "--first-frame",
                        str(src_img),
                        "--mode",
                        "loop360",
                        "--length",
                        "124",
                        "--poll-interval",
                        "0.05",
                        "--out-dir",
                        str(out_dir),
                    ]
                )
                result = h3.run_orbit_generation(args)
                out_mp4 = Path(result["videoPath"])
                self.assertTrue(out_mp4.is_file())
                self.assertEqual(out_mp4.read_bytes(), b"ftypmp42mockvideodata")
                self.assertEqual(captured["prompt"]["131"]["inputs"]["first_frame"], ["101", 0])
                self.assertEqual(captured["prompt"]["131"]["inputs"]["last_frame"], ["101", 0])
                self.assertTrue((out_dir / "h3_loop360_cw_seed42.manifest.json").is_file())
        finally:
            server.shutdown()
            server.server_close()

    def test_check_comfy_health(self):
        class HealthHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path == "/system_stats":
                    payload = {"system": {"comfyui_version": "0.46.0"}, "devices": [{"name": "A100"}]}
                elif self.path == "/object_info/MiniMaxH3ImageToVideo":
                    payload = {"MiniMaxH3ImageToVideo": {"input": {}}}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            ok = h3.check_comfy_health(f"http://127.0.0.1:{port}", timeout_sec=3.0)
            self.assertTrue(ok["ok"])
            self.assertEqual(ok["comfyuiVersion"], "0.46.0")
            self.assertTrue(ok["hasMiniMaxH3Node"])
        finally:
            server.shutdown()
            server.server_close()

        bad = h3.check_comfy_health(f"http://127.0.0.1:{port}", timeout_sec=1.0)
        self.assertFalse(bad["ok"])


if __name__ == "__main__":
    unittest.main()
