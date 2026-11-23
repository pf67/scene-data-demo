"""Exercise the independent worker against a local fake ComfyUI, without weights."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image


def png(image):
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


class WorkerTest(unittest.TestCase):
    def test_environment_request_and_exact_equipment_composite(self):
        generated = png(Image.new('RGB', (32, 32), 'blue'))
        submitted = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, image=False):
                data = value if image else json.dumps(value).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                if self.path == '/upload/image':
                    self.reply({'name': 'environment.png', 'subfolder': ''})
                else:
                    submitted.append(json.loads(raw)['prompt'])
                    self.reply({'prompt_id': 'test'})

            def do_GET(self):
                if self.path.startswith('/history/'):
                    self.reply({'test': {'outputs': {'9': {'images': [{'filename': 'result.png', 'type': 'output', 'subfolder': ''}]}}}})
                else:
                    self.reply(generated, image=True)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix='scene-refine-test-') as folder:
                directory = Path(folder)
                source = Image.new('RGB', (32, 32), 'red')
                mask = Image.new('L', (32, 32))
                mask.putpixel((16, 16), 255)
                encode = lambda im: base64.b64encode(png(im)).decode()
                params = {'base64_image': encode(source), 'protection_mask': encode(mask), 'environment_image': encode(Image.new('RGB', (32, 32), 'green')), 'prompt': 'Environment only', 'steps': 40, 'cfg_scale': 1}
                (directory / 'input.json').write_text(json.dumps(params), 'utf-8')
                worker = Path(__file__).resolve().parents[1] / 'refinement/worker.py'
                subprocess.run([sys.executable, str(worker), folder], env={**os.environ, 'COMFYUI_URL': f'http://127.0.0.1:{server.server_port}'}, check=True, timeout=30, capture_output=True)
                with Image.open(directory / 'result.png') as result:
                    self.assertEqual(result.getpixel((16, 16)), (255, 0, 0))
                    self.assertEqual(result.getpixel((18, 18)), (255, 0, 0))
                    self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
                self.assertEqual(submitted[0]['1']['inputs']['image'], 'environment.png')
                self.assertEqual(submitted[0]['6']['inputs']['prompt'], 'Environment only')
                self.assertEqual(submitted[0]['7']['inputs']['denoise'], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
