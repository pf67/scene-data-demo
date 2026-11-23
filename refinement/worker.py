"""Qwen 2.1 ComfyUI adapter; only the environment is sent to the image model."""
import json
import sys
from pathlib import Path

from protection import decode_image, prepare_protection, composite_protected
from qwen import generate


def run(directory: Path):
    params = json.loads((directory / 'input.json').read_text('utf-8'))
    original, mask = prepare_protection(params['base64_image'], params['protection_mask'], directory)
    environment = decode_image(params['environment_image'])
    if environment.size != original.size:
        raise ValueError('Environment and original image dimensions must match')
    output = directory / 'result.png'
    output.write_bytes(generate(environment, params['prompt'], directory, params.get('steps', 40), params.get('cfg_scale', 1)))
    composite_protected(original, mask, output)


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]))
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
