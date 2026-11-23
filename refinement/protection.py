"""Keep protected source pixels exact after an image model edits the environment."""
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageFilter


def decode_image(value: str) -> Image.Image:
    encoded = value.split(',', 1)[1] if value.startswith('data:') else value
    data = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 16777216:
            raise ValueError('Scene snapshot must not exceed 16 megapixels')
        image.load()
        return image.convert('RGB')


def prepare_protection(source: str, mask: str, directory: Path):
    original = decode_image(source)
    protection = decode_image(mask).convert('L')
    if protection.size != original.size:
        raise ValueError('Protection mask and source dimensions must match')
    # Include anti-aliased silhouette pixels and a two-pixel guard outside them.
    protection = protection.point(lambda v: 255 if v > 0 else 0)
    if not protection.getbbox():
        raise ValueError('No visible equipment in the protection mask')
    protection = protection.filter(ImageFilter.MaxFilter(5))
    directory.mkdir(parents=True, exist_ok=True)
    original.save(directory / 'source.png')
    protection.save(directory / 'protection.png')
    return original, protection


def composite_protected(original: Image.Image, mask: Image.Image, output: Path):
    with Image.open(output) as image:
        edited = image.convert('RGB')
    if edited.size != original.size:
        # Never resize/crop the source or its equipment. Normalize generated background only.
        # Qwen 2.1 rounds each reference dimension to a multiple of 32.
        # Permit that small relative aspect change while rejecting reframing/cropping.
        if abs((edited.width / edited.height) / (original.width / original.height) - 1) > 0.05:
            raise ValueError('Model changed the aspect ratio; configure same-ratio output')
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    edited.save(output.with_name(output.stem + '-model.png'))
    result = Image.composite(original, edited, mask)
    result.save(output, format='PNG')
    output.with_suffix('.protection.json').write_text(json.dumps({
        'mode': 'exact_source_pixels', 'width': original.width, 'height': original.height,
        'guard_pixels': 2, 'source': 'source.png', 'mask': 'protection.png',
        'note': 'Only the environment is repainted; source equipment lighting is preserved.'
    }, indent=2), encoding='utf-8')
