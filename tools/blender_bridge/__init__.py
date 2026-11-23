from pathlib import Path

__all__ = ["ADDON_PATH"]

ADDON_PATH = Path(__file__).resolve().parent / "blender_bridge_addon.py"
