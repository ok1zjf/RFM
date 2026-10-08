"""The libjpeg-turbo Pillow used by the dataset converters (convert_<dataset>.py).

The published datasets/ images were converted with a libjpeg-turbo build of
Pillow, while training and testing decode them with the IJG libjpeg 9e build
the environment pins (see README.md, "Pillow and JPEG decoding"). The two
libraries decode and encode JPEGs differently, so converting with the IJG
build gives slightly different images. The converters therefore load a second
Pillow, PyPI's (which bundles libjpeg-turbo), from .prep_pillow/ in the
project directory; every other script keeps the environment's IJG Pillow.

    python turbo_pillow.py      # install .prep_pillow/ if missing or wrong (prepare_datasets.sh does this)
"""
import os
import subprocess
import sys

PREP_PILLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".prep_pillow")
PREP_PILLOW_VERSION = "11.1.0"
INSTALL_CMD = [sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "--target", PREP_PILLOW_DIR,
               f"pillow=={PREP_PILLOW_VERSION}"]


def use_turbo_pillow():
    """Make `import PIL` in this process load the .prep_pillow/ libjpeg-turbo
    Pillow. Must run before anything imports PIL: Python caches modules, so a
    later sys.path change would silently keep the IJG build."""
    if "PIL" in sys.modules:
        raise RuntimeError("use_turbo_pillow() must run before PIL is imported "
                           f"(already loaded from {sys.modules['PIL'].__file__})")
    sys.path.insert(0, PREP_PILLOW_DIR)
    import PIL
    import PIL.features
    if not (PIL.__file__.startswith(PREP_PILLOW_DIR + os.sep) and PIL.__version__ == PREP_PILLOW_VERSION
            and PIL.features.check_feature("libjpeg_turbo")):
        raise RuntimeError(f"the dataset converters need Pillow {PREP_PILLOW_VERSION} with libjpeg-turbo in "
                           f"{PREP_PILLOW_DIR}, but loaded Pillow {PIL.__version__} from {PIL.__file__} "
                           f"(libjpeg-turbo: {PIL.features.check_feature('libjpeg_turbo')}). "
                           f"Install it with: python turbo_pillow.py")


def _installed() -> bool:
    # Checked in a fresh interpreter, so this process never imports a PIL.
    check = "import turbo_pillow; turbo_pillow.use_turbo_pillow()"
    return subprocess.run([sys.executable, "-c", check], cwd=os.path.dirname(PREP_PILLOW_DIR),
                          capture_output=True).returncode == 0


def main():
    if _installed():
        print(f"[turbo_pillow] Pillow {PREP_PILLOW_VERSION} (libjpeg-turbo) present in {PREP_PILLOW_DIR}")
        return
    print(f"[turbo_pillow] installing Pillow {PREP_PILLOW_VERSION} (libjpeg-turbo) into {PREP_PILLOW_DIR}", flush=True)
    subprocess.run(INSTALL_CMD, check=True)
    if not _installed():
        sys.exit(f"[turbo_pillow] {PREP_PILLOW_DIR} still does not provide a libjpeg-turbo Pillow "
                 f"{PREP_PILLOW_VERSION} - see the error from: python -c 'import turbo_pillow; "
                 f"turbo_pillow.use_turbo_pillow()'")


if __name__ == "__main__":
    main()
