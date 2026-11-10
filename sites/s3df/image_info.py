"""Print the installed software of an image as JSON. Stdlib only.

nightly.py pipes this script into the image's python:

    apptainer exec IMAGE.sif python - < image_info.py

Output: {"os", "python", "pip": [[name, version], ...],
"conda": {name: version}, "cmake"}.
"""
import json
import pathlib
import platform
import subprocess
from importlib import metadata


def os_name():
    for line in pathlib.Path("/etc/os-release").read_text().splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return ""


def conda_packages():
    out = {}
    for path in pathlib.Path("/opt/conda/conda-meta").glob("*.json"):
        parts = path.stem.rsplit("-", 2)  # name-version-build
        if len(parts) == 3:
            out[parts[0]] = parts[1]
    return out


def cmake_version():
    try:
        text = subprocess.run(["cmake", "--version"], capture_output=True, text=True, timeout=30).stdout
        return text.split()[2]
    except (OSError, IndexError, subprocess.SubprocessError):
        return ""


pip = sorted({(d.metadata["Name"].lower().replace("_", "-"), d.version) for d in metadata.distributions()})
print(json.dumps({"os": os_name(), "python": platform.python_version(), "pip": pip,
                  "conda": conda_packages(), "cmake": cmake_version()}))
