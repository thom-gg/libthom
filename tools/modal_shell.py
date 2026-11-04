"""Interactive shell on a Modal GPU with the libthom toolchain image.

    modal shell tools/modal_shell.py

Modal builds the image from the Dockerfile itself (linux/amd64, no registry needed) and caches it
between runs. The repository is copied on top of it in /workspace, as regular files that can be edited,
so the same commands as in the local container work:

    cmake -S . -B build -G Ninja -DARCH=80 && cmake --build build

The copy is a snapshot taken when the shell starts: changes made on the laptop afterwards need a new shell.
Only the copy layer is rebuilt then, the Dockerfile part stays cached.
The container is deleted when the shell exits: bring back what you want to keep (bench CSVs,
`git diff` of your changes) before exiting.
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = "A100"
TIMEOUT_SECONDS = 2 * 60 * 60   # Modal's default (5 min) also applies to the shell

image = (
    # Modal needs its own Python in the image to start the container. It may come first in the PATH:
    # run tools/plot.py with /usr/bin/python3, the one that has matplotlib
    modal.Image.from_dockerfile(REPO / "Dockerfile", add_python="3.12")
    # copy=True: the files are part of the image, so they're writable (a plain mount is read-only)
    .add_local_dir(REPO, remote_path="/workspace", copy=True,
                   ignore=["build", ".git", "*.csv", "*.nsys-rep", "*.ncu-rep", "plot.png"])
)

app = modal.App("libthom", image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_SECONDS)
def shell():
    """Never called: `modal shell` only uses this function's spec (image, GPU, timeout)."""
