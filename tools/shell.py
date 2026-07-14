from modal import Image, App

image = (
    Image.from_registry("nvidia/cuda:12.3.0-devel-ubuntu22.04")
    .apt_install("cmake", "build-essential")
)

app = App("dev", image=image)