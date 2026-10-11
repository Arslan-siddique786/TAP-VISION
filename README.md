# TAP-VISION

**Real-Time Point Tracking and Motion Visualization Powered by TAPIR**

TAP-VISION is a computer-vision project built around **Tracking Any Point (TAP)** research. Its featured demo lets you select points in a live webcam feed and visualize their motion as objects or the camera move.

This repository is a personal adaptation and experimentation workspace based on the original TapNet research codebase, with a dedicated Skill Fair demo and additional tracking experiments.

> **Attribution:** This project builds on [Google DeepMind's official TapNet repository](https://github.com/google-deepmind/tapnet). TAP-VISION is an independent project and is not an official Google DeepMind product. Please see the included license and upstream documentation for original authorship and research details.

## ✨ Features

- **Real-time point tracking:** Select points in a webcam feed and track their movement.
- **Multiple tracking points:** The featured demo supports up to eight point slots.
- **Motion trails:** Color-coded points and trails help visualize movement.
- **Live information panel:** Displays tracking status, active points, and approximate FPS.
- **GPU acceleration:** Uses CUDA when a compatible PyTorch installation is available, with CPU fallback.
- **Research experiments:** Includes notebooks and code for TAPIR, TAPNext, TAP-Vid, RoboTAP, TAPVid-3D, and related work from the upstream project.

## 🎮 Demo Controls

In the TAP-VISION Skill Fair demo:

| Action | Control |
|---|---|
| Select a tracking point | Left-click on the video |
| Reset tracking | Press `R` |
| Exit the demo | Press `Esc` |

A working webcam and Windows camera permissions are required for live camera demos.

## 📁 Project Structure

```text
TAP-VISION/
├── assets/                  # Project assets
├── colabs/                  # Research and demo notebooks
├── configs/                 # Model configurations
├── tapnet/
│   ├── models/              # Model components
│   ├── torch/               # PyTorch TAPIR implementation
│   ├── skill_fair_demo.py   # Featured Skill Fair demo
│   ├── pytorch_live_demo.py # PyTorch webcam demo
│   └── tapir_performance_test.py
├── requirements.txt
├── requirements_inference.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

### Key scripts

- `tapnet/skill_fair_demo.py`: Live point-tracking demo with a status panel and motion trails.
- `tapnet/pytorch_live_demo.py`: PyTorch online TAPIR webcam demo.
- `tapnet/tapir_performance_test.py`: Local performance-testing script.

Some files are adapted from or included with the upstream research project. File headers and upstream documentation should be consulted when determining individual authorship.

## 🛠️ Requirements

- Windows 10 or Windows 11
- Python 3.11 recommended for this setup
- A webcam for live tracking
- PyTorch and the project's inference dependencies
- The pretrained model checkpoint described below
- A compatible NVIDIA GPU and CUDA-enabled PyTorch build for GPU acceleration

CPU execution is possible but may be significantly slower. Performance depends on the hardware, model, input resolution, and software configuration.

## 🚀 Installation on Windows

Open PowerShell and clone the repository:

```powershell
git clone https://github.com/Arslan-siddique786/TAP-VISION.git
cd TAP-VISION

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install **PyTorch and torchvision** using the official selector so the installation matches your machine and CUDA setup:

[PyTorch: Get Started](https://pytorch.org/get-started/locally/)

Then install the repository's inference dependencies:

```powershell
python -m pip install -r requirements_inference.txt
```

Verify your PyTorch installation:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is available, you can also inspect the GPU name:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU')"
```

## 📦 Model Checkpoint

The pretrained model weights are **not stored in this Git repository** because the checkpoint is large.

Create the checkpoint directory and download the online BootsTAPIR checkpoint from the upstream project's storage:

```powershell
New-Item -ItemType Directory -Force "tapnet\checkpoints" | Out-Null

Invoke-WebRequest -Uri "https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt" -OutFile "tapnet\checkpoints\causal_bootstapir_checkpoint.pt"
```

The expected file path is:

```text
tapnet/checkpoints/causal_bootstapir_checkpoint.pt
```

The live scripts load the checkpoint using a relative path, so launch them from the repository root. If you already have the correct checkpoint locally, you do not need to download it again.

## ▶️ Run the Demos

Make sure your virtual environment is activated, the dependencies are installed, and the checkpoint is present.

### 1. TAP-VISION Skill Fair Demo

```powershell
python -m tapnet.skill_fair_demo
```

### 2. PyTorch Live Demo

```powershell
python -m tapnet.pytorch_live_demo
```

### 3. Performance Test

```powershell
python -m tapnet.tapir_performance_test
```

### Troubleshooting

- **Camera not opening:** Check Windows camera permissions, close other camera applications, and verify that your webcam is connected.
- **Checkpoint loading error:** Confirm that the checkpoint exists at the expected path and is compatible with the model.
- **CUDA unavailable:** Verify that you installed a compatible CUDA-enabled PyTorch build and that your NVIDIA drivers are working.
- **Low FPS:** Performance varies with hardware and input settings. CPU inference can be substantially slower than GPU inference.

## 🔬 Upstream Research

TAP-VISION builds on research and tools developed by the computer-vision community. Explore the original resources:

- [Google DeepMind TapNet](https://github.com/google-deepmind/tapnet)
- [TAP-Vid Benchmark](https://tapvid.github.io/)
- [TAPIR Project](https://deepmind-tapir.github.io/)
- [BootsTAP](https://bootstap.github.io/)
- [TAPVid-3D Benchmark](https://tapvid3d.github.io/)
- [Pretrained TapNet Models](https://huggingface.co/google/tapnet)

For full benchmark instructions, dataset details, training information, and research citations, refer to the official TapNet repository.

## 📄 License and Attribution

This repository retains the upstream Apache 2.0 license file. Some datasets, benchmarks, notebooks, and related components may have separate terms, so check their individual documentation before reuse.

Please credit the original researchers and cite the relevant papers when using their work. TAP-VISION's project-specific demo and experiments are maintained in this repository by its owner.

Contributions, suggestions, and bug reports are welcome through GitHub issues and pull requests.
