#  Language-Enhanced Cross-Dimensional Synthesis (LECS)
### *From 2D Observations to 3D Medical Scene Reconstruction for Real-Time Interventional Navigation*

<div align="center">

![LECS Logo](https://img.shields.io/badge/LECS-Medical_AI-red?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iNDUiIGZpbGw9IiNmZjRhNGEiLz4KPHBhdGggZD0iTTUwIDI1IEw2NSA0NSBMNTIGNSBTMCA2NSBMNTAgODUgTDM1IDY1IFoiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPg==)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-orange.svg)](https://anonymous.4open.science/r/continuum_control-3DDC)
[![Stars](https://img.shields.io/github/stars/username/LECS?style=social)](https://github.com/username/LECS)

<h3>🔬 Bridging Semantic Understanding and 3D Reconstruction in Interventional Cardiology</h3>

[**📖 Code**]([https://anonymous.4open.science/r/continuum_control-3DDC](https://anonymous.4open.science/status/LECS-7EA7)) | [**🎥 Demo**](#demo) | [**📊 Dataset**](#dataset) | [**🤝 Contributing**](#contributing)

</div>

---

## 📋 Overview

LECS revolutionizes minimally invasive cardiovascular interventions by introducing **semantic-guided cross-modal synthesis** that leverages multimodal large language models (MLLMs) to transform 2D fluoroscopic observations into accurate 3D medical scene reconstructions.

### ✨ Key Features

- 🧠 **Medical-Domain MLLM**: Fine-tuned on extensive cardiovascular datasets for superior anatomical understanding
- 🔄 **Semantic-to-Geometry Translation**: Novel neural architecture converting medical descriptions to 3D primitives
- 🎯 **Adaptive Neural Rendering**: Real-time scene synthesis with uncertainty quantification
- 📈 **State-of-the-Art Performance**: 15% improvement in anatomical consistency, 30% better robustness under low-contrast conditions
- ⚡ **Real-Time Processing**: ~278ms inference time for clinical applicability


## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# CUDA 11.0+ for GPU acceleration
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/username/LECS.git
cd LECS

# Install dependencies
pip install -e .

# Download pre-trained models
python scripts/download_models.py
```

### Basic Usage

```python
from LECS import MedicalSceneReconstructor

# Initialize the model
reconstructor = MedicalSceneReconstructor(
    model_path="models/lecs_cardiovascular.pth",
    device="cuda"
)

# Process fluoroscopic sequence
result = reconstructor.process(
    fluoroscopy_frames="path/to/frames",
    metadata="path/to/metadata.json"
)

# Get 3D reconstruction
reconstruction = result.get_3d_scene()
semantic_description = result.get_semantic_analysis()
```

## 📂 Repository Structure

```
LECS/
│
├── 📁 applications/          # Clinical application demos
│   ├── coronary_intervention/
│   ├── guidewire_tracking/
│   └── vessel_reconstruction/
│
├── 📁 baseline/              # Baseline comparison methods
│   ├── geometric_methods/
│   ├── deep_learning/
│   └── multimodal_fusion/
│
├── 📁 data/                  # Data processing utilities
│   ├── loaders/
│   ├── preprocessing/
│   └── augmentation/
│
├── 📁 networks/              # Neural network architectures
│   ├── mllm_adapter.py
│   ├── semantic_encoder.py
│   └── neural_renderer.py
│
├── 📁 training_scripts/      # Training configurations
│   ├── train_mllm.py
│   ├── train_translator.py
│   └── train_renderer.py
│
├── 📁 util/                  # Utility functions
│   ├── medical_knowledge.py
│   ├── metrics.py
│   └── visualization.py
│
├── 📜 BasicWireNav_train.py  # Basic guidewire navigation training
├── 📜 DualDeviceNav_train.py # Dual-device navigation training
└── 📜 setup.py               # Package setup configuration
```

## 🎯 Training

### Single Device Navigation
```bash
python BasicWireNav_train.py \
    --data_path /path/to/dataset \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### Dual Device Navigation
```bash
python DualDeviceNav_train.py \
    --data_path /path/to/dataset \
    --multi_view \
    --temporal_consistency
```

## 📊 Performance

| Method | ACS ↑ | CDE (mm) ↓ | VOC ↑ | MCAS ↑ | Time (ms) ↓ |
|--------|-------|------------|-------|--------|-------------|
| **LECS (Ours)** | **0.887** | **1.94** | **0.896** | **0.834** | 278.3 |
| VL-MedAnalysis | 0.812 | 2.28 | 0.834 | 0.718 | 287.9 |
| NeRF-Med | 0.798 | 2.35 | 0.841 | 0.672 | 312.5 |
| Swin-MedSeg | 0.772 | 2.58 | 0.815 | 0.571 | 198.6 |

## 🔬 Example Results

<div align="center">
<table>
<tr>
<td><img src="example/coronary_navigation.gif" width="250"/></td>
<td><img src="example/bifurcation_lesion.gif" width="250"/></td>
<td><img src="example/3d_reconstruction.gif" width="250"/></td>
</tr>
<tr>
<td align="center">Coronary Navigation</td>
<td align="center">Bifurcation Analysis</td>
<td align="center">3D Reconstruction</td>
</tr>
</table>
</div>

## 📚 Dataset

Our model is trained on multi-institutional cardiovascular datasets. For access to the training data:

1. **Public Dataset**: Available at [[Dataset Link](https://anonymous.4open.science/r/LECS-7EA7/data)]
2. **Clinical Dataset**: Request access through institutional review

## 🛠️ Environment Setup

Create a conda environment:

```bash
conda create -n lecs python=3.8
conda activate lecs

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r environment/requirements.txt
```


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Roadmap

- [x] Core framework implementation
- [x] Cardiovascular application
- [ ] Extension to neurological interventions
- [ ] Real-time robotic integration
- [ ] Multi-center clinical validation
- [ ] FDA regulatory submission preparation


## 🙏 Acknowledgments

- Medical imaging datasets from partner institutions
- Clinical validation support from interventional cardiology teams


---

<div align="center">
<b>Advancing Interventional Medicine Through Intelligent Scene Understanding</b><br>
Made with ❤️ by the LECS Research Team
</div>
