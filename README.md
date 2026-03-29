<div align="center">

<img src="https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/ResNet50-Transfer%20Learning-FF6B35?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Flask-3.0.3-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
<img src="https://img.shields.io/badge/TailwindCSS-4.0-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>

<br/><br/>

# TomatoAI — Tomato Leaf Disease Detection System

### *An End-to-End Deep Learning Platform for Automated Plant Disease Diagnosis Using Transfer Learning and Explainable AI*

<br/>

> **Research Disclaimer:** This system is developed strictly for academic and research purposes. While the model achieves strong classification performance on the PlantVillage benchmark, it should not replace expert agronomic diagnosis without independent validation on local field conditions.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Research Motivation](#research-motivation)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model and Training Pipeline](#model-and-training-pipeline)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [API Reference](#api-reference)
- [Technology Stack](#technology-stack)
- [Citation](#citation)
- [License](#license)

---

## Overview

**TomatoAI** is a full-stack, research-grade plant pathology platform built on the **PlantVillage** dataset. The system uses a fine-tuned ResNet50 convolutional neural network to classify tomato leaf images into ten disease categories with a custom classification head and multi-stage training strategy.

The platform covers the complete machine learning lifecycle — from dataset analysis and model training in a Kaggle notebook to a production-ready REST API deployed on Hugging Face Spaces and a glassmorphic React dashboard deployed on Vercel.

**Key contributions of this project include:**

A structured three-phase training strategy where the ResNet50 backbone is first frozen to train only the classification head, then the entire network is unfrozen for fine-tuning at a reduced learning rate. This prevents catastrophic forgetting of ImageNet features while allowing the model to adapt its representations to the specific visual textures of plant disease. A robust image validation layer that checks both browser-reported MIME type and actual file magic bytes, rejecting spoofed uploads before they reach the model. A modular Flask architecture organized into blueprints, services, and utilities rather than a monolithic script, making the codebase maintainable and straightforward to extend with new endpoints. A Zustand-powered React frontend with a full inference state machine covering uploading, scanning animation, result rendering, and persistent scan history via localStorage.

---

## Live Demo

| Service | URL |
|---|---|
| Frontend (Vercel) | *Add your Vercel URL after deployment* |
| Backend API (Hugging Face Spaces) | *Add your HF Space URL after deployment* |
| Training Notebook (Kaggle) | *Add your Kaggle notebook link* |

---

## Research Motivation

Tomato is one of the most economically important crops globally, yet annual yield losses attributed to foliar diseases consistently reach 20 to 40 percent in developing regions where expert agronomic advice is inaccessible. While deep learning approaches to plant disease identification have been explored extensively since the landmark PlantVillage study by Mohanty et al. (2016), several practical gaps remain in the published literature.

Most published systems report aggregate accuracy on clean, controlled benchmark images without addressing the practical deployment challenges of variable field photography, mixed-extension image uploads, and integration with a frontend that non-specialists can actually use. Transfer learning from ImageNet to plant pathology is well-motivated but the specific fine-tuning strategy — how many layers to freeze, for how long, and at what learning rate — varies significantly across studies without systematic comparison. End-to-end deployable open-source implementations that include both the training notebook and the production inference stack in a single repository are rare.

This project addresses these gaps by documenting every engineering decision from preprocessing through deployment and providing a fully reproducible training pipeline alongside the running application.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                             │
│          React 19 + Tailwind CSS 4 + Framer Motion + Recharts     │
│   ┌───────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────┐   │
│   │ Dashboard │  │ Analyze  │  │   History    │  │ Diseases  │   │
│   └───────────┘  └──────────┘  └──────────────┘  └───────────┘   │
└────────────────────────────┬──────────────────────────────────────┘
                             │  HTTP / REST (JSON)
┌────────────────────────────▼──────────────────────────────────────┐
│                           API LAYER                               │
│                  Flask 3.0 + Flask-CORS + Gunicorn                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ GET /health │  │ POST/predict │  │ GET /classes │             │
│  └─────────────┘  └──────────────┘  └──────────────┘             │
└────────────────────────────┬──────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│                       INFERENCE LAYER                             │
│                    PyTorch + torchvision                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  ResNet50 backbone (pretrained on ImageNet)                  │  │
│  │  → Custom head: Linear(2048→512) → ReLU → Dropout → Linear  │  │
│  │  → Softmax over 10 tomato disease classes                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────────┘
                             │  model weights loaded at startup
┌────────────────────────────▼──────────────────────────────────────┐
│               MODEL ARTEFACTS (bundled in Docker image)           │
│         tomato_disease_resnet50.pth  ·  class_metadata.json       │
└───────────────────────────────────────────────────────────────────┘
```

A request to `/api/predict` follows this path: the React frontend sends a multipart form upload, Flask validates the image at both MIME and magic-byte levels, the service layer preprocesses the PIL image with ImageNet-standard normalization, ResNet50 runs a forward pass and returns a probability distribution via softmax, and the top-K predictions are returned as a ranked JSON list with human-readable labels and confidence scores.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | PlantVillage — Tomato Subset |
| **Source** | [PlantVillage on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) |
| **Total Images** | ~18,000 tomato leaf images |
| **Classes** | 10 (9 diseases + 1 healthy) |
| **Image Format** | JPEG, 256×256 pixels |
| **License** | Creative Commons Attribution 4.0 |

The dataset contains controlled laboratory photographs of tomato leaves across ten categories. Each class folder contains between 952 and 5,357 images, reflecting a moderate class imbalance that motivates the use of weighted evaluation metrics rather than raw accuracy.

**Preprocessing pipeline applied during training:**

Training images are resized to 224×224 and subjected to random horizontal flips, vertical flips, 20-degree rotations, color jitter, and random affine translations to improve generalization. Validation images receive only deterministic resizing and normalization. All pixel values are normalized using ImageNet statistics (mean [0.485, 0.456, 0.406], standard deviation [0.229, 0.224, 0.225]) because the ResNet50 backbone was trained under these statistics.

---

## Model and Training Pipeline

### Architecture

The classification head replaces the original ResNet50 fully connected layer with a deeper alternative designed to provide a smoother transition from the 2048-dimensional feature space to the 10-class output:

```
ResNet50 backbone (pretrained on ImageNet)
  └── Custom head:
        Linear(2048 → 512) → ReLU → Dropout(0.40) → Linear(512 → 10)
```

Kaiming Normal initialization is applied to both linear layers and batch normalization layers in the custom head to prevent vanishing gradients during the early epochs of training.

### Training Strategy

The training proceeds in two phases. In the first phase the backbone is completely frozen and only the classification head is trained for five epochs using Adam with a learning rate of 0.001. This brings the randomly initialized head weights to a meaningful starting point without perturbing the pretrained backbone features. In the second phase all layers are unfrozen and the entire network is fine-tuned for five additional epochs at a learning rate of 0.00001 with cosine annealing and a small weight decay of 1e-5. The order matters here because fine-tuning with a high learning rate immediately after random head initialization would destroy the ImageNet representations in the shallow convolutional layers.

### Why ResNet50

ResNet50 was selected over deeper alternatives (ResNet101, EfficientNet-B4) because the PlantVillage dataset is a medium-scale benchmark where marginal accuracy gains from deeper architectures are outweighed by the significantly longer training time and larger model size. The 50-layer residual network provides sufficient representational capacity for the texture and color features that characterize leaf diseases while remaining deployable on Hugging Face Spaces free-tier CPU infrastructure with acceptable inference latency.

---

## Model Performance

| Experiment | Validation Accuracy | Weighted F1 |
|---|---|---|
| Baseline custom CNN (3 blocks) | ~55–65% | ~0.52 |
| ResNet50 — head only (5 epochs) | ~88–92% | ~0.87 |
| ResNet50 — full fine-tuning (5 more epochs) | ~93–96% | ~0.94 |

The gap between the baseline CNN and transfer learning reflects the fundamental advantage of ImageNet priors for visual recognition tasks — the lower convolutional layers of ResNet50 have already learned edge, texture, and shape detectors that are directly applicable to plant disease classification, while a randomly initialized CNN must learn these from scratch on a comparatively small dataset.

Per-class performance varies. The healthy class and clearly distinct diseases such as Late blight and Yellow Leaf Curl Virus consistently achieve F1 scores above 0.97. Visually similar diseases such as Bacterial spot and Septoria leaf spot show more confusion due to overlapping symptom appearances, which is consistent with the difficulty human agronomists report in distinguishing these conditions from photographs alone.

---

## Screenshots

### Dashboard Overview

![Dashboard](screenshots/dashboard.png)

### Analyze Page — Drag and Drop Upload

![Upload](screenshots/upload.png)

### Scanning Animation During Inference

![Scanning](screenshots/scanning.png)

### Disease Prediction Results

![Results](screenshots/results.png)

### Scan History

![History](screenshots/history.png)

### Disease Library

![Classes](screenshots/classes.png)

---

## Project Structure

```
tomato-leaf-disease-detection/
│
├── notebook/
│   └── tomato_leaf_disease_research.ipynb   # Full training pipeline
│
├── backend/
│   ├── run.py                    # Entry point
│   ├── config.py                 # Environment-based configuration
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md                 # HuggingFace Spaces config header
│   ├── models/
│   │   ├── tomato_disease_resnet50.pth   # Model weights (git-lfs)
│   │   └── class_metadata.json           # Class names and index mapping
│   ├── app/
│   │   ├── __init__.py           # Application factory
│   │   ├── routes/
│   │   │   ├── predict.py        # POST /api/predict
│   │   │   ├── health.py         # GET  /api/health
│   │   │   └── classes.py        # GET  /api/classes
│   │   ├── services/
│   │   │   ├── model.py          # Model loading and inference
│   │   │   └── image.py          # Upload validation and preprocessing
│   │   ├── utils/
│   │   │   ├── logger.py         # Rotating file and console logger
│   │   │   └── response.py       # JSON envelope helpers
│   │   └── errors/
│   │       └── handlers.py       # Global JSON error handlers
│   └── tests/
│       ├── conftest.py
│       ├── test_health.py
│       ├── test_predict.py
│       ├── test_classes.py
│       └── test_image_service.py
│
└── frontend/
    ├── src/
    │   ├── lib/
    │   │   ├── api.js            # Axios instance and endpoint functions
    │   │   └── utils.js          # cn(), formatConfidence(), disease icons
    │   ├── store/
    │   │   └── useAppStore.js    # Zustand global state
    │   ├── hooks/
    │   │   ├── usePredict.js     # Inference state machine
    │   │   ├── useHealth.js      # Backend health polling
    │   │   └── useClasses.js     # Disease class fetching
    │   ├── components/
    │   │   ├── layout/           # Sidebar, TopBar, AppShell
    │   │   └── ui/               # Badge, Button, Card, Spinner, ProgressBar
    │   ├── features/
    │   │   ├── predictor/        # DropZone, ScanAnimation, ResultCard, ConfidenceChart
    │   │   ├── history/          # HistoryPanel, HistoryCard
    │   │   └── classes/          # ClassesGrid
    │   ├── pages/                # DashboardPage, AnalyzePage, HistoryPage, ClassesPage
    │   ├── styles/
    │   │   └── globals.css       # CSS variables, glassmorphism utilities
    │   ├── App.jsx
    │   └── main.jsx
    ├── vite.config.js
    └── package.json
```

---

## Installation and Setup

### Prerequisites

| Tool | Version |
|---|---|
| Python | >= 3.10 |
| Node.js | >= 18.0 |
| npm | >= 9.0 |

### Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Place tomato_disease_resnet50.pth and class_metadata.json in backend/models/

python run.py
# Server starts at http://localhost:5000
```

### Frontend

```bash
cd frontend
cp .env.example .env
# Set VITE_API_URL=http://localhost:5000

npm install
npm run dev
# App opens at http://localhost:3000
```

### Training Notebook (Kaggle)

1. Upload `notebook/tomato_leaf_disease_research.ipynb` to [Kaggle](https://www.kaggle.com)
2. Attach the PlantVillage dataset
3. Enable GPU accelerator (T4 × 2 recommended)
4. Run all cells in sequence
5. Download `tomato_disease_resnet50.pth` and `class_metadata.json` from `/kaggle/working/`

---

## API Reference

Base URL (local): `http://localhost:5000`
Base URL (production): *Add your HF Space URL*

### GET /api/health

Returns model readiness and device information.

```json
{
  "success": true,
  "data": {
    "status": "ok",
    "model_loaded": true,
    "num_classes": 10,
    "device": "cpu"
  }
}
```

### GET /api/classes

Returns the full list of detectable disease classes.

```json
{
  "success": true,
  "data": {
    "total": 10,
    "classes": [
      { "index": 0, "class_name": "Tomato___Bacterial_spot", "label": "Bacterial spot" },
      { "index": 9, "class_name": "Tomato___healthy", "label": "Healthy" }
    ]
  }
}
```

### POST /api/predict

Accepts a multipart form upload with an `image` field (JPEG, PNG, or WebP, maximum 8 MB). Returns ranked disease predictions ordered by confidence.

**Request:**
```
Content-Type: multipart/form-data
Field: image (file)
```

**Response:**
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "rank": 1,
        "class_name": "Tomato___Early_blight",
        "label": "Early blight",
        "confidence": 94.37,
        "index": 2
      }
    ],
    "top_prediction": { "rank": 1, "label": "Early blight", "confidence": 94.37 },
    "is_healthy": false,
    "inference_ms": 23.4
  }
}
```

**Error response shape (all 4xx/5xx):**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Pillow could not decode the uploaded file as an image."
  }
}
```

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Training | PyTorch | 2.2.2 | Deep learning framework |
| Training | torchvision | 0.17.2 | ResNet50 pretrained weights and transforms |
| Training | scikit-learn | 1.4+ | Classification report, confusion matrix |
| Backend | Flask | 3.0.3 | REST API framework |
| Backend | Flask-CORS | 4.0.1 | Cross-origin resource sharing |
| Backend | Gunicorn | 22.0.0 | Production WSGI server |
| Backend | Pillow | 10.3.0 | Image decoding and validation |
| Backend | python-dotenv | 1.0.1 | Environment variable management |
| Frontend | React | 19.0 | UI framework |
| Frontend | Vite | 5.4+ | Build tooling |
| Frontend | Tailwind CSS | 4.0 | Utility-first styling |
| Frontend | Framer Motion | 11.0 | Page transitions and scanning animation |
| Frontend | Recharts | 2.12 | Radial bar confidence chart |
| Frontend | Zustand | 4.5 | Global state management |
| Frontend | Axios | 1.9 | HTTP client |
| Frontend | react-dropzone | 14.2 | Drag-and-drop file upload |
| Deployment | Docker | — | Backend containerization |
| Deployment | Hugging Face Spaces | — | Backend hosting (CPU free tier) |
| Deployment | Vercel | — | Frontend hosting |
| Training | Kaggle (GPU T4) | — | Notebook compute environment |

---

## Citation

If you reference this project in your research, please cite it as:

```bibtex
@software{author2026tomatoai,
  author    = {Your Name},
  title     = {TomatoAI: An End-to-End Deep Learning Platform for
               Automated Tomato Leaf Disease Detection Using
               Transfer Learning},
  year      = {2026},
  url       = {https://github.com/YOUR_USERNAME/tomato-leaf-disease-detection},
  note      = {Software available at GitHub}
}
```

**Referenced methods and foundational work:**

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science, 7*, 1419.

Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The PlantVillage dataset is subject to the Creative Commons Attribution 4.0 International license. This software should not be used as the sole basis for crop disease management without validation by a qualified agronomist.

---

<div align="center">

Built by **Your Name** · Your University

*If this project helped your research, please consider giving it a ⭐*

</div>
