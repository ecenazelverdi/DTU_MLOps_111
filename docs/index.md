# DTU MLOps 111: Semantic Segmentation for Drone Imagery

Welcome to the documentation for Team 111's MLOps project!

## Project Overview

This project implements **semantic segmentation for drone imagery** using the nnU-Net architecture. The system identifies five different classes of objects from aerial urban scenes:

- 🟪 **Obstacles** - Buildings, structures, and physical barriers
- 🟦 **Water** - Rivers, lakes, and water bodies
- 🟩 **Soft Surfaces** - Grass, vegetation, and unpaved areas
- 🟥 **Moving Objects** - Vehicles, people, and dynamic obstacles
- ⬜️ **Landing Zones** - Safe areas for drone landing

### Purpose

The trained model enhances the safety of **autonomous drone flights and landings** in urban areas by accurately distinguishing different types of obstacles from safe landing zones.

## Quick Navigation

### 🚀 Getting Started

New to the project? Start here:

1. [**Installation**](source/installation.md) - Set up your development environment
2. [**Environment Setup**](source/environment.md) - Configure credentials and paths
3. [**Quick Start**](source/quickstart.md) - Run your first workflow

### 📚 User Guide

- [**Commands Reference**](source/commands.md) - Complete `tasks.py` command documentation
- [**Workflows**](source/workflows.md) - Step-by-step guides for data, training, and API
- [**Troubleshooting**](source/troubleshooting.md) - Solutions to common issues

## Key Features

- ✅ **nnU-Net Architecture** - State-of-the-art medical imaging segmentation adapted for drones
- ✅ **MLOps Best Practices** - DVC for data versioning, Docker for containerization
- ✅ **FastAPI Inference** - Production-ready REST API for real-time predictions
- ✅ **High-Performance Serving** - BentoML integration for model packaging and adaptive batching
- ✅ **Cloud Deployment** - GCP Cloud Run deployment with data drift monitoring
- ✅ **Comprehensive Testing** - Automated tests with coverage reporting

## Dataset

The project uses the [Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset/data) containing 400 RGB drone images with pixel-level semantic labels.

_Original dataset from [TU Graz, IVC](https://ivc.tugraz.at/research-project/semantic-drone-dataset/)_

## Technology Stack

- **Framework**: PyTorch with nnU-Net
- **Package Manager**: UV
- **Version Control**: Git + DVC
- **API**: FastAPI + Uvicorn & BentoML
- **Containers**: Docker
- **Cloud**: Google Cloud Platform (GCS, Cloud Run)
- **Monitoring**: Weights & Biases, Evidently

## Team 111

- Ecenaz Elverdi (s252699@dtu.dk)
- Akin Mert Gümüs (s242508@dtu.dk)
- Elif Pulukcu (s252749@dtu.dk)
- Kerick Jon Walker (s252618@dtu.dk)
- Bruno Zorrila Medina Luna (s260015@dtu.dk)

---

Ready to get started? Head to the [Installation Guide](source/installation.md)!
