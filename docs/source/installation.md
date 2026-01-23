# Installation Guide

This guide will walk you through setting up the DTU MLOps 111 project on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12 or higher** - Check with `python --version`
- **Git** - For cloning the repository
- **UV Package Manager** - Modern Python package and project manager (we'll install this below)

!!! note "Why UV?"
UV is a fast Python package manager written in Rust. It's significantly faster than pip and handles dependency resolution more efficiently. The project uses UV for all package management tasks.

## Step 1: Install UV Package Manager

UV is the package manager used throughout this project. Install it based on your operating system:

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:

```bash
source $HOME/.cargo/env
```

### Windows

Using PowerShell:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Verify Installation

Check that UV is installed correctly:

```bash
uv --version
```

You should see output like: `uv 0.x.x`

!!! tip "UV Documentation"
For more details about UV, visit the [official UV documentation](https://docs.astral.sh/uv/).

## Step 2: Clone the Repository

Clone the project repository from GitHub:

```bash
git clone https://github.com/ecenazelverdi/DTU_MLOps_111.git
cd DTU_MLOps_111
```

## Step 3: Install Project Dependencies

UV will automatically create a virtual environment and install all dependencies defined in `pyproject.toml`:

```bash
uv sync
```

This command will:

- Create a virtual environment in `.venv/`
- Install all project dependencies
- Install development dependencies (pytest, ruff, pre-commit, etc.)
- Install the `dtu_mlops_111` package in editable mode

!!! note "First Installation"
The first `uv sync` may take a few minutes as it downloads and installs all dependencies, including PyTorch and nnU-Net.

### Installing PyTorch

The project configuration includes indexes for both CPU and CUDA versions of PyTorch. `uv sync` will automatically install the appropriate version for your platform.

For installation:

```bash
uv sync
```

## Step 4: Verify Installation

After installation, verify everything is set up correctly:

```bash
# Activate the virtual environment (UV does this automatically, but you can do it manually)
source .venv/bin/activate  # On macOS/Linux
# Or on Windows:
# .venv\Scripts\activate

# Check Python version
python --version

# Verify the package is installed
python -c "import dtu_mlops_111; print('Package installed successfully!')"
```

## Step 5: Install DVC (Data Version Control)

DVC is used for data and model versioning. It should already be installed via `uv sync`, but verify:

```bash
uv run dvc --version
```

## Step 6: Configure Git (Optional for Contributors)

If you plan to contribute to the project, set up pre-commit hooks:

```bash
uv run pre-commit install
```

This will automatically run code formatters and linters before each commit.

## Next Steps

Now that you have the project installed, proceed to:

1. [**Environment Setup**](environment.md) - Configure your `.env` file with credentials and paths
2. [**Quick Start**](quickstart.md) - Run your first workflow

## Troubleshooting

### UV Command Not Found

If `uv` is not recognized after installation:

- Restart your terminal
- Ensure the cargo bin directory is in your PATH: `export PATH="$HOME/.cargo/bin:$PATH"`

### Python Version Issues

If you have multiple Python versions:

```bash
# Use UV with a specific Python version (e.g., 3.13)
uv sync --python 3.13
```

### Dependency Conflicts

If you encounter dependency resolution issues:

```bash
# Clear the cache and reinstall
uv cache clean
uv sync --reinstall
```

For more issues, see the [Troubleshooting](troubleshooting.md) page.
