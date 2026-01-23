# Troubleshooting

Common issues and their solutions.

## Installation Issues

### UV Command Not Found

**Problem:** After installing UV, the command is not recognized.

```bash
uv: command not found
```

**Solution:**

1. Restart your terminal
2. Add cargo bin directory to PATH:
   ```bash
   export PATH="$HOME/.cargo/bin:$PATH"
   ```
3. Make it permanent by adding to your shell profile:
   ```bash
   echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc  # macOS
   echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc  # Linux
   source ~/.zshrc  # or ~/.bashrc
   ```

### Python Version Mismatch

**Problem:** Wrong Python version installed.

```
error: Python 3.11 is not supported (requires Python 3.12+)
```

**Solution:**

```bash
# Use specific Python version with UV
uv sync --python 3.13
```

### Dependency Resolution Errors

**Problem:** UV can't resolve dependencies.

```
error: Failed to resolve dependencies
```

**Solution:**

```bash
# Clear UV cache
uv cache clean

# Reinstall dependencies
uv sync --reinstall

# If still failing, check pyproject.toml for conflicts
```

---

## Environment Configuration

### Environment Variables Not Set

**Problem:** nnU-Net can't find data directories.

```
RuntimeError: nnUNet_raw is not defined
```

**Solution:**

1. Ensure `.env` file exists:
   ```bash
   ls .env
   ```
2. Load environment variables:
   ```bash
   source .env
   ```
3. Verify they're set:
   ```bash
   echo $nnUNet_raw
   echo $nnUNet_preprocessed
   echo $nnUNet_results
   ```
4. If empty, check `.env` file has correct `export` syntax:
   ```bash
   nnUNet_raw="/absolute/path/to/nnUNet_raw"
   ```
   or
   ```bash
   export nnUNet_raw="/absolute/path/to/nnUNet_raw"
   ```

### Kaggle API Credentials Invalid

**Problem:** Can't download data from Kaggle.

```
401 Unauthorized: Invalid API credentials
```

**Solution:**

1. Verify credentials in `.env`:
   ```bash
   cat .env | grep KAGGLE
   ```
2. Check Kaggle API key is current:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   - Create new token
   - Update `.env` with new credentials
3. Ensure no extra spaces or quotes:
   ```bash
   KAGGLE_USERNAME="your_username"
   KAGGLE_KEY="your_api_key"
   ```

### Weights & Biases Authentication

**Problem:** W&B login fails.

```
wandb: ERROR authentication failed
```

**Solution:**

```bash
# Set API key in .env
WANDB_API_KEY="your_key_here"

# Or login manually
uv run wandb login
```

---

## Data Issues

### Data Download Fails

**Problem:** Download from Kaggle times out or fails.

**Solution:**

```bash
# Try again with verbose output
uv run python -c "import kaggle; kaggle.api.dataset_download_files(
    'santurini/semantic-segmentation-drone-dataset',
    path='data/raw',
    unzip=True
)"

# Check internet connection
ping kaggle.com

# Check disk space
df -h
```

### Data Export Fails

**Problem:** Converting to nnU-Net format fails.

```
FileNotFoundError: data/raw/classes_dataset not found
```

**Solution:**

1. Verify data was downloaded:
   ```bash
   ls data/raw/classes_dataset/classes_dataset/original_images/
   ```
2. If missing, re-download:
   ```bash
   uv run invoke download-data
   ```

### Preprocessing Fails

**Problem:** nnU-Net preprocessing crashes.

```
RuntimeError: Dataset verification failed
```

**Solution:**

1. Check data integrity:
   ```bash
   ls nnUNet_raw/Dataset101_DroneSeg/imagesTr/ | wc -l
   # Should show number of images
   ```
2. Verify environment variables are absolute paths:
   ```bash
   echo $nnUNet_raw
   # Should be /full/path/to/nnUNet_raw, not ./nnUNet_raw
   ```
3. Re-export data:
   ```bash
   rm -rf nnUNet_raw/Dataset101_DroneSeg
   uv run invoke export-data
   ```

### Python 3.13 Compatibility / Distutils Error

**Problem:** Preprocessing can fail on Python 3.13 with errors related to `distutils` (e.g., `ModuleNotFoundError: No module named 'distutils'`). This is due to `distutils` being removed in recent Python versions. Note that **this error may not occur for everyone**, as it depends on your specific environment and how dependencies are resolved.

**Solution:** Downgrade your environment to Python 3.12, which is the fully supported version for this project's dependencies.

```bash
# Force UV to use Python 3.12
uv python install 3.12
uv venv --python 3.12
uv sync

# Then try preprocessing again
uv run invoke preprocess
```

---

## Training Issues

### CUDA Out of Memory

**Problem:** GPU runs out of memory during training.

```
RuntimeError: CUDA out of memory
```

**Solution:**

1. Reduce batch size (nnU-Net does this automatically, but you can force smaller batch):
   ```bash
   # Train on CPU if GPU is too small
   uv run invoke train --device cpu
   ```
2. Use smaller model dimension:
   ```bash
   uv run invoke train --dim 2d  # Instead of 3d_fullres
   ```
3. Close other GPU-using applications
4. Monitor GPU usage:
   ```bash
   nvidia-smi
   ```

### Frozen Training / Background Workers Stopped

**Problem:** Training hangs indefinitely or fails with an error like `DataLoader worker (pid X) is killed` or `Background workers stopped`. No explicit "Out of Memory" error is shown. This is often due to shared memory (shm) exhaustion or system resource leaks during long training runs.

**Solution:**

1. **Clear System Cache:** Often a system restart (as you found) clears the leaked resources.
2. **Reduce Workers:** If it happens frequently, try reducing the number of data loader workers in your training configuration (though nnU-Net handles this, system-level pressure can still trigger it).
3. **Check Shared Memory:** If running in Docker, ensure you have increased `--shm-size` (see Docker OOM section above).
4. **Monitor RAM:** Ensure your system is not swapping heavily.

### Docker Out of Memory

**Problem:** Training crashes inside Docker with a "Killed" message or generic memory errors, even if your GPU has enough memory. This usually happens because the Docker VM itself (on macOS or Windows) has allocated too little RAM.

**Solution:** Increase the memory allocated to Docker Desktop.

1. Open **Docker Desktop Settings** (gear icon).
2. Go to **Resources** -> **Advanced**.
3. Increase **Memory** (we recommend at least 16GB for stable training).
4. Increase **Swap** to 4GB or more.
5. Click **Apply & Restart**.

**For Linux / CLI Users:**

On Linux, Docker usually has access to all system memory unless limited. If you encounter issues, ensure you are providing enough shared memory (essential for multi-processing in nnU-Net) and avoiding hard limits.

```bash
# Add these flags to your 'docker run' command if needed:
docker run --shm-size=8gb \        # Increase shared memory (v. important)
           --memory=16g \          # Set a limit if host is unstable
           --memory-swap=20g \     # Total limit including swap
           ...
```

### Device Not Found

**Problem:** Can't use GPU for training.

```
RuntimeError: No CUDA-capable device is detected
```

**Solution:**

1. Check CUDA availability:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```
2. If False, train on CPU:
   ```bash
   uv run invoke train --device cpu
   ```
3. On Mac with M chip, use MPS:
   ```bash
   uv run invoke train --device mps
   ```

### Training Extremely Slow

**Problem:** Training takes forever.

**Solution:**

1. **Use GPU:**
   ```bash
   uv run invoke train --device cuda  # or mps if using Mac with M chip
   ```
2. **Check you have preprocessed data:**
   ```bash
   ls nnUNet_preprocessed/Dataset101_DroneSeg/
   ```
3. **Monitor system resources:**
   ```bash
   htop  # or Activity Monitor on macOS
   ```

---

## API Issues

### Port Already in Use

**Problem:** Can't start API server.

```
ERROR: [Errno 48] Address already in use
```

**Solution:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Restart it
uv run invoke app
```

### Module Not Found Error

**Problem:** API or scripts refer to the project package but can't find it.

```
ModuleNotFoundError: No module named 'dtu_mlops_111'
```

**Solution:**

1. Reinstall dependencies:
   ```bash
   uv sync --reinstall
   ```
2. If it persists, install in editable mode manually:
   ```bash
   uv pip install -e .
   ```

### Model Not Found

**Problem:** API can't find model checkpoint.

```
FileNotFoundError: nnUNet_results/Dataset101_DroneSeg/.../checkpoint_final.pth
```

**Solution:**

1. Download pre-trained model:
   ```bash
   uv run invoke download-models
   ```
2. Or train your own:
   ```bash
   uv run invoke train
   ```
3. Verify checkpoint exists:
   ```bash
   find nnUNet_results/ -name "checkpoint*.pth"
   ```

### Prediction Fails

**Problem:** API returns error on prediction.

```
422 Unprocessable Entity
```

**Solution:**

1. Check image format (should be PNG or JPG)
2. Verify file is uploaded correctly in request:
   ```bash
   curl -X POST "http://localhost:8000/predict/" \
     -F "data=@/path/to/image.png"
   ```
3. Check API logs for detailed error
4. Run integration tests:
   ```bash
   uv run pytest tests/integrationtests/test_apis.py
   ```
5. Test with a sample image from the dataset

---

## Docker Issues

### Permission Denied

**Problem:** Docker build or run fails with permission error.

```
permission denied while trying to connect to Docker daemon
```

**Solution:**

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo
sudo docker build ...
```

### GPU Not Available in Docker

**Problem:** Docker container can't access GPU.

**Solution:**

1. Install NVIDIA Container Toolkit:
   ```bash
   # Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```
2. Verify Docker can see GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
3. Ensure you use `--gpus all` flag:
   ```bash
   docker run --gpus all ...
   ```

### Build Context Too Large

**Problem:** Docker build is very slow.

```
Sending build context to Docker daemon  5.234GB
```

**Solution:**

1. Check `.dockerignore` file exists
2. Add large directories to `.dockerignore`:
   ```
   data/
   .venv/
   nnUNet_preprocessed/
   nnUNet_results/
   wandb/
   .git/
   ```

---

## DVC Issues

### DVC Pull Fails

**Problem:** Can't download data from DVC.

```
ERROR: failed to pull data from remote
```

**Solution:**

1. Check you have access to GCS bucket (team members only)
2. Verify you are logged into Google Cloud with Application Default Credentials (ADC):
   ```bash
   uv run gcloud auth application-default login
   ```
3. Ensure the correct project is set:
   ```bash
   gcloud config set project your-project-id
   ```
4. Force re-pull:
   ```bash
   uv run dvc pull --force
   ```

### DVC Push Fails

**Problem:** Can't push data to DVC remote.

```
ERROR: failed to push data to remote - permission denied
```

**Solution:**

1. Check you have write permissions to GCS bucket.
2. Re-authenticate with ADC:
   ```bash
   uv run gcloud auth application-default login
   ```

---

## Testing Issues

### Tests Fail

**Problem:** Pytest tests fail.

**Solution:**

1. Ensure dependencies are installed:
   ```bash
   uv sync
   ```
2. Check test requirements:
   ```bash
   uv run pytest tests/ -v
   ```
3. Run specific failing test with more output:
   ```bash
   uv run pytest tests/integrationtests/test_apis.py::test_function -vv -s
   ```

### Import Errors in Tests

**Problem:** Tests can't import modules.

```
ModuleNotFoundError: No module named 'dtu_mlops_111'
```

**Solution:**

1. Install package in editable mode:
   ```bash
   uv sync
   ```
2. Verify pythonpath in `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   pythonpath = ["src"]
   ```

---

## Documentation Issues

### MkDocs Build Fails

**Problem:** Documentation won't build.

```
Error: Config file 'mkdocs.yaml' does not exist
```

**Solution:**

```bash
# Ensure you're running from project root
cd /path/to/DTU_MLOps_111

# Check config file exists
ls docs/mkdocs.yaml

# Build with explicit config path
uv run mkdocs build --config-file docs/mkdocs.yaml
```

### Missing Plugin

**Problem:** MkDocs complains about missing plugin.

```
Error: Plugin 'mkdocstrings' not found
```

**Solution:**

```bash
# Install mkdocs dependencies
uv add --dev mkdocs-material mkdocstrings mkdocstrings-python

# Or sync existing dependencies
uv sync
```

---

## General Tips

### Check Logs

Most tools provide verbose logging:

```bash
# UV with debug output
uv -v sync

# Invoke with echo
uv run invoke command --echo

# Python with debug
PYTHONVERBOSE=1 uv run python script.py
```

### Clean Start

If all else fails, start fresh:

```bash
# Remove virtual environment
rm -rf .venv

# Remove UV cache
uv cache clean

# Reinstall everything
uv sync --reinstall

# Reload environment
source .env
```

### Get Help

If you're still stuck:

1. Check error messages carefully
2. Search GitHub issues
3. Check nnU-Net documentation: [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
4. Contact team members (see homepage)

---

## Still Having Issues?

If your problem isn't listed here:

1. Check the [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation)
2. Review error messages and stack traces
3. Enable verbose/debug logging
4. Contact the project team
