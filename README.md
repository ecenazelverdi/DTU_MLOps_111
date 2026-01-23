# dtu MLOps Project: Semantic segmentation for drone imagery

Team 111

📚 **[Live Documentation](https://ecenazelverdi.github.io/DTU_MLOps_111/)**

## Project Description

Our project aims to perform 5-class image segmentation on [drone imagery](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset/data), using a semantic segmentation mask to identify the precise location of obstacles, water, soft-surfaces, moving-objects, and landing-zones.

We train a the nnUNet model (https://github.com/MIC-DKFZ/nnUNet) on this segmentation dataset to detect five different classes of objects from drone imagery of urban scenes.

The trained model should enhance the safety of autonomous drone flights and landings in urban areas by distinguishing different kinds of obstacles from landing-zones.

We expect to use a CNN for the image classification and a U-net for the segmentation. We will implement our models in the pytorch library, potentially leveraging transfer-learning for classification. A U-Net architecture is chosen for the project because it performs well with small objects, preserves spacial detail via skip connections, and performs well with low-data availability

## How to Use

To run the project on your machine, use the following commands

### Installation

clone this repository and run

```
uv sync
```

### To Download and Configure the Dataset

_Note_: kaggle api key and kaggle username key are required. the `.env.example` file provides an example of how to structure your own `.env` file and fill with with your personal info. After updating and refreshing your .env file, run:

```
uv run invoke download-data
uv run invoke export-data
```

### To Preprocess the Data

_Note_: nnU-Net makes use of specific [_environment variables_](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) to locate data in the project. `.env.example` has the appropriate predefined structure. After updating your .env and refresing with _environment variables_ for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`, run:

```
uv run invoke preprocess
```

### To Train the Model

```
uv run invoke train
```

### To Run Tests

```
uv run invoke test
```

### To Run API

```
uv run invoke app
```

### To Run with BentoML

For high-performance serving, you can use BentoML:

```
uv run invoke bento-serve
```

For dockerized BentoML instructions, see [Docker Workflow](#docker-workflow) or full [documentation](docs/source/commands.md).

### To Download and Configure the Dataset

_Note_: kaggle api key and kaggle username key are required. the `.env.example` file provides an example of how to structure your own `.env` file and fill with with your personal info. After updating and refreshing your .env file, run:

```
uv run invoke download-data
uv run invoke export-data
```

### To Preprocess the Data

_Note_: nnU-Net makes use of specific [_environment variables_](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) to locate data in the project. `.env.example` has the appropriate predefined structure. After updating your .env and refresing with _environment variables_ for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`, run:

```
uv run invoke preprocess
```

### To Train the Model

```
uv run invoke train
```

### To Run Tests

```
uv run invoke test
```

### Development

To install the package in editable mode (changes reflected instantly), run:

```bash
uv pip install -e .
```

To use the recommended pre-commit hooks for this repository, run:

```
To use pre-commit, run
```

### Production

To install the package for production (immutable, copying files), run:

```bash
pip install .
```

## Further Information

### Dataset structure

To install the package in editable mode (changes reflected instantly), run:

```bash
uv pip install -e .
```

### Production

To install the package for production (immutable, copying files), run:

```bash
pip install .
```

### To Run FastAPI

Make sure current package is installed in either Development or production. Afterwards, run

```
uv run invoke app
```

### To use FastAPI

**Deployed Version:**

You can access the live API here: [https://model-api-32512441443.europe-west1.run.app](https://model-api-32512441443.europe-west1.run.app)

Example request to deployed API:

```bash
curl --location 'https://model-api-32512441443.europe-west1.run.app/predict/' \
--form 'data=@"<YOUR_PATH_TO_IMAGE>/<IMAGE_NAME>.png"' \
```

### BentoML

This BentoML deployment exposes a `/predict_base64` endpoint that accepts an input image as **Base64-encoded bytes** inside a JSON payload.

**Quick start:**

```bash
# Build BentoML serving container
uv run invoke docker-build-bento

# Run serving container (port 8080)
uv run invoke docker-run-bento
```


### Payload Generation for BentoML
The following script reads a local image file, encodes it into Base64, and writes a `payload.json` file in the expected request format:

```bash
python - <<'PY'
import base64, json
img_path = "sample_image.jpg"  
b64 = base64.b64encode(open(img_path, "rb").read()).decode()
payload = {"req": {"image_b64": b64, "content_type": "image/jpeg"}}
with open("payload.json", "w") as f:
    json.dump(payload, f)
print("saved payload.json")
PY
```


### Curl Request
Once the payload has been created, send it to the deployed BentoML endpoint using `curl`:

```bash
curl -s -o resp.json -w "\nHTTP %{http_code}\n" \
  -X POST "https://drone-seg-32512441443.europe-north1.run.app/predict_base64" \
  -H "Content-Type: application/json" \
  --data-binary @payload.json
```

The response will be saved into `resp.json`. If the request succeeds, the HTTP status code should be `200`.

**Results:** Service available locally at http://localhost:8080/ or live in https://drone-seg-32512441443.europe-north1.run.app

## Docker Workflow

This project provides a complete Docker-based pipeline for training and inference:

```
┌─────────────────┐
│  Training       │
│  Container      │──┐
│  (train.        │  │
│   dockerfile)   │  │
└─────────────────┘  │
         │           │
         ▼           │
  ┌─────────────┐   │ Shared volume:
  │ nnUNet_     │◄──┘ nnUNet_results/
  │ results/    │
  │ ├─Model     │
  │ └─Checkpts  │
  └─────────────┘
         │
         │
         ▼
┌─────────────────┐
│  Inference      │
│  Container      │
│  (inference.    │
│   dockerfile)   │
└─────────────────┘
         │
         ▼
  ┌─────────────┐
  │ inference_  │
  │ outputs/    │
  │ ├─masks     │
  │ └─results   │
  └─────────────┘
```

Both containers share the same `nnUNet_results/` folder, enabling seamless workflow!

### How to run

**Local Development:**

Ensure model checkpoint is available. This can be done by pulling from our Cloud bucket. Assuming you have access to it, simply run

```
uv run dvc nnUNet_results.dvc --no-run-cache
```

This will take a couple of minutes. Afterwards, you should see that the `nnUNet_results/` is present and has the latest version of the model checkpoint (\*\*Note: you might need to add the `--force` option in if you already have a local model checkpoint).

For example, to use the API to predict segments for a specific image, run

```
curl --location 'http://127.0.0.1:8000/predict/' \
--form 'data=@"<YOUR_PATH_TO_IMAGE>/<IMAGE_NAME>.png"'
```

To use the recommended pre-commit hooks for this repository, run:

```
To use pre-commit, run
```

## Further Information

### Dataset structure

After downloading, you will find the data is structured in the following way:

```
data/
│
│
└── raw
    └── classes_dataset
        └── classes_dataset
            ├── label_images_semantic
            └── original_images
```

- `original_images` contains 400 png drone images, with format `<id>.png`, in RGB coloring
- `label_images_semantic/` contains the same 400 images, but pixel RGB values have been replaced with the corresponding class RGB values according to the table below:

| Class          | Color                                                                                                                  | R   | G   | B   |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- | --- | --- | --- |
| obstacles      | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(155,38,182);border-radius:3px;"></span>  | 155 | 38  | 182 |
| water          | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(14,135,204);border-radius:3px;"></span>  | 14  | 135 | 204 |
| soft-surfaces  | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(124,252,0);border-radius:3px;"></span>   | 124 | 252 | 0   |
| moving-objects | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(255,20,147);border-radius:3px;"></span>  | 255 | 20  | 147 |
| landing-zones  | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(169,169,169);border-radius:3px;"></span> | 169 | 169 | 169 |

_Note_: Original imagery dataset comes from [TU Graz, IVC](https://ivc.tugraz.at/research-project/semantic-drone-dataset/).

configureing the data with the above commands will create an additional folder necessary for the use of [nnU-Net models](https://github.com/MIC-DKFZ/nnUNet). This folder is structured as follows:

```
nnUNet_raw/
└── Dataset101_DroneSeg
    ├── imagesTr
    └── labelsTr
```

### Project structure

### Preprocessing

nnU-Net makes use of specific [_environment variables_](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) to locate data in the project. `.env.example` has the appropriate predefined structure.

After setting up _environment variables_ for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`, you are ready for
data preprocessing. Make sure your `.env` file is loaded, then run

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

to preprocess, where `DATASET_ID` is `101` in this case.

You should now find a new directory was created with the following structure:

```
nnUNet_preprocessed
└── Dataset101_DroneSeg
    ├── gt_segmentations
    └── nnUNetPlans_2d
```

### Training

For detailed instructions on training the model using Docker, see [DOCKER_TRAINING.md](DOCKER_TRAINING.md).

**Quick start:**

```bash
# Build training container
docker build -f train.dockerfile -t droneseg-training .

# Run training (1 epoch)
docker run --gpus all --ipc=host \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/nnUNet_raw:/app/nnUNet_raw \
  -v $(pwd)/nnUNet_preprocessed:/app/nnUNet_preprocessed \
  -v $(pwd)/nnUNet_results:/app/nnUNet_results \
  droneseg-training
```

**Expected output:** Model checkpoints saved to `nnUNet_results/Dataset101_DroneSeg/nnUNetTrainer_1epoch__nnUNetPlans__2d/fold_0/`

### Inference

For detailed instructions on running inference using Docker, see [DOCKER_INFERENCE.md](DOCKER_INFERENCE.md).

**Quick start (after training):**

```bash
# Prepare input images (convert RGB to nnU-Net format)
mkdir -p images_raw input
cp your_drone_image.jpg images_raw/
python prepare_inference_input.py images_raw/ input/

# Build inference container
docker build -f inference.dockerfile -t droneseg-inference .

# Run inference (uses model from training automatically)
docker run --gpus all --ipc=host \
  -v $(pwd)/input:/input \
  -v $(pwd)/nnUNet_results:/nnUnet_results \
  droneseg-inference

# Create visualizations
python visualize_results.py images_raw/ nnUNet_results/inference_outputs/ visualizations/
```

**Results:** Segmentation masks saved to `nnUNet_results/inference_outputs/`, visualizations in `visualizations/`

> **Note:** Both training and inference containers share the same `nnUNet_results/` folder, enabling seamless workflow from training to inference!


## Contributer Setup

### Optional: pre-commit

To use pre-commit, run

```
uv run pre-commit install
```

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
](https://github.com/ecenazelverdi/DTU_MLOps_111.git)](https://github.com/ecenazelverdi/DTU_MLOps_111.git)
