# dtu MLOps Project: Semantic segmentation for drone imagery

Team 111

## Project Description

Our project aims to perform 5-class image segmentation on [drone imagery](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset/data), using a semantic segmentation mask to identify the precise location of obstacles, water, soft-surfaces, moving-objects, and landing-zones.

We train a Computer Vision model on this segmentation dataset to detect five different classes of objects from drone imagery of urban scenes.

The trained model should enhance the safety of autonomous drone flights and landings in urban areas by distinguishing different kinds of obstacles from landing-zones.

We expect to use a CNN for the image classification and a U-net for the segmentation. We will implement our models in the pytorch library, potentially leveraging transfer-learning for classification. A U-Net architecture is chosen for the project because it performs well with small objects, preserves spacial detail via skip connections, and performs well with low-data availability

### How to run

See below for instructions on how to run.

To download and preprocess the Kaggle Dataset, run

```
uv run python src/dtu_mlops_111/data.py main
```

### To Download and Configure the Dataset
_Note_: kaggle api key and kaggle username key are required. the `.env.example` file provides an example of how to structure your own `.env` file and fill with with your personal info.  After updating and refreshing your .env file, run:
```
uv run invoke download-data
uv run invoke export-data
```

### To Preprocess the Data
_Note_: nnU-Net makes use of specific [_environment variables_](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) to locate data in the project. `.env.example` has the appropriate predefined structure.  After updating your .env and refresing with _environment variables_ for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`, run:
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

### Production

To install the package for production (immutable, copying files), run:

```bash
pip install .
```

### To Run API

Make sure current package is installed in either Development or production. Afterwards, run

```
uv run invoke app
```

### To use API

Ensure model checkpoint is available. This can be done by pulling from our Cloud bucket. Assuming you have access to it, simply run
```
uv run dvc nnUNet_results.dvc --no-run-cache
```
This will take a couple of minutes. Afterwards, you should see that the `nnUNet_results/` is present and has the latest version of the model checkpoint (**Note: you might need to add the `--force` option in if you already have a local model checkpoint).

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


## nnU-Net

### Setup
Running [nnU-Net models](https://github.com/MIC-DKFZ/nnUNet) segmentation models requires a specific file and data structuring. 

Running

```
uv run python src/dtu_mlops_111/data.py nnunet-export
```

will create an additional folder necessary for the use of [nnU-Net models](https://github.com/MIC-DKFZ/nnUNet). This folder is structured as follows:

```
nnUNet_raw/
└── Dataset101_DroneSeg
    ├── imagesTr
    └── labelsTr
```
### Preprocessing

 nnU-Net makes use of specific [*environment variables*](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) to locate data in the project. `.env.example` has the appropriate predefined structure.

After setting up *environment variables* for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`, you are ready for 
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


### Training.

[ ] TODO

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
