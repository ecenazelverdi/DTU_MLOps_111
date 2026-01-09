# dtu_mlops_111

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
# DTU_MLOps_111


## Project description
Our project aims to perform binary image classification on satellite images to determine whether or not they contain ships, as well as using a semantic segmentation segmentation mask to identify the precise location of ships within the images.  We aim for near 100% accuracy on classification.  For segmentation, we aim for an IoU metric of 0.7 or better and Dice Coeffeicient of 0.8 or better.

An additional goal is to deliver a reproducible, well-dcoumented, containerized ML pipeline.  This will include data downloading, preprocessing, model training, evulation, and environment management using uv and docker.  The final deliverable should be easy for other users to run end-to-end with minimal setup.

Here you can find the [Shipping imagery Dataset]
(https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) from Kaggle, which contains 4000 satellite images from the bays and oceans of the California area.  Images include a mixture of large cargo ships, small civilian boats, ships partically occluded by wakes or waves, and background-only scenes with no ships.

We expect to use a CNN for the image classification and a U-net for the segmentation.  We will implement our models in the pytorch library, potentially leveraging transfer-learning for classification.  A U-Net architecture is chosen for the project because it performs well with small objects, preserves spacial detail via skip connections, and performs well with low-data availability


See below for instructions on how to run.

To download and preprocess the Kaggle Dataset
```
uv run python src/dtu_mlops_111/data.py --download
```
Note: to download the data, a kaggle api key is required

To preprocess the dataset (if already downloaded)
```
uv run python src/dtu_mlops_111/data.py
```



