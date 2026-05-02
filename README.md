# ai-Olympiad Land Classification

This project is a fastai-based image classification workflow for the Zindi land/road-segment task. The notebook `land.ipynb` loads `.tif` images, trains a `resnet18` classifier on `Train.csv`, and writes predictions for `Test.csv` to `submission1.csv`.

## Project Files

The workspace currently contains:

- `land.ipynb` - main training and inference notebook
- `submission1.csv` - example submission produced by the notebook
- `test/Train.csv` - training labels and image IDs
- `test/Test.csv` - test image IDs
- `test/StarterNB_RoadSegment.ipynb` - reference notebook

The notebook expects the image archive to be unpacked into an `Images/` directory containing `.tif` files.

## Requirements

- Python 3.9 or newer
- pandas
- fastai
- fastcore
- torch and torchvision
- Jupyter Notebook or Google Colab

If you are using Google Colab, the notebook can upload files through `google.colab.files`. If you are running locally, place the CSV files and `Images.zip` in the working directory and unzip them before training.

## Setup

Install the Python packages:

```bash
pip install --upgrade fastcore fastai pandas torch torchvision
```

If you are running in Colab, you can install the notebook dependencies from a cell:

```python
!pip install -q --upgrade fastcore fastai
```

## Data Preparation

1. Download the competition data.
2. Make sure you have the image archive, usually `Images.zip`.
3. Place `Train.csv` and `Test.csv` in the working directory, or update the notebook paths if you keep them in `test/`.
4. Unzip the images into an `Images/` folder.

Example:

```bash
unzip -q Images.zip
```

## End-to-End Workflow

The notebook follows these steps:

1. Load the training CSV.
2. Build image dataloaders from the `Image_ID` column.
3. Train a `resnet18` image classifier.
4. Build a test dataloader from the test image IDs.
5. Export predicted probabilities for the positive class to `submission1.csv`.

## Complete Notebook Code

The code below reproduces the notebook flow in a single script-style cell sequence.

```python
# Optional in Colab: upload files manually
from google.colab import files
files.upload()

import pandas as pd
from fastai.vision.all import *

# Unpack images
!unzip -q Images.zip

# Load data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Create dataloaders
dls = ImageDataLoaders.from_df(
    train,
    path='Images',
    suff='.tif',
    seed=42
)

# Train model
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5, 1e-3)

# Create test dataloader and predict
tdl = learn.dls.test_dl(
    test['Image_ID'].map(lambda s: 'Images/' + s + '.tif').values
)
preds = learn.get_preds(dl=tdl)

# Save the probability for class 1
test['Target'] = [float(p[1]) for p in preds[0]]
test.to_csv('submission1.csv', index=False)

# Optional in Colab: download the submission
files.download('submission1.csv')
```

## Notebook Breakdown

### 1. Install and extract data

The notebook first installs fastai/fastcore and extracts `Images.zip`.

### 2. Load training data

`Train.csv` is read into a pandas DataFrame. The expected training schema is:

```text
Image_ID,Target
ID_0073qfb8,0
ID_00gy3vH2,1
```

### 3. Build fastai dataloaders

The notebook uses `ImageDataLoaders.from_df(train, path='Images', suff='.tif')` so each image is read from `Images/<Image_ID>.tif`.

### 4. Train the model

The model is a pretrained `resnet18` created with `vision_learner` and trained with `fine_tune(5, 1e-3)`.

### 5. Predict on the test set

The notebook loads `Test.csv`, builds a test dataloader from the image paths, runs inference, and stores the probability of class 1 in the `Target` column.

### 6. Write the submission

Predictions are saved to `submission1.csv` with the same `Image_ID` values from `Test.csv` and a numeric `Target` column.

## Expected Data Format

Training CSV:

```text
Image_ID,Target
ID_0073qfb8,0
ID_00gy3vH2,1
```

Test CSV:

```text
Image_ID
ID_01c6i2wd
ID_03sPqBLY
```

Submission CSV:

```text
Image_ID,Target
ID_01c6i2wd,0.6270752549
```

## Output

After training and prediction, the notebook produces:

- `submission1.csv` - competition submission file

## Notes

- The notebook is written for a Colab-style workflow, but it can be run locally if the data files are placed in the working directory.
- If your `Train.csv` and `Test.csv` files are inside `test/`, update the notebook paths from `Train.csv` and `Test.csv` to `test/Train.csv` and `test/Test.csv`.
- The classifier predicts a probability for the positive class; the notebook stores that value directly in the `Target` column.

## Troubleshooting

- If `Images.zip` is missing, the image paths will fail during dataloader creation.
- If you see file-not-found errors, confirm that `Images/`, `Train.csv`, and `Test.csv` are in the paths referenced by the notebook.
- If you are not in Colab, remove the `google.colab.files.upload()` and `files.download()` calls and use local file paths instead.

## License

No license file was provided in the workspace.