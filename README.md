# BirdClassification
Bird Song Classification for EG4578 group project 

Install the required dependencies using pip: 
    ```bash
    pip install -r requirements.txt
    ```

## Notebooks

- `Prediction.ipynb`: Demonstrates how to use the classifier to predict the bird class from both an audio file or a spectrogram and map the bird ID to a bird name.

- `ModelTraining.ipynb`: Shows how to train the model on the dataset, visualized the loss over training, and tests the model's performance on the test set.

- `FilterTesting.ipynb`: Tests the model performance on the filtered audio recordings that contain wind for more realistic real-world performance estimation.

- `DataVisualization.ipynb`: Inspects the dataset and visualizes the data to get a better understanding of the data.

- `DataPreprocessing.ipynb`: Used to split each audio file in the dataset into 5-second clips.

## Models

- `model1.pth`: First model trained on original data.

- `model2.pth`: Second model trained on original data.

- `model3.pth`: Trained on updated dataset (containing examples of filtered wind noise).

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset/data).
