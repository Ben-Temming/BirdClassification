# BirdClassification
Classifying bird species from their birdsong.

Install the required dependencies using pip: 
    `pip install -r requirements.txt`

## Notebooks

- `Prediction.ipynb`: Demonstrates how to use the classifier to predict the bird class from both an audio file or a spectrogram and map the bird ID to a bird name.

- `ModelTraining.ipynb`: Shows how to train the model on the dataset, visualises the loss over training, and tests the model's performance on the test set.

- `FilterTesting.ipynb`: Tests the model performance on the filtered audio recordings that contain wind for more realistic real-world performance estimation.

- `DataVisualization.ipynb`: Inspects the dataset and visualizes the data to get a better understanding of the data.

- `DataPreprocessing.ipynb`: Used to split each audio file in the dataset into 5-second clips.

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset/data).
