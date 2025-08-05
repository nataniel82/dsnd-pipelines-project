
# StyleSense Customer Review Prediction Pipeline

This project develops a complete machine learning pipeline to predict whether a customer will recommend a product based on their review data. The model handles mixed data types (numerical, categorical, and text) and has been optimized for the best possible performance.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Dependencies

The project uses the following Python libraries. Make sure you have them installed in your environment.

*   pandas
*   numpy
*   scikit-learn
*   spaCy
*   matplotlib
*   seaborn
*   wordcloud
*   joblib
*   Jupyter Notebook / Jupyter Lab

### Installation

1.  **Clone the repository**

    Clone the project repository to your local machine. If you received the files, ensure they are in a single project folder.
    ```bash
    git clone https://github.com/udacity/dsnd-pipelines-project.git
    cd dsnd-pipelines-project
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies**

    Install all required libraries using a `requirements.txt` file.
    ```bash
    python -m pip install -r requirements.txt
    ```

4.  **Download the spaCy model**

    The project requires the 'en_core_web_sm' model from spaCy for natural language processing.
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Launch Jupyter Notebook**

    Once the installation is complete, you can launch Jupyter to run the project notebook or execute the Python script.
    ```bash
    jupyter notebook
    ```

## Usage and Testing

The main project file is the Python script (`pipeline_project.py`) or a Jupyter Notebook.

### Script/Notebook Structure

The code is divided into the following sections:

1.  **Setup and Data Loading**: Imports libraries, loads data, and performs initial preparation.
2.  **Exploratory Data Analysis (EDA)**: Visualizations to understand the dataset.
3.  **Pipeline Construction**: Defines and combines the preprocessing pipelines.
4.  **Training and Evaluation**: Trains the initial model and evaluates it.
5.  **Fine-Tuning**: Optimizes hyperparameters with `RandomizedSearchCV`.
6.  **Feature Importance**: Analyzes the most influential features.
7.  **Saving the Model**: Serializes the final model into a `.pkl` file.

### Model Evaluation

The model is evaluated using the test set, which is never used during the training or fine-tuning phase. The evaluation metrics used include:
*   **Accuracy**: The overall percentage of correct predictions.
*   **Precision**: The model's ability to not label a negative instance as positive.
*   **Recall**: The model's ability to find all positive instances.
*   **F1-Score**: The harmonic mean of Precision and Recall, useful for imbalanced data.

A `classification_report` is shown for both the baseline and the optimized models.

## Deliverables

*   **Python Script/Notebook**: A complete notebook (`.ipynb`) documenting every step of the process, from data cleaning to final evaluation.
*   **Saved Model**: A `fashion_recommendation_pipeline.pkl` file containing the trained and optimized Scikit-learn pipeline, ready to be used for predictions on new data.

## Built With

*   [Python](https://www.python.org/) - The programming language used.
*   [Scikit-learn](https://scikit-learn.org/) - The machine learning library.
*   [Pandas](https://pandas.pydata.org/) - The data manipulation and analysis library.
*   [spaCy](https://spacy.io/) - The library for advanced natural language processing.
*   [Jupyter Notebook](https://jupyter.org/) - The interactive development environment.

