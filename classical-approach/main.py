from run_model import MLModelPipeline
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load your dataset
    data = pd.read_csv('cleaned_data_sw.csv')  # Replace with actual data loading method

    # Initialize the ML pipeline
    ml_pipeline = MLModelPipeline(data)

    # Define parameters and run the pipeline for SVM
    svm_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    ml_pipeline.run_pipeline(SVC(), svm_params, 'SVM')

if __name__ == "__main__":
    main()

