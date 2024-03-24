import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from textprocess import TextProcessor  # Assuming this is your actual text processing class from textprocess.py
from train import ModelTrainer  # Assuming this is your model training class from train.py

class MLModelPipeline:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.text_processor = TextProcessor()

    def run_pipeline(self, model, param_grid, model_name):
        """
        Runs the machine learning pipeline for a given model and parameters and plots the results.
        """
        # Preprocess and extract features
        processed_data = self.raw_data.drop(columns=['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'])
        processed_data['tweet'] = self.text_processor.preprocess(processed_data['tweet'])
        processed_data.rename(columns={'tweet': 'preprocessed_tweet'}, inplace=True)
        data_features = self.text_processor.extract_features(processed_data['preprocessed_tweet'])

        # Split the data
        labels = self.raw_data['class']  # Make sure 'class' is the correct label column in your dataset
        X_train, y_train, X_val, y_val, X_test, y_test = self.text_processor.split_data(data_features, labels)

        # Train and tune the model
        model_trainer = ModelTrainer(model)
        model_trainer.train_model(X_train, y_train)
        model_trainer.tune_hyperparameters(param_grid, X_val, y_val)
        evaluation_metrics = model_trainer.evaluate_model(X_test, y_test)

        # Display results
        self.display_results(model_name, evaluation_metrics, evaluation_metrics['confusion_matrix'])

    def display_results(self, model_name, evaluation_metrics, conf_matrix):
        """
        Displays evaluation metrics and confusion matrix for a given model.
        """
        print(f"{model_name} Evaluation Metrics:")
        for metric, value in evaluation_metrics.items():
            print(f'{metric}: {value}')
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'{model_name} Confusion Matrix')
        plt.show()
