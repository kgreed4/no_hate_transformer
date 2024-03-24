from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, model):
        """
        Initializes the ModelTraining class.

        Parameters:
        - model: The machine learning model to be used.
        """
        self.model = model

    def train_model(self, X_train, y_train):
        """
        Trains the machine learning model using the provided training data.

        Parameters:
        - X_train: The features of the training data.
        - y_train: The labels of the training data.
        """
        # Fit the model to the training data
        self.model.fit(X_train, y_train)

    def tune_hyperparameters(self, param_grid, X_val, y_val):
        """
        Tunes hyperparameters for the model using GridSearchCV on the validation set.

        Parameters:
        - param_grid: The hyperparameter settings to try as a dictionary.
        - X_val: The features of the validation data.
        - y_val: The labels of the validation data.
        """
        # Initialize the GridSearchCV object
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        
        # Fit GridSearchCV to the validation data
        grid_search.fit(X_val, y_val)
        
        # Update the model with the best estimator
        self.model = grid_search.best_estimator_

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the trained machine learning model using the test set.

        Parameters:
        - X_test: The features of the test data.
        - y_test: The labels of the test data.

        Returns:
        - A dictionary containing evaluated metrics: accuracy, precision, recall, and F1-score, and a confusion matrix.
        """
        # Make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Compile and return the metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }

    def execute_pipeline(self, X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
        """
        Executes the training, tuning, and evaluation pipeline.

        Parameters:
        - X_train, y_train: Training data features and labels.
        - X_val, y_val: Validation data features and labels.
        - X_test, y_test: Test data features and labels.
        - param_grid: Hyperparameter settings to try.
        """
        self.train_model(X_train, y_train)
        self.tune_hyperparameters(param_grid, X_val, y_val)
        return self.evaluate_model(X_test, y_test)

