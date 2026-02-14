# Import necessary libraries
import unittest
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class TestBreastCancerModel(unittest.TestCase):
    """Test cases for Breast Cancer Detection Model using XGBoost"""
    
    @classmethod
    def setUpClass(cls):
        """Load the dataset and train the model once for all tests"""
        cls.breast_cancer = load_breast_cancer()
        cls.X, cls.y = cls.breast_cancer.data, cls.breast_cancer.target
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )
        
        # Train the model
        cls.model = XGBClassifier(n_estimators=100, random_state=42, 
                                  eval_metric='logloss')
        cls.model.fit(cls.X_train, cls.y_train)
    
    def test_dataset_loaded(self):
        """Test that the Breast Cancer dataset is loaded correctly"""
        self.assertEqual(self.X.shape[0], 569)  # Number of samples
        self.assertEqual(self.X.shape[1], 30)   # Number of features
        self.assertEqual(len(self.y), 569)      # Labels length
    
    def test_dataset_binary_classification(self):
        """Test that the dataset has binary classification labels (0 and 1)"""
        unique_labels = np.unique(self.y)
        self.assertTrue(len(unique_labels) == 2)
        self.assertTrue(set(unique_labels) == {0, 1})
    
    def test_train_test_split(self):
        """Test that train/test split is correct"""
        self.assertEqual(len(self.X_train) + len(self.X_test), 569)
        self.assertAlmostEqual(len(self.X_test) / 569, 0.2, places=1)
    
    def test_model_training(self):
        """Test that the model is trained successfully"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.n_estimators, 100)
    
    def test_model_prediction_shape(self):
        """Test that model predictions have correct shape"""
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_model_prediction_values(self):
        """Test that model predictions are binary (0 or 1)"""
        predictions = self.model.predict(self.X_test)
        unique_predictions = np.unique(predictions)
        self.assertTrue(all(pred in [0, 1] for pred in unique_predictions))
    
    def test_model_prediction_probability(self):
        """Test that model can predict probabilities"""
        probabilities = self.model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertEqual(probabilities.shape[1], 2)
        # Check that probabilities sum to 1
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
    
    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy on test set"""
        accuracy = self.model.score(self.X_test, self.y_test)
        self.assertGreater(accuracy, 0.90)  # XGBoost should achieve >90% accuracy
        print(f"Model Accuracy: {accuracy:.2%}")
    
    def test_model_serialization(self):
        """Test that model can be saved and loaded"""
        model_path = 'test_breast_cancer_model.pkl'
        joblib.dump(self.model, model_path)
        loaded_model = joblib.load(model_path)
        
        # Test that loaded model makes same predictions
        original_predictions = self.model.predict(self.X_test)
        loaded_predictions = loaded_model.predict(self.X_test)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_model_handles_single_sample(self):
        """Test that model can predict on a single sample"""
        single_sample = self.X_test[0:1]
        prediction = self.model.predict(single_sample)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1])


if __name__ == '__main__':
    unittest.main()
