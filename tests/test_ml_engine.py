import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Assuming ml_engine is available in the Python path due to conftest.py
from ml_engine import AnomalyEngine

@pytest.fixture
def mock_mlflow():
    with patch('ml_engine.mlflow') as mock:
        yield mock

@pytest.fixture
def mock_config():
    with patch('ml_engine.config') as mock:
        mock.MLFLOW_URI = "sqlite:///mock_mlflow.db"
        mock.EXPERIMENT_NAME = "Mock_Experiment"
        mock.N_ESTIMATORS = 10
        mock.CONTAMINATION = 0.05
        mock.MODEL_PATH = "mock_models/mock_model.pkl"
        yield mock

@pytest.fixture
def dummy_data():
    # Generate dummy normal data
    np.random.seed(42)
    return np.random.normal(loc=0, scale=1, size=(100, 5))

def test_anomaly_engine_initialization(mock_mlflow, mock_config):
    """Test that the AnomalyEngine initializes correctly with MLflow tracking."""
    # Ensure get_experiment_by_name returns None to trigger creation
    mock_mlflow.get_experiment_by_name.return_value = None
    
    engine = AnomalyEngine()
    
    # Assert side effects on mlflow
    mock_mlflow.set_tracking_uri.assert_called_once_with(mock_config.MLFLOW_URI)
    mock_mlflow.get_experiment_by_name.assert_called_once_with(mock_config.EXPERIMENT_NAME)
    mock_mlflow.create_experiment.assert_called_once_with(mock_config.EXPERIMENT_NAME)
    mock_mlflow.set_experiment.assert_called_once_with(mock_config.EXPERIMENT_NAME)
    
    assert engine.model is None

@patch('ml_engine.os.makedirs')
@patch('ml_engine.pickle.dump')
@patch('builtins.open', new_callable=MagicMock)
def test_train_model(mock_open, mock_pickle_dump, mock_makedirs, mock_mlflow, mock_config, dummy_data):
    """Test the training process of the IsolationForest model."""
    engine = AnomalyEngine()
    
    # Train the model
    model = engine.train(dummy_data)
    
    # Verify model is created and trained
    assert engine.model is not None
    assert engine.model == model
    
    # Assert MLflow operations
    mock_mlflow.start_run.assert_called_once_with(run_name="Satellite_Retrain")
    mock_mlflow.sklearn.log_model.assert_called_once_with(model, "model")
    
    # Assert persistence operations
    mock_makedirs.assert_called_once_with("models", exist_ok=True)
    mock_open.assert_called_once_with(mock_config.MODEL_PATH, "wb")
    mock_pickle_dump.assert_called_once()

def test_predict_model(mock_mlflow, mock_config, dummy_data):
    """Test that predictions work correctly after training."""
    engine = AnomalyEngine()
    engine.train(dummy_data)
    
    # Predict on new data
    new_data = np.random.normal(loc=0, scale=1, size=(10, 5))
    preds, scores = engine.predict(new_data)
    
    assert len(preds) == 10
    assert len(scores) == 10
    # Predictions are either 1 (inlier) or -1 (outlier) for Isolation forest
    assert set(np.unique(preds)).issubset({1, -1})
