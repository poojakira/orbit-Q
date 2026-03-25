import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch, MagicMock

from orbit_q.engine.ml_engine import AnomalyEngine


@pytest.fixture
def mock_mlflow():
    with patch('orbit_q.engine.ml_engine.mlflow') as mock:
        yield mock


@pytest.fixture
def mock_config():
    with patch('orbit_q.engine.ml_engine.config') as mock:
        mock.MLFLOW_URI = "sqlite:///mock_mlflow.db"
        mock.EXPERIMENT_NAME = "Mock_Experiment"
        mock.N_ESTIMATORS = 10
        mock.CONTAMINATION = 0.05
        mock.MODEL_PATH = "mock_models/mock_model.pkl"
        yield mock


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    return np.random.normal(loc=0, scale=1, size=(100, 5))


def test_anomaly_engine_initialization(mock_mlflow, mock_config):
    """AnomalyEngine should configure MLflow and start with no trained models."""
    mock_mlflow.get_experiment_by_name.return_value = None

    engine = AnomalyEngine()

    mock_mlflow.set_tracking_uri.assert_called_once_with(mock_config.MLFLOW_URI)
    mock_mlflow.get_experiment_by_name.assert_called_once_with(mock_config.EXPERIMENT_NAME)
    mock_mlflow.create_experiment.assert_called_once_with(mock_config.EXPERIMENT_NAME)
    mock_mlflow.set_experiment.assert_called_once_with(mock_config.EXPERIMENT_NAME)

    # Ensemble engine starts with no trained sub-models
    assert engine.iso_model is None
    assert engine.ae_model is None


@patch('orbit_q.engine.ml_engine.os.makedirs')
@patch('orbit_q.engine.ml_engine.pickle.dump')
@patch('builtins.open', new_callable=MagicMock)
def test_train_model(mock_open, mock_pickle_dump, mock_makedirs, mock_mlflow, mock_config, dummy_data):
    """Training should produce both IsolationForest and Autoencoder sub-models."""
    engine = AnomalyEngine()

    iso_model, ae_model = engine.train(dummy_data)

    assert engine.iso_model is not None
    assert engine.ae_model is not None
    assert engine.iso_model is iso_model

    # MLflow run with ensemble name
    mock_mlflow.start_run.assert_called_once_with(run_name="Satellite_Ensemble_Retrain")
    mock_mlflow.sklearn.log_model.assert_called_once_with(iso_model, "iso_model")

    # Persistence
    mock_makedirs.assert_called_once_with("models", exist_ok=True)
    mock_open.assert_called_once_with(mock_config.MODEL_PATH, "wb")
    mock_pickle_dump.assert_called_once()


@patch('orbit_q.engine.ml_engine.os.makedirs')
@patch('orbit_q.engine.ml_engine.pickle.dump')
@patch('builtins.open', new_callable=MagicMock)
def test_predict_model(mock_open, mock_pickle_dump, mock_makedirs, mock_mlflow, mock_config, dummy_data):
    """Ensemble predictions should return labels and scores for all input samples."""
    engine = AnomalyEngine()
    engine.train(dummy_data)

    new_data = np.random.normal(loc=0, scale=1, size=(10, 5))
    preds, scores = engine.predict(new_data)

    assert len(preds) == 10
    assert len(scores) == 10
    assert set(np.unique(preds)).issubset({1, -1})
