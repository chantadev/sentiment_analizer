import pytest
import sys
import csv
from pathlib import Path

"""
NOTE: some warnings raised by some deprecations, but it should be an issue. 
venv\Lib\site-packages\datasets\utils\_dill.py:385: DeprecationWarning: co_lnotab is deprecated, use co_lines instead.
    obj.co_lnotab,  # for < python 3.10 [not counted in args]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
"""


# Add the current directory to the path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture
def texts():
    """Example texts to test language identification"""
    return [
        "This is a test sentence in English.",
        "这是一个测试句子，用中文写的。",
        "Este es un ejemplo de texto en español.",
        "これは日本語のテスト文です。",
    ]

import src.data.data_processing as dl
from src.utils.language_processing import identify_language

class TestDataLoader:
    """Tests for the DataLoader functions."""

    def test_identify_language(self, texts): 
        for text in texts:  
            assert identify_language(text) in ["en", "zh-cn", "es", "ja"]
    
    # WORKS: commmented for time improvement
    # def test_load_data(self):
    #     for language in ["en", "zh-cn", "es", "ja"]:
    #         assert dl.load_data(language) is not None

    def test_load_data_invalid_language(self):
        with pytest.raises(ValueError):
            dl.load_data("invalid_language")

    def test_load_data_from_path(self):
        path = "tests/mock_data.csv"
        assert dl.load_data(language="en", data_files=path) is not None

    def test_load_data_from_path_invalid_language(self):
        path = "tests/mock_data.csv"
        with pytest.raises(ValueError):
            dl.load_data(language="invalid_language", data_files=path)

import src.data.data_processing as dp

class TestDataProcessing:
    def test_preprocess_data(self):
        data = dl.load_data(language="en")
        train_dataset, val_dataset, test_dataset = dp.preprocess_data(data, "distilbert-base-uncased-finetuned-sst-2-english", "sentence", "label")
        assert train_dataset is not None
        assert val_dataset is not None
        assert test_dataset is not None

    def test_preprocess_data_with_custom_max_length(self):
        data = dl.load_data(language="en")
        train_dataset, val_dataset, test_dataset = dp.preprocess_data(data, "distilbert-base-uncased-finetuned-sst-2-english", "sentence", "label", max_length=256)
        assert train_dataset is not None
        assert val_dataset is not None
        assert test_dataset is not None

    def test_preprocess_data_invalid_model(self):
        data = dl.load_data(language="en")
        with pytest.raises(Exception):
            dp.preprocess_data(data, "invalid-model-name", "sentence", "label")

    def test_preprocess_data_invalid_columns(self):
        data = dl.load_data(language="en")
        with pytest.raises(Exception):
            dp.preprocess_data(data, "distilbert-base-uncased-finetuned-sst-2-english", "invalid_column", "label")

    def test_preprocess_data_tokenization(self):
        data = dl.load_data(language="en")
        train_dataset, val_dataset, test_dataset = dp.preprocess_data(
            data, 
            "distilbert-base-uncased-finetuned-sst-2-english", 
            "sentence", 
            "label"
        )
        
        # Check that tokenization added expected columns
        expected_columns = ['input_ids', 'attention_mask', 'labels']
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for col in expected_columns:
                assert col in dataset.column_names  # type: ignore