import pytest
import sys
import csv
from pathlib import Path

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

import src.data.data_loader as dl
from src.utils.language_processing import identify_language

class TestDataLoader:
    """Tests for the DataLoader functions."""

    def test_identify_language(self, texts): 
        for text in texts:  
            assert identify_language(text) in ["en", "zh-cn", "es", "ja"]
    
    # WORKS: commmented for time improvement
    def test_load_data(self):
        for language in ["en", "zh-cn", "es", "ja"]:
            assert dl.load_data(language) is not None

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
