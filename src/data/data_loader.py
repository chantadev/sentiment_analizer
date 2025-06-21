from datasets import load_dataset
from pathlib import Path

SUPPORTED_LANGUAGES = ["en", "zh-cn", "es", "ja"]

def load_data(language: str, file_type: str = "csv", data_files = None):
    """
    Loads data from the corresponding source for the given language.
    Allows to load data from the Hugging Face dataset hub or from a local path.
    """
    if data_files is None:
        # Pre-defined datasets for each language
        datasets = {
            # TODO: add datasets for each language, right now only english is supported
            "en": "stanfordnlp/sst2",
            "zh-cn": "stanfordnlp/sst2",  
            "es": "stanfordnlp/sst2",
            "ja": "stanfordnlp/sst2"
        }
        
        if language not in datasets:
            raise ValueError(f"Language {language} not supported")
        
        return load_dataset(datasets[language])
    
    else:
        # Check if the language is supported for our models
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language} not supported. Supported languages: {SUPPORTED_LANGUAGES}")
        
        normalized_path = str(Path(data_files).resolve())
        # Check if the file exists
        if not Path(normalized_path).exists():
            raise FileNotFoundError(f"File not found: {normalized_path}")
        
        return load_dataset(file_type, data_files=normalized_path)