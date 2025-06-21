from ast import Tuple
from typing import Optional
from datasets import Dataset, load_dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

SUPPORTED_LANGUAGES = ["en", "zh-cn", "es", "ja"]
MODELS = {
    "distilbert-base-uncased-finetuned-sst-2-english": "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base-uncased-finetuned-sst-2-english": "bert-base-uncased-finetuned-sst-2-english",
    "roberta-base-finetuned-sst-2-english": "roberta-base-finetuned-sst-2-english",
    "albert-base-v2-finetuned-sst-2-english": "albert-base-v2-finetuned-sst-2-english",
    "albert-base-v2-finetuned-sst-2-english": "albert-base-v2-finetuned-sst-2-english",
}


def load_data(language: str, file_type: str = "csv", data_files = None):
    """
    Loads data from the corresponding source for the given language.
    Allows to load data from the Hugging Face dataset hub or from a local path.

    Parameters:
    - language: Language of the data to load
    - file_type: Type of the file to load (optional)
    - data_files: Path to the file to load (optional)

    Returns:
    - dataset: Dataset object
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


def preprocess_data(data , model_name: str, text_column: str, label_column: str, max_length: Optional[int] = None):
    """
    Preprocesses the data for the given model.

    Parameters:
    - data: Dataset object
    - model_name: Name of the model to use for preprocessing
    - text_column: Name of text column 
    - label_column: Name of label column 
    - max_length: Maximum token length (uses env var if None)

    Returns:
    - train_dataset: Dataset object
    - val_dataset: Dataset object  
    - test_dataset: Dataset object
    """
    if text_column is None or label_column is None:
        raise ValueError("Text and label columns must be provided")
    
    if model_name not in MODELS.keys():
        raise ValueError(f"Model {model_name} not supported")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if max_length is None:
        max_length = int(os.getenv("MAX_TOKEN_LENGTH", "512"))
    
    print(f"Using text column: '{text_column}', label column: '{label_column}'")
    print(f"Max token length: {max_length}")
    
    # Create tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,
            max_length=max_length
        )

    tokenized_dataset = data.map(
        tokenize_function,
        batched=True,
    )
    
    # Rename label column to 'labels' if different (required by transformers)
    if label_column != "labels":
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")

    # Split the dataset
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"] 
    test_dataset = tokenized_dataset["test"]

    return train_dataset, val_dataset, test_dataset