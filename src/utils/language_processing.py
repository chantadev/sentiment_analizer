from langdetect import detect

def identify_language(prompt: str):
    return detect(prompt)
