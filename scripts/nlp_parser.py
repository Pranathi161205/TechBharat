# scripts/nlp_parser.py

import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def parse_command(command_text, available_metrics):
    """
    Parses a natural language command and extracts key entities.
    Returns a dictionary with 'command', 'district', and 'metrics'.
    """
    doc = nlp(command_text.lower())
    
    # Initialize variables
    command = None
    district = None
    metrics = []

    # Check for keywords to determine the command
    if 'show' in doc.text or 'display' in doc.text or 'dashboard' in doc.text:
        if 'dashboard' in doc.text:
            command = 'generate_dashboard'
        elif 'coverage' in doc.text or 'ratio' in doc.text:
            command = 'get_insights'
    
    # Extract district names using spaCy's NER (Named Entity Recognition)
    for ent in doc.ents:
        if ent.label_ == 'GPE':  # GPE stands for Geopolitical Entity
            district = ent.text.title()
    
    # Extract metrics by checking against our list of available metrics
    for token in doc:
        if token.text in available_metrics:
            metrics.append(token.text)
    
    return {'command': command, 'district': district, 'metrics': metrics}