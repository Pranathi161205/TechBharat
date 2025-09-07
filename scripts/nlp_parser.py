
# scripts/nlp_parser.py
import re
import spacy

# load spaCy model (make sure en_core_web_sm is installed)
nlp = spacy.load("en_core_web_sm")

# synonyms map: user phrases -> canonical metric keys or dataset indicators
METRIC_SYNONYMS = {
    # --- existing health mappings ---
    'infant deaths': 'infant_mortality',
    'infant mortality': 'infant_mortality',
    'maternal deaths': 'maternal_mortality',
    'maternal mortality': 'maternal_mortality',
    'vaccination': 'vaccination_rate',
    'vaccination rate': 'vaccination_rate',
    'vaccinations': 'vaccination_rate',
    'cases': 'cases',
    'deaths': 'deaths',
    'malaria': 'malaria',
    'covid': 'covid',

    # --- tourism metrics ---
    'visitors': 'visitors_count',
    'visitor count': 'visitors_count',
    'visitors count': 'visitors_count',
    'tourist inflow': 'tourist_inflow',
    'tourist arrivals': 'tourist_inflow',
    'hotel occupancy': 'hotel_occupancy',

    # --- temperature metrics ---
    'average temperature': 'avg_temp',
    'avg temp': 'avg_temp',
    'maximum temperature': 'max_temp',
    'max temp': 'max_temp',
    'minimum temperature': 'min_temp',
    'min temp': 'min_temp',

    # --- dataset signals ---
    'health data': 'dataset_health',
    'health': 'dataset_health',
    'dataset': 'dataset_marker',
}


# Known datasets (canonical names)
KNOWN_DATASETS = ["health_data", "temperature_data", "tourist_domestic"]

def _match_metric_token(tok_text: str, available_metrics):
    tok = tok_text.lower().strip()
    if not tok:
        return None
    if tok in available_metrics:
        return tok
    if tok in METRIC_SYNONYMS:
        cand = METRIC_SYNONYMS[tok]
        # dataset markers should not be returned as metrics
        if cand.startswith('dataset_') or cand == 'dataset_marker':
            return None
        return cand
    return None

def parse_command(command_text: str, available_metrics):
    """
    Parse a natural-language command.

    Returns:
      {
        'command': <intent string or None>,
        'district': <string or None>,
        'metrics': [<metric1>, <metric2>, ...],
        'dataset': <dataset_name like 'health_data' or None>
      }
    """
    if not command_text:
        return {'command': None, 'district': None, 'metrics': [], 'dataset': None}

    # Normalize: keep raw doc for NER, but use a cleaned lowercase string for heuristics
    doc = nlp(command_text)
    raw_text = doc.text
    text_low = raw_text.lower().replace('_', ' ')  # normalize underscores

    command = None
    district = None
    metrics = []
    dataset = None

    # --- dataset detection first (so words like 'health' won't be captured as district) ---
    for ds in KNOWN_DATASETS:
        ds_phrase = ds.replace('_', ' ')
        if ds_phrase in text_low or ds in text_low:
            dataset = ds
            # remove the phrase from text_low so later regex/NLP doesn't pick it up as district
            text_low = re.sub(re.escape(ds_phrase), ' ', text_low, flags=re.I)
            break

    # also accept generic 'health data' mention mapping to health_data
    if not dataset:
        for phrase in ('health data', 'health', 'dataset'):
            if phrase in text_low:
                dataset = 'health_data'
                text_low = re.sub(re.escape(phrase), ' ', text_low, flags=re.I)
                break

    # --- Intent detection (simple keyword heuristics) ---
    if 'dashboard' in text_low:
        command = 'generate_dashboard'
    elif 'insight' in text_low or 'coverage' in text_low or 'ratio' in text_low:
        command = 'get_insights'
    elif 'summary' in text_low or 'overview' in text_low:
        command = 'show_summary'
    elif 'analysis' in text_low or 'analyze' in text_low:
        command = 'run_analysis'
    elif 'anomaly' in text_low or 'anomalies' in text_low:
        command = 'find_anomalies'
    elif 'predict' in text_low or 'forecast' in text_low or 'estimate' in text_low:
        command = 'predict'
    elif 'root cause' in text_low or 'rootcause' in text_low or text_low.strip().startswith('why'):
        command = 'root_cause'

    # --- District extraction: try prepositions first (using cleaned text_low) ---
    m = re.search(
        r'\b(?:in|for|at|of|within|across)\s+([A-Za-z0-9_\-\s]+?)(?:\b(?:for|in|on|about|with|that|which|across|within)\b|$)',
        text_low
    )
    if m:
        cand = m.group(1).strip()
        # strip obviously non-district words that might remain
        cand = re.sub(r'\b(data|dataset|insights|summary|analysis|health)\b', '', cand, flags=re.I).strip()
        if cand:
            district = ' '.join(part.capitalize() for part in cand.split())

    # fallback to spaCy GPE/LOC/FAC entities if no preposition match
    if not district:
        for ent in doc.ents:
            if ent.label_ in ('GPE', 'LOC', 'FAC'):
                district = ent.text
                break

    # --- Metric extraction: tokens and noun chunks, with synonyms ---
    for token in doc:
        cand = _match_metric_token(token.text, available_metrics)
        if cand and cand not in metrics:
            metrics.append(cand)

    # check noun chunks to capture multi-word metrics ("infant deaths")
    for nc in doc.noun_chunks:
        cand = _match_metric_token(nc.text.lower().strip(), available_metrics)
        if cand and cand not in metrics:
            metrics.append(cand)

    # final normalization: if metrics empty but certain words appear, try direct substring matches
    if not metrics:
        for mkey in available_metrics:
            if mkey.replace('_', ' ') in text_low:
                metrics.append(mkey)
                break

    return {
        'command': command,
        'district': district,
        'metrics': metrics,
        'dataset': dataset
    }

