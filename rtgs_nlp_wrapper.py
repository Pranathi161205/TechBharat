#!/usr/bin/env python3
"""
RTGS-CLI Natural Language Wrapper (with better token normalization)
"""

import argparse
import subprocess
import threading
import queue
import time
import re
import sys
import logging
import os
import shutil
from typing import Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Try to import user's parser
try:
    from scripts.nlp_parser import parse_command as user_parse_command  # type: ignore
    USER_PARSER_AVAILABLE = True
    logging.info("Loaded user's parser from scripts.nlp_parser")
except Exception as e:
    user_parse_command = None
    USER_PARSER_AVAILABLE = False
    logging.info("User's parser not available: %s", e)

# Configure available metrics (update to match your dataset)
AVAILABLE_METRICS = [
    # health
    'maternal_mortality', 'infant_mortality', 'vaccination_rate',
    'cases', 'deaths', 'malaria', 'covid',
    # tourism
    'visitors_count', 'tourist_inflow', 'hotel_occupancy',
    # temperature
    'avg_temp', 'max_temp', 'min_temp'
]

# Build quick lookup maps for normalization
# e.g. "tourist inflow" -> "tourist_inflow"
METRIC_NORMALIZATION = {}
for m in AVAILABLE_METRICS:
    METRIC_NORMALIZATION[m] = m
    METRIC_NORMALIZATION[m.replace('_', ' ')] = m
    METRIC_NORMALIZATION[m.lower()] = m
    METRIC_NORMALIZATION[m.replace('_', '').lower()] = m
    METRIC_NORMALIZATION[m.replace('_', '-').lower()] = m

def normalize_metric_token(token: str) -> Optional[str]:
    """Try to canonicalize a metric token into one of AVAILABLE_METRICS."""
    if not token:
        return None
    t = token.strip().lower()
    t = re.sub(r'[^a-z0-9\s_\-]', '', t)
    candidates = [t, t.replace('-', ' '), t.replace(' ', '_'), t.replace(' ', ''), t.replace('_', ' ')]
    for c in candidates:
        if c in METRIC_NORMALIZATION:
            return METRIC_NORMALIZATION[c]
    # fallback: try simple substring match (first match)
    for m in AVAILABLE_METRICS:
        if re.search(r'\b' + re.escape(m) + r'\b', t):
            return m
    for m in AVAILABLE_METRICS:
        if m in t or t in m:
            return m
    return None

def normalize_district_token(token: str) -> str:
    """Normalize district names for CLI: lowercase, replace spaces with underscores."""
    if not token:
        return 'all'
    t = token.strip()
    t = re.sub(r'[^A-Za-z0-9\s_\-]', '', t)
    t = t.replace(' ', '_')
    return t.lower()

# CLI command generators (now use normalized tokens)
def cmd_set_dataset(entities: Dict[str, Any]) -> str:
    name = entities.get('dataset') or entities.get('name') or 'health_data'
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name)).lower()
    return f"set_dataset {name}"

def cmd_show_summary(entities: Dict[str, Any]) -> str:
    return "show_summary"

def cmd_run_analysis(entities: Dict[str, Any]) -> str:
    return "run_analysis"

def cmd_get_insights(entities: Dict[str, Any]) -> str:
    district = entities.get('district') or 'all'
    metric = entities.get('metric') or entities.get('metrics') or 'all'
    # normalize district
    district = normalize_district_token(district) if isinstance(district, str) else 'all'
    # normalize metric (handle list or single)
    if isinstance(metric, list):
        metric = metric[0] if metric else 'all'
    norm_metric = normalize_metric_token(metric) or 'all'
    return f"get_insights {district} {norm_metric}"

def cmd_find_anomalies(entities: Dict[str, Any]) -> str:
    metric = entities.get('metric') or 'all'
    metric = normalize_metric_token(metric) or 'all'
    return f"find_anomalies {metric}"

def cmd_predict(entities: Dict[str, Any]) -> str:
    what = entities.get('what') or 'kits'
    if what not in ['kits', 'high_risk']:
        if 'kit' in str(what).lower():
            what = 'kits'
        else:
            what = 'high_risk'
    return f"predict {what}"

def cmd_root_cause(entities: Dict[str, Any]) -> str:
    metric = entities.get('metric') or 'all'
    metric = normalize_metric_token(metric) or 'all'
    return f"root_cause {metric}"

def cmd_generate_dashboard(entities: Dict[str, Any]) -> str:
    metrics = entities.get('metrics') or entities.get('metric_list') or 'all'
    if isinstance(metrics, list):
        normalized = []
        for m in metrics:
            nm = normalize_metric_token(m)
            if nm:
                normalized.append(nm)
        metrics = ','.join(normalized) if normalized else 'all'
    else:
        nm = normalize_metric_token(metrics)
        metrics = nm or 'all'
    return f"generate_dashboard {metrics}"

def cmd_dashboard_for(entities: Dict[str, Any]) -> str:
    district = entities.get('district') or 'all'
    district = normalize_district_token(district)
    metrics = entities.get('metrics') or 'all'
    if isinstance(metrics, list):
        normalized = [normalize_metric_token(m) for m in metrics]
        normalized = [m for m in normalized if m]
        metrics = ','.join(normalized) if normalized else 'all'
    else:
        metrics = normalize_metric_token(metrics) or 'all'
    return f"dashboard_for {district} {metrics}"

# Fallback rule-based extractor (improved to capture multi-word metrics)
def extract_entities_rule_based(text: str) -> Dict[str, Any]:
    text_low = text.lower()
    entities: Dict[str, Any] = {}

    m = re.search(r"dataset[:\s]+([a-z0-9_\-]+)", text_low)
    if m:
        entities['dataset'] = m.group(1)

    m = re.search(r"district\s+([a-z0-9_\-]+)", text_low)
    if m:
        entities['district'] = m.group(1)
    else:
        m = re.search(r"\bfor\s+([a-z0-9_\-]+)\b", text_low)
        if m:
            maybe = m.group(1)
            if maybe not in ['insights', 'summary', 'dashboard', 'analysis', 'dataset', 'metrics', 'metric']:
                entities['district'] = maybe

    # try to detect multi-word metric phrases from AVAILABLE_METRICS (space variants)
    for metric in AVAILABLE_METRICS:
        variants = {metric, metric.replace('_', ' '), metric.replace('_', ''), metric.replace('_', '-')}
        for v in variants:
            if v in text_low:
                entities['metric'] = metric
                break
        if entities.get('metric'):
            break

    # explicit "metrics ..." capture
    m = re.search(r"metrics?\s+([a-z0-9_,\s\-]+)", text_low)
    if m:
        raw = m.group(1)
        parts = re.split(r"[,\s]+", raw.strip())
        normalized = []
        for p in parts:
            nm = normalize_metric_token(p)
            if nm:
                normalized.append(nm)
        if normalized:
            entities['metrics'] = normalized
            entities['metric'] = normalized[0]

    # synonyms
    if 'kits' in text_low:
        entities['what'] = 'kits'
    if 'high_risk' in text_low or 'high risk' in text_low:
        entities['what'] = 'high_risk'

    return entities

# Main mapper
def map_nl_to_cli(text: str, use_user_parser: bool = True) -> Tuple[Optional[str], str, Dict[str, Any]]:
    text_low = text.strip()
    intent: Optional[str] = None
    entities: Dict[str, Any] = {}

    # Try user's parser if available
    if use_user_parser and USER_PARSER_AVAILABLE and user_parse_command is not None:
        try:
            parsed = user_parse_command(text_low, AVAILABLE_METRICS) or {}
            intent = parsed.get('command')
            if parsed.get('district'):
                # normalize district immediately
                entities['district'] = normalize_district_token(parsed.get('district'))
            if parsed.get('metrics'):
                # try to canonicalize each metric
                metrics = parsed.get('metrics')
                if isinstance(metrics, list):
                    normalized = []
                    for m in metrics:
                        nm = normalize_metric_token(m)
                        if nm:
                            normalized.append(nm)
                    if normalized:
                        entities['metrics'] = normalized
                        entities['metric'] = normalized[0]
                else:
                    nm = normalize_metric_token(metrics)
                    if nm:
                        entities['metrics'] = [nm]
                        entities['metric'] = nm
            if parsed.get('metric'):
                nm = normalize_metric_token(parsed.get('metric'))
                if nm:
                    entities['metric'] = nm
            if parsed.get('dataset'):
                entities['dataset'] = parsed.get('dataset')
            if parsed.get('what'):
                entities['what'] = parsed.get('what')
        except Exception as e:
            logging.warning("User parser failed (%s). Falling back to rule-based parser.", e)
            intent = None
            entities = extract_entities_rule_based(text_low)
    else:
        entities = extract_entities_rule_based(text_low)

    # If parser found metrics/district/dataset but did not set an intent,
    # assume the user wants 'get_insights'
    if not intent:
        if entities.get('metrics') or entities.get('district') or entities.get('dataset') or entities.get('metric'):
            intent = 'get_insights'

    # Intent -> generator mapping
    intent_to_generator = {
        'set_dataset': cmd_set_dataset,
        'show_summary': cmd_show_summary,
        'run_analysis': cmd_run_analysis,
        'get_insights': cmd_get_insights,
        'find_anomalies': cmd_find_anomalies,
        'predict': cmd_predict,
        'root_cause': cmd_root_cause,
        'generate_dashboard': cmd_generate_dashboard,
        'dashboard_for': cmd_dashboard_for,
    }

    generator = intent_to_generator.get(intent, cmd_run_analysis)
    cli_cmd = generator(entities)
    return intent, cli_cmd, entities

# (Remaining RTGSClient and REPL unchanged from previous corrected version)
# For brevity, re-use the earlier RTGSClient and repl implementations you already have.
# If you want, I can paste the whole file again with those unchanged sections included verbatim.
