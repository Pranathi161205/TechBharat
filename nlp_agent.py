# nlp_agent.py
"""
Minimal NLP agent for RTGS API.
- Uses spaCy (if installed) for tokenization + NER.
- Falls back to regex/keyword rules if spaCy not available.
- Implements parse_nl_query(text, dataset_hint=None) -> dict(intent, dataset, group, metric, n, raw)
"""

import re
from typing import Dict, Optional, Any

try:
    import spacy
    # load small English model (install with: python -m spacy download en_core_web_sm)
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# known datasets -> synonyms mapping
DATASET_SYNONYMS = {
    "health_data": ["health", "mch", "kits", "maternal", "mch kits", "health data"],
    "tourism_domestic": ["tourism", "tourists", "visitors", "tourism domestic", "tourism_domestic"],
    "consumption_details": ["consumption", "consumption details", "consumption_details", "billing", "services"],
    "temperature_data": ["temperature", "temp", "weather", "temperature_data"],
}

# metric synonyms (map common noun phrases to metric names produced by transform)
METRIC_SYNONYMS = {
    "total_services": ["total services", "services total", "tot services", "totservices"],
    "total_billed_services": ["billed services", "billdservices", "billed"],
    "total_visitors": ["total visitors", "visitors", "tourists"],
    "avg_monthly_visitors": ["average monthly visitors", "avg monthly visitors"],
    "kit_coverage_ratio": ["kit coverage", "kit coverage ratio", "kit coverage %", "kits coverage"],
    "high_risk_ratio": ["high risk", "high risk ratio", "high risk pregnancies"],
    "avg_temp": ["average temperature", "avg temp", "avg_temperature"],
    "max_temperature": ["max temp", "maximum temperature", "high temp"],
    # add more as needed...
}

INTENT_KEYWORDS = {
    "run_pipeline": ["run pipeline", "refresh", "re-run", "re run pipeline", "process dataset", "process data", "load data"],
    "list_metrics": ["list metrics", "show metrics", "columns", "what columns", "list columns"],
    "get_insights": ["show", "give", "get", "what is", "how many", "what are", "find", "lookup"],
    "top_n": ["top", "top 5", "top 10", "highest", "largest", "most"],
    "summary": ["summary", "show summary", "executive summary"],
}

# helper utilities
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _match_dataset(text: str, hint: Optional[str] = None) -> Optional[str]:
    if hint:
        hint = hint.lower()
        # if hint is dataset key already
        if hint in DATASET_SYNONYMS:
            return hint
    t = text.lower()
    # check exact mentions
    for ds, syns in DATASET_SYNONYMS.items():
        for syn in syns:
            if syn in t:
                return ds
    return None

def _match_metric(text: str) -> Optional[str]:
    t = text.lower()
    for metric, syns in METRIC_SYNONYMS.items():
        for syn in syns:
            if syn in t:
                return metric
    # fallback: look for words like 'kits' -> kit_coverage_ratio
    if "kit" in t and "coverage" in t:
        return "kit_coverage_ratio"
    if "visitor" in t or "tourist" in t:
        return "total_visitors"
    return None

def _extract_top_n(text: str) -> Optional[int]:
    # capture "top 5", "top 10", "top N"
    m = re.search(r"top\s+(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # capture "top n" where n as word (five, ten) - naive mapping
    word2num = {"five":5,"ten":10,"three":3,"one":1,"two":2}
    m2 = re.search(r"top\s+([a-z]+)", text, re.IGNORECASE)
    if m2 and m2.group(1).lower() in word2num:
        return word2num[m2.group(1).lower()]
    return None

def _extract_group_via_spacy(doc) -> Optional[str]:
    # look for ORG / GPE / LOC named entities or proper nouns
    for ent in doc.ents:
        if ent.label_ in ("GPE","LOC","ORG","PERSON","NORP"):
            return ent.text
    # fallback: look for proper nouns (PROPN) sequences
    tokens = [t.text for t in doc if t.pos_ == "PROPN"]
    if tokens:
        return " ".join(tokens)
    return None

def _extract_group_via_regex(text: str) -> Optional[str]:
    # look for "in <place>", "for <place>", "for <dataset> in <place>"
    m = re.search(r"(?:in|for|at)\s+([A-Za-z0-9\-\s&]+)", text, re.IGNORECASE)
    if m:
        g = m.group(1).strip()
        # cut if next is metric keyword
        g = re.split(r"\b(total|top|by|show|give|list|summary|metrics)\b", g, flags=re.IGNORECASE)[0].strip()
        return g
    # direct small phrase like 'Hyderabad' alone (naive)
    tokens = text.split()
    if len(tokens) <= 3 and tokens[0].istitle():
        return text.strip()
    return None

def parse_nl_query(text: str, dataset_hint: Optional[str] = None) -> Dict[str,Any]:
    """
    Returns a dict:
      {
        intent: one of [run_pipeline,list_metrics,get_insights,top_n,summary],
        dataset: dataset key or None,
        group: group value or None,
        metric: metric key or None,
        n: integer or None,
        raw: original text
      }
    """
    txt = _normalize_text(text)
    result: Dict[str, Any] = {"intent": None, "dataset": None, "group": None, "metric": None, "n": None, "raw": text}

    # 1) dataset detection
    ds = _match_dataset(txt, hint=dataset_hint)
    result["dataset"] = ds

    # 2) metric detection
    metric = _match_metric(txt)
    result["metric"] = metric

    # 3) top N detection
    n = _extract_top_n(txt)
    result["n"] = n

    # 4) intent detection (simple keyword heuristics)
    for intent, keys in INTENT_KEYWORDS.items():
        for k in keys:
            if k in txt:
                # map 'show/list' ambiguous to get_insights unless 'top' or 'summary' present
                if intent == "get_insights":
                    if "top" in txt or "highest" in txt or re.search(r"top\s+\d+", txt):
                        result["intent"] = "top_n"
                    elif "summary" in txt or "executive summary" in txt:
                        result["intent"] = "summary"
                    else:
                        result["intent"] = "get_insights"
                else:
                    result["intent"] = intent
                break
        if result["intent"]:
            break

    # fallback: if text starts with 'how many' or 'what is' choose get_insights
    if not result["intent"]:
        if txt.startswith("how many") or txt.startswith("what is") or txt.startswith("what are"):
            result["intent"] = "get_insights"
        elif txt.startswith("show") or txt.startswith("list") or txt.startswith("give"):
            result["intent"] = "get_insights"

    # 5) group/entity extraction (prefer spaCy)
    group = None
    if nlp is not None:
        try:
            doc = nlp(text)
            group = _extract_group_via_spacy(doc)
        except Exception:
            group = None
    if not group:
        group = _extract_group_via_regex(text)
    result["group"] = group

    # final adjustments: if intent is top_n but metric missing, try to infer from nearest known metric
    if result["intent"] == "top_n" and not result["metric"]:
        for mname, syns in METRIC_SYNONYMS.items():
            for syn in syns:
                if syn in txt:
                    result["metric"] = mname
                    break
            if result["metric"]:
                break

    # if dataset unknown but we have metric hints (e.g., 'kits' -> health)
    if result["dataset"] is None and result["metric"]:
        if result["metric"].startswith("kit") or "anc" in result["metric"]:
            result["dataset"] = "health_data"
        elif "visitor" in result["metric"]:
            result["dataset"] = "tourism_domestic"

    return result
