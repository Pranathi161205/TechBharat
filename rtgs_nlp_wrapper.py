#!/usr/bin/env python3
"""
RTGS-NLP wrapper (with LLM fallback)

Features:
- Loads AVAILABLE_METRICS dynamically from catalog/catalog.db
- Refreshes metrics before each REPL turn
- When parser+rules can't decide an intent, optionally calls an LLM
  (OpenAI if configured) to translate the user's NL query into an RTGS-CLI command.
- Validates LLM output with a strict regex and writes mapping audit records.
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
import sqlite3
import json
from typing import Tuple, Optional, Dict, Any, Set
from datetime import datetime

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

# Catalog DB path (relative to this file)
CATALOG_DB = os.path.join(os.path.dirname(__file__), "catalog", "catalog.db")
MAPPING_AUDIT_LOG = os.path.join(os.path.dirname(__file__), "mapping_audit.log")

# --- Helper: LLM call (optional) ---
def call_llm(prompt: str, timeout: int = 10) -> Optional[str]:
    """Call an LLM to get a response. Optional: uses openai if available and key set."""
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not api_key:
            logging.info("OPENAI_API_KEY not set; skipping LLM step")
            return None
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates user analytics requests into RTGS-CLI commands."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                request_timeout=timeout
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception:
            resp2 = openai.Completion.create(
                model=os.environ.get("OPENAI_MODEL", "text-davinci-003"),
                prompt=prompt,
                max_tokens=200,
                temperature=0.0,
                request_timeout=timeout
            )
            return resp2.choices[0].text.strip()
    except Exception as e:
        logging.info("LLM call skipped or failed: %s", e)
        return None

# Strict validation regex for allowed RTGS-CLI commands
ALLOWED_CMD_RE = re.compile(r"^(?:set_dataset|get_insights|find_anomalies|generate_dashboard|predict|root_cause)(?:\s+.+)?$", re.IGNORECASE)

def validate_cli_command(cmd: str) -> bool:
    if not cmd or not isinstance(cmd, str):
        return False
    cmd = cmd.strip()
    return bool(ALLOWED_CMD_RE.match(cmd))

def audit_mapping(dataset_id: Optional[str], user_query: str, suggested: Optional[str], valid: bool):
    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_id": dataset_id,
        "user_query": user_query,
        "suggested_command": suggested,
        "valid": bool(valid)
    }
    try:
        with open(MAPPING_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("Failed to write mapping audit log")

# --- Load metrics from catalog ---
def load_metrics_from_catalog(db_path: str = CATALOG_DB) -> Set[str]:
    metrics = set()
    if not os.path.exists(db_path):
        return metrics
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT columns_json FROM datasets WHERE columns_json IS NOT NULL")
        for (cols_json,) in cur.fetchall():
            if not cols_json:
                continue
            try:
                cols = json.loads(cols_json)
                if isinstance(cols, list):
                    for c in cols:
                        metrics.add(str(c))
            except Exception:
                for c in str(cols_json).split(','):
                    c = c.strip().strip('[]"')
                    if c:
                        metrics.add(c)
    except Exception as e:
        logging.warning("Failed to read catalog for metrics: %s", e)
    finally:
        if conn:
            conn.close()
    return metrics

# Initially load metrics
_initial_metrics = list(load_metrics_from_catalog())
if _initial_metrics:
    AVAILABLE_METRICS = _initial_metrics
else:
    AVAILABLE_METRICS = [
        'maternal_mortality', 'infant_mortality', 'vaccination_rate',
        'cases', 'deaths', 'malaria', 'covid',
        'visitors_count', 'tourist_inflow', 'hotel_occupancy',
        'avg_temp', 'max_temp', 'min_temp'
    ]

# Normalizers and command generators
METRIC_NORMALIZATION = {}
for m in AVAILABLE_METRICS:
    METRIC_NORMALIZATION[m] = m
    METRIC_NORMALIZATION[m.replace('_', ' ')] = m
    METRIC_NORMALIZATION[m.lower()] = m
    METRIC_NORMALIZATION[m.replace('_', '').lower()] = m
    METRIC_NORMALIZATION[m.replace('_', '-').lower()] = m

def normalize_metric_token(token: str) -> Optional[str]:
    if not token:
        return None
    t = str(token).strip().lower()
    t = re.sub(r'[^a-z0-9\s_\-]', '', t)
    candidates = [t, t.replace('-', ' '), t.replace(' ', '_'), t.replace(' ', ''), t.replace('_', ' ')]
    for c in candidates:
        if c in METRIC_NORMALIZATION:
            return METRIC_NORMALIZATION[c]
    for m in AVAILABLE_METRICS:
        if re.search(r'\b' + re.escape(m) + r'\b', t):
            return m
    for m in AVAILABLE_METRICS:
        if m in t or t in m:
            return m
    return None

def normalize_district_token(token: str) -> str:
    if not token:
        return 'all'
    t = token.strip()
    t = re.sub(r'[^A-Za-z0-9\s_\-]', '', t)
    t = t.replace(' ', '_')
    return t.lower()

# CLI command generators
def cmd_set_dataset(entities: Dict[str, Any]) -> str:
    name = entities.get('dataset') or entities.get('name') or 'health_data'
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name)).lower()
    return f"set_dataset {name}"

def cmd_get_insights(entities: Dict[str, Any]) -> str:
    district = entities.get('district') or 'all'
    metric = entities.get('metric') or entities.get('metrics') or 'all'
    if isinstance(metric, list):
        metric = metric[0] if metric else 'all'
    district = normalize_district_token(district) if isinstance(district, str) else 'all'
    metric = normalize_metric_token(metric) or 'all'
    return f"get_insights {district} {metric}"

def cmd_show_summary(entities: Dict[str, Any]) -> str:
    return "show_summary"

def cmd_run_analysis(entities: Dict[str, Any]) -> str:
    return "run_analysis"

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
        metrics = ','.join(metrics)
    return f"generate_dashboard {metrics}"

def cmd_dashboard_for(entities: Dict[str, Any]) -> str:
    district = entities.get('district') or 'all'
    metrics = entities.get('metrics') or 'all'
    if isinstance(metrics, list):
        metrics = ','.join(metrics)
    return f"dashboard_for {district} {metrics}"

# Fallback rule-based extractor
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

    for metric in AVAILABLE_METRICS:
        variants = {metric, metric.replace('_', ' '), metric.replace('_', ''), metric.replace('_', '-')}
        for v in variants:
            if v in text_low:
                entities['metric'] = metric
                break
        if entities.get('metric'):
            break

    m = re.search(r"metrics?\s+([a-z0-9_,\s\-]+)", text_low)
    if m:
        raw = m.group(1)
        parts = re.split(r"[,\\s]+", raw.strip())
        normalized = []
        for p in parts:
            nm = normalize_metric_token(p)
            if nm:
                normalized.append(nm)
        if normalized:
            entities['metrics'] = normalized
            entities['metric'] = normalized[0]

    if 'kits' in text_low:
        entities['what'] = 'kits'
    if 'high_risk' in text_low or 'high risk' in text_low:
        entities['what'] = 'high_risk'

    return entities

# LLM-based fallback: build prompt and ask LLM for a CLI command
def build_prompt_for_llm(dataset_id: Optional[str], metadata: Dict[str, Any], metrics_list: list, user_query: str) -> str:
    cols = metadata.get('columns', [])
    lines = []
    for c in cols:
        if isinstance(c, dict):
            name = c.get('name') or c.get('original_name') or ''
            dtype = c.get('dtype', 'unknown')
            samples = c.get('sample_values', [])[:5]
        else:
            name = str(c)
            dtype = 'unknown'
            samples = []
        lines.append(f"{name} ({dtype}) - samples: {', '.join(map(str, samples))}")
    columns_block = "\n".join(lines)

    prompt = f"""
You are an assistant that translates plain English analytics questions into RTGS-CLI commands.
RTGS-CLI supports commands like:
- set_dataset <dataset_id>
- get_insights <district|all> <metric|all>
- find_anomalies <metric|all>
- generate_dashboard <metric1,metric2,...>
Return only the CLI command on a single line.

Dataset id: {dataset_id}\n
Columns:\n{columns_block}\n
Available canonical metrics: {', '.join(metrics_list)}\n
User query: {user_query}\n
Instructions: choose the best matching column(s) from the columns list; use canonical metric names when possible. If you cannot map a metric, use 'all' as fallback. Output only the final RTGS-CLI command.
"""
    return prompt

def llm_translate_to_cli(dataset_id: Optional[str], metadata: Dict[str, Any], metrics_list: list, user_query: str) -> Optional[str]:
    prompt = build_prompt_for_llm(dataset_id, metadata, metrics_list, user_query)
    suggested = call_llm(prompt)
    if not suggested:
        return None
    # single-line normalize
    suggested = suggested.strip().split('\n')[0].strip()
    # validate
    valid = validate_cli_command(suggested)
    audit_mapping(dataset_id, user_query, suggested, valid)
    if valid:
        return suggested
    else:
        return None

# Map NL to CLI (with optional LLM fallback)
def map_nl_to_cli(text: str, use_user_parser: bool = True) -> Tuple[Optional[str], str, Dict[str, Any]]:
    text_low = text.strip()
    intent: Optional[str] = None
    entities: Dict[str, Any] = {}

    if use_user_parser and USER_PARSER_AVAILABLE and user_parse_command is not None:
        try:
            parsed = user_parse_command(text_low, AVAILABLE_METRICS) or {}
            intent = parsed.get('command')
            if parsed.get('district'):
                entities['district'] = normalize_district_token(parsed.get('district'))
            if parsed.get('metrics'):
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

    if not intent:
        if entities.get('metrics') or entities.get('district') or entities.get('dataset') or entities.get('metric'):
            intent = 'get_insights'

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

# Robust RTGSClient
class RTGSClient:
    def __init__(self, cli_path: str = 'rtgs-cli'):
        self.proc: Optional[subprocess.Popen] = None
        self.current_dataset: Optional[str] = None
        tried = []

        def try_start(args_list):
            try:
                self.proc = subprocess.Popen(args_list, stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             text=True, bufsize=1)
                return True
            except FileNotFoundError:
                return False
            except Exception as e:
                logging.exception("Failed to start process with %s: %s", args_list, e)
                return False

        if isinstance(cli_path, str) and ' ' in cli_path:
            parts = cli_path.split()
            tried.append(parts)
            if try_start(parts):
                logging.info("Started CLI via: %s", parts)
                return

        if os.path.exists(cli_path):
            tried.append([cli_path])
            if try_start([cli_path]):
                logging.info("Started CLI via path: %s", cli_path)
                return

        if os.name == 'nt':
            for ext in ('.exe', '.bat', '.cmd', '.py'):
                candidate = cli_path + ext
                tried.append([candidate])
                if os.path.exists(candidate):
                    if ext == '.py':
                        if try_start(['python', candidate]):
                            logging.info("Started CLI via python %s", candidate)
                            return
                    else:
                        if try_start([candidate]):
                            logging.info("Started CLI via: %s", candidate)
                            return

        which = shutil.which(cli_path)
        if which:
            tried.append([which])
            if try_start([which]):
                logging.info("Started CLI via PATH: %s", which)
                return

        raise FileNotFoundError(f"Could not start RTGS-CLI. Tried: {tried}")

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                self._outq.put(line)
        except Exception:
            logging.exception('Error reading stdout')

    def run_command(self, cmd: str, timeout: float = 10.0) -> str:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError('RTGS-CLI process has exited or was not started')
        logging.info('> %s', cmd)
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            raise RuntimeError('Failed to write to RTGS-CLI stdin: %s' % e)

        out_lines = []
        start = time.time()
        while True:
            try:
                line = self._outq.get(timeout=0.5)
                out_lines.append(line)
                if 'RTGS-CLI>' in line or line.strip().endswith('>'):
                    break
            except queue.Empty:
                if time.time() - start > timeout:
                    logging.warning("run_command timeout for: %s", cmd)
                    break
        return ''.join(out_lines)

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                try:
                    if self.proc.stdin:
                        self.proc.stdin.write('exit\n')
                        self.proc.stdin.flush()
                except Exception:
                    pass
                time.sleep(0.2)
                try:
                    self.proc.terminate()
                except Exception:
                    pass
        except Exception:
            pass

    def _start_reader(self):
        self._outq: "queue.Queue[str]" = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

# Helper: fetch metadata JSON path for a dataset id or original filename
def lookup_dataset_metadata(dataset_identifier: Optional[str]) -> Optional[Dict[str, Any]]:
    if not dataset_identifier:
        return None
    if not os.path.exists(CATALOG_DB):
        return None
    try:
        conn = sqlite3.connect(CATALOG_DB)
        cur = conn.cursor()
        cur.execute("SELECT metadata_json_path, id FROM datasets WHERE id=? OR original_filename=?", (dataset_identifier, dataset_identifier))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        metadata_path, dsid = row
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.warning("lookup_dataset_metadata failed: %s", e)
    return None

# Interactive REPL
def repl(cli_path: str, use_user_parser: bool = True, set_default_dataset: bool = True):
    print('Starting RTGS-NLP wrapper. Type "quit" or "exit" to stop.')
    client = RTGSClient(cli_path=cli_path)
    client._start_reader()
    try:
        if set_default_dataset and client.current_dataset is None:
            try:
                out = client.run_command('set_dataset health_data')
                client.current_dataset = "health_data"
            except Exception:
                pass
        while True:
            text = input('You: ').strip()
            if not text:
                continue
            if text.lower() in ('quit', 'exit'):
                break

            # refresh available metrics from catalog so new uploads appear immediately
            try:
                new_metrics = list(load_metrics_from_catalog())
                if new_metrics:
                    AVAILABLE_METRICS[:] = new_metrics
                    METRIC_NORMALIZATION.clear()
                    for m in AVAILABLE_METRICS:
                        METRIC_NORMALIZATION[m] = m
                        METRIC_NORMALIZATION[m.replace('_', ' ')] = m
                        METRIC_NORMALIZATION[m.lower()] = m
                        METRIC_NORMALIZATION[m.replace('_', '').lower()] = m
                        METRIC_NORMALIZATION[m.replace('_', '-').lower()] = m
                    logging.info("Refreshed AVAILABLE_METRICS from catalog: %s", AVAILABLE_METRICS)
            except Exception as e:
                logging.warning("Could not refresh metrics from catalog: %s", e)

            # --- parser + rule-based call ---
            intent, cli_cmd, entities = map_nl_to_cli(text, use_user_parser)

            # If we couldn't determine intent from parser/rules, try LLM fallback
            if not intent:
                ds_for_context = entities.get('dataset') if entities.get('dataset') else client.current_dataset
                metadata = lookup_dataset_metadata(ds_for_context)
                metrics_list = list(AVAILABLE_METRICS)
                llm_cmd = None
                try:
                    if metadata:
                        llm_cmd = llm_translate_to_cli(ds_for_context, metadata, metrics_list, text)
                    else:
                        dummy_meta = {"columns": [{"name": c} for c in metrics_list]}
                        llm_cmd = llm_translate_to_cli(ds_for_context, dummy_meta, metrics_list, text)
                except Exception as e:
                    logging.warning("LLM translation failed: %s", e)
                    llm_cmd = None

                if llm_cmd:
                    intent = 'llm_generated'
                    cli_cmd = llm_cmd
                    logging.info("LLM produced command: %s", cli_cmd)

            # --- auto set_dataset if parser found one ---
            if entities.get('dataset'):
                ds = entities.get('dataset')
                if client.current_dataset != ds:
                    try:
                        logging.info('> set_dataset %s', ds)
                        print(client.run_command(f"set_dataset {ds}"))
                        client.current_dataset = ds
                    except Exception as e:
                        print("Warning: failed to set dataset:", e)

                    intent, cli_cmd, entities = map_nl_to_cli(text, use_user_parser)

            logging.info("Sending CLI command after dataset step: %s", cli_cmd)

            # --- run the intended command ---
            if intent and intent != 'set_dataset':
                print(f'[parsed intent: {intent} | entities: {entities}]')
                try:
                    out = client.run_command(cli_cmd)
                    print(out)
                except Exception as e:
                    print("Error running command:", e)
            else:
                print(f'[parsed intent: {intent} | entities: {entities}]')
    except KeyboardInterrupt:
        print('\nInterrupted')
    finally:
        client.close()

# CLI entrypoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli-path', default='./rtgs-cli', help='Path to RTGS-CLI executable')
    parser.add_argument('--no-default-dataset', dest='no_default', action='store_true', help='Do not auto set default dataset on start')
    parser.add_argument('--no-user-parser', dest='no_user_parser', action='store_true', help='Disable using scripts.nlp_parser even if available')
    args = parser.parse_args()
    repl(args.cli_path, use_user_parser=not args.no_user_parser and USER_PARSER_AVAILABLE, set_default_dataset=not args.no_default)

if __name__ == '__main__':
    main()
