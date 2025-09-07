#!/usr/bin/env python3
"""
Debug-version of RTGS-NLP wrapper — heavy diagnostics to capture why the REPL
may exit immediately. Save as rtgs_nlp_wrapper_debug.py and run with -u to
ensure unbuffered output.

This file intentionally prints many debug lines and writes a wrapper_debug.txt
log with the same content.
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

LOGFILE = "wrapper_debug.txt"

def log_and_print(msg: str):
    # print immediately and append to logfile
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# Clear previous logfile
try:
    open(LOGFILE, "w").close()
except Exception:
    pass

log_and_print("=== rtgs_nlp_wrapper_debug START ===")
log_and_print(f"cwd={os.getcwd()}")
log_and_print(f"sys.executable={sys.executable}")
log_and_print(f"python argv: {sys.argv}")

# Basic logging to console (still use our own log_and_print for clarity)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Try to import user's parser
USER_PARSER_AVAILABLE = False
user_parse_command = None
try:
    log_and_print("Attempting to import scripts.nlp_parser...")
    from scripts.nlp_parser import parse_command as user_parse_command  # type: ignore
    USER_PARSER_AVAILABLE = True
    log_and_print("Imported scripts.nlp_parser successfully.")
except Exception as e:
    user_parse_command = None
    USER_PARSER_AVAILABLE = False
    log_and_print(f"Could not import scripts.nlp_parser: {repr(e)}")

# minimal metric set for debug
AVAILABLE_METRICS = [
    'visitors_count', 'tourist_inflow', 'hotel_occupancy'
]

# Simple normalizers (kept minimal for debug)
def normalize_metric_token(token: str) -> Optional[str]:
    if not token:
        return None
    t = token.strip().lower().replace(' ', '_').replace('-', '_')
    for m in AVAILABLE_METRICS:
        if t == m or t.replace('_','') == m.replace('_',''):
            return m
    return None

def normalize_district_token(token: str) -> str:
    if not token:
        return 'all'
    t = token.strip()
    t = re.sub(r'[^A-Za-z0-9\s_\-]', '', t)
    t = t.replace(' ', '_')
    return t.lower()

# Map generators
def cmd_set_dataset(entities: Dict[str, Any]) -> str:
    name = entities.get('dataset') or 'health_data'
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name)).lower()
    return f"set_dataset {name}"

def cmd_get_insights(entities: Dict[str, Any]) -> str:
    district = normalize_district_token(entities.get('district') or 'all')
    metric = entities.get('metric') or entities.get('metrics') or 'all'
    if isinstance(metric, list):
        metric = metric[0] if metric else 'all'
    metric = normalize_metric_token(metric) or 'all'
    return f"get_insights {district} {metric}"

def cmd_run_analysis(entities: Dict[str, Any]) -> str:
    return "run_analysis"

intent_to_generator = {
    'set_dataset': cmd_set_dataset,
    'get_insights': cmd_get_insights,
    'run_analysis': cmd_run_analysis,
}

def extract_entities_rule_based(text: str) -> Dict[str, Any]:
    text_low = text.lower()
    ent = {}
    # dataset
    m = re.search(r"dataset[:\s]+([a-z0-9_\-]+)", text_low)
    if m:
        ent['dataset'] = m.group(1)
    # district via "in <word>" or "for <word>"
    m = re.search(r"\b(in|for)\s+([A-Za-z0-9\-_ ]+)", text_low)
    if m:
        ent['district'] = m.group(2).strip()
    # metric detection
    for mname in AVAILABLE_METRICS:
        if mname.replace('_', ' ') in text_low or mname in text_low:
            ent['metric'] = mname
            break
    return ent

def map_nl_to_cli(text: str, use_user_parser: bool = True) -> Tuple[Optional[str], str, Dict[str, Any]]:
    log_and_print(f"[map_nl_to_cli] input: {text} (use_user_parser={use_user_parser})")
    intent = None
    entities = {}
    if use_user_parser and USER_PARSER_AVAILABLE and user_parse_command is not None:
        try:
            parsed = user_parse_command(text, AVAILABLE_METRICS) or {}
            log_and_print(f"[user parser] returned: {parsed}")
            intent = parsed.get('command')
            if parsed.get('metric'):
                entities['metric'] = normalize_metric_token(parsed.get('metric'))
            if parsed.get('metrics'):
                metrics = parsed.get('metrics')
                if isinstance(metrics, list):
                    entities['metrics'] = [normalize_metric_token(m) for m in metrics]
                    entities['metric'] = entities['metrics'][0]
            if parsed.get('district'):
                entities['district'] = normalize_district_token(parsed.get('district'))
            if parsed.get('dataset'):
                entities['dataset'] = parsed.get('dataset')
        except Exception as e:
            log_and_print(f"[user parser] raised exception: {repr(e)}")
            intent = None
            entities = extract_entities_rule_based(text)
    else:
        entities = extract_entities_rule_based(text)
        log_and_print(f"[rule-based] entities: {entities}")

    if not intent and (entities.get('metric') or entities.get('district') or entities.get('dataset')):
        intent = 'get_insights'

    gen = intent_to_generator.get(intent, cmd_run_analysis)
    cli_cmd = gen(entities)
    log_and_print(f"[map_nl_to_cli] intent={intent}, cli_cmd='{cli_cmd}', entities={entities}")
    return intent, cli_cmd, entities

# Minimal RTGSClient for debug (spawn subprocess and show outputs)
class RTGSClient:
    def __init__(self, cli_path='rtgs-cli'):
        log_and_print(f"[RTGSClient] constructing with cli_path={cli_path!r}")
        self.proc = None
        self.current_dataset = None
        tried = []

        def try_start(args_list):
            try:
                log_and_print(f"[RTGSClient] trying to start: {args_list}")
                self.proc = subprocess.Popen(args_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                log_and_print("[RTGSClient] subprocess started, pid=%s" % (self.proc.pid,))
                return True
            except FileNotFoundError as fe:
                log_and_print(f"[RTGSClient] FileNotFoundError: {fe}")
                return False
            except Exception as e:
                log_and_print(f"[RTGSClient] Exception starting: {repr(e)}")
                return False

        if isinstance(cli_path, str) and ' ' in cli_path:
            parts = cli_path.split()
            tried.append(parts)
            if try_start(parts):
                return

        if os.path.exists(cli_path):
            tried.append([cli_path])
            if try_start([cli_path]):
                return

        which = shutil.which(cli_path)
        if which:
            tried.append([which])
            if try_start([which]):
                return

        raise FileNotFoundError(f"Could not start RTGS-CLI. Tried: {tried}")

    def _reader(self):
        log_and_print("[RTGSClient] reader thread started")
        try:
            for line in self.proc.stdout:
                if line is None:
                    break
                log_and_print("[RTGS-CLI stdout] " + line.rstrip())
        except Exception as e:
            log_and_print("[RTGSClient] reader exception: " + repr(e))

    def _start_reader(self):
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def run_command(self, cmd: str, timeout: float = 5.0) -> str:
        log_and_print(f"[RTGSClient] run_command -> {cmd!r}")
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Process not available")
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            log_and_print("[RTGSClient] write exception: " + repr(e))
            raise
        # collect output for a short time
        out = []
        start = time.time()
        try:
            while time.time() - start < timeout:
                # can't reliably read stdout here (reader thread is logging), so just sleep a bit
                time.sleep(0.1)
        except Exception:
            pass
        return "\n".join(out)

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                log_and_print("[RTGSClient] sending exit to subprocess")
                try:
                    self.proc.stdin.write("exit\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
                time.sleep(0.1)
                try:
                    self.proc.terminate()
                except Exception:
                    pass
        except Exception:
            pass

def repl(cli_path, use_user_parser=True, set_default_dataset=True):
    log_and_print("Entering repl()")
    print("Starting RTGS-NLP wrapper. Type 'quit' or 'exit' to stop.", flush=True)
    try:
        client = RTGSClient(cli_path=cli_path)
    except Exception as e:
        log_and_print("ERROR: failed to start RTGS-CLI subprocess: " + repr(e))
        print("ERROR: failed to start RTGS-CLI subprocess:", e, flush=True)
        return

    client._start_reader()
    print("RTGSClient started; entering interactive loop. You should see a 'You:' prompt next.", flush=True)
    try:
        if set_default_dataset and client.current_dataset is None:
            try:
                log_and_print("[repl] setting default dataset health_data")
                print(client.run_command("set_dataset health_data"), flush=True)
                client.current_dataset = "health_data"
            except Exception as e:
                log_and_print("[repl] failed to set default dataset: " + repr(e))
        while True:
            try:
                # show prompt
                text = input("You: ").strip()
            except EOFError:
                log_and_print("[repl] EOFError on input; exiting")
                break
            except KeyboardInterrupt:
                log_and_print("[repl] KeyboardInterrupt; exiting")
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break
            intent, cli_cmd, entities = map_nl_to_cli(text, use_user_parser)
            # auto set dataset if needed
            if entities.get("dataset"):
                ds = entities.get("dataset")
                if client.current_dataset != ds:
                    try:
                        log_and_print(f"[repl] auto set_dataset {ds}")
                        print(client.run_command(f"set_dataset {ds}"), flush=True)
                        client.current_dataset = ds
                    except Exception as e:
                        log_and_print("[repl] set_dataset failed: " + repr(e))
            # recompute command
            intent, cli_cmd, entities = map_nl_to_cli(text, use_user_parser)
            log_and_print(f"[repl] sending: {cli_cmd}")
            if intent and intent != "set_dataset":
                print(f"[parsed intent: {intent} | entities: {entities}]", flush=True)
                try:
                    out = client.run_command(cli_cmd)
                    print(out, flush=True)
                except Exception as e:
                    print("Error running command:", e, flush=True)
            else:
                print(f"[parsed intent: {intent} | entities: {entities}]", flush=True)
    finally:
        client.close()
        log_and_print("Exiting repl()")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli-path', default='./rtgs-cli', help='Path to RTGS-CLI executable')
    parser.add_argument('--no-default-dataset', dest='no_default', action='store_true', help='Do not auto set default dataset on start')
    parser.add_argument('--no-user-parser', dest='no_user_parser', action='store_true', help='Disable using scripts.nlp_parser even if available')
    args = parser.parse_args()
    log_and_print(f"main() args: {args}")
    try:
        repl(args.cli_path, use_user_parser=not args.no_user_parser and USER_PARSER_AVAILABLE, set_default_dataset=not args.no_default)
    except Exception as e:
        log_and_print("main() caught exception: " + repr(e))
        print("main() exception:", e, flush=True)

if __name__ == "__main__":
    main()
    log_and_print("=== rtgs_nlp_wrapper_debug END ===")
