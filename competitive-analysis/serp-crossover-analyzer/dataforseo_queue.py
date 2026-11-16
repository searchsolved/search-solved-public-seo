"""
DataForSEO Standard queue for Google organic results: send searches now, collect them later.

Why: since Google started routing result links through google.com/goto (2026), DataForSEO's
Live and High Priority requests return many organic URLs unresolved (as google.com/goto
links with domain "google.com"). The Standard queue resolves them, but a search can take
25 minutes or more to come back. So tools send the searches, save a job file, and collect
the results from that file once DataForSEO has finished.

Master copy: shared/dataforseo_queue.py. Each tool keeps an identical copy next to its own
code so it can be deployed on its own; scripts/check_dataforseo_queue.py checks they match.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import csv
import io
import math
import time
from base64 import b64encode
from datetime import datetime, timezone

import requests

BASE_URL = 'https://api.dataforseo.com/v3/serp/google/organic'

# DataForSEO accepts up to 100 tasks per task_post call
MAX_TASKS_PER_POST = 100

# Standard queue price per 10 results (normal priority), checked against the cost
# DataForSEO reported on 16 Nov 2026
COST_PER_10_RESULTS = 0.0006

# task_get status codes meaning "not finished yet": task handed, task in queue
PENDING_CODES = {40601, 40602}

# Status codes that stop a run: authorisation failure, account verification,
# insufficient balance, suspicious activity, cost limit, access denied,
# IP not whitelisted, insufficient funds
ACCOUNT_ERROR_CODES = {40100, 40104, 40200, 40201, 40203, 40204, 40207, 40210}

JOB_COLUMNS = ['task_id', 'keyword', 'location_code', 'language_code', 'device', 'depth', 'posted_at']


class QueueError(ValueError):
    """An error reported by the DataForSEO API, with its status code when known."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_account_error(self):
        return self.status_code in ACCOUNT_ERROR_CODES


def build_headers(login, password):
    """Basic auth headers for the DataForSEO API."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    return {'Authorization': f'Basic {cred}', 'Content-Type': 'application/json'}


def estimate_cost(num_keywords, depth):
    """Estimated Standard queue cost in USD."""
    return num_keywords * COST_PER_10_RESULTS * math.ceil(depth / 10)


def _response_json(response):
    try:
        return response.json()
    except ValueError:
        status = 40100 if response.status_code == 401 else None
        raise QueueError(f"Could not parse the API response (HTTP {response.status_code})", status)


def send_searches(keywords, login, password, location_code, language_code, device='desktop',
                  depth=10, delay=2.0, timeout=60):
    """Queue one Google organic search per keyword.

    Returns (job_rows, warnings, cost). job_rows is a list of dicts with JOB_COLUMNS, one per
    queued keyword; save it with job_to_csv() and collect later with collect_results().
    Raises QueueError on account errors (bad login, no balance) before anything is queued.
    """
    headers = build_headers(login, password)
    job_rows, warnings, cost = [], [], 0.0
    posted_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    for start in range(0, len(keywords), MAX_TASKS_PER_POST):
        batch = keywords[start:start + MAX_TASKS_PER_POST]
        body = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "device": device.lower(),
            "depth": depth,
            "tag": keyword[:255],
        } for keyword in batch]
        try:
            response = requests.post(f"{BASE_URL}/task_post", headers=headers, json=body, timeout=timeout)
        except requests.exceptions.RequestException as e:
            raise QueueError(f"Request failed: {e}")
        data = _response_json(response)
        if data.get('status_code') != 20000:
            raise QueueError(f"API error: {data.get('status_message', 'Unknown error')}", data.get('status_code'))
        cost += data.get('cost') or 0.0

        tasks = data.get('tasks') or []
        for keyword, task in zip(batch, tasks):
            if task.get('status_code') == 20100:
                job_rows.append({
                    'task_id': task['id'],
                    'keyword': keyword,
                    'location_code': location_code,
                    'language_code': language_code,
                    'device': device.lower(),
                    'depth': depth,
                    'posted_at': posted_at,
                })
            else:
                warnings.append(f"{keyword}: not queued: {task.get('status_message', 'Unknown error')}")
        for keyword in batch[len(tasks):]:
            warnings.append(f"{keyword}: not queued: no task returned")

        if start + MAX_TASKS_PER_POST < len(keywords) and delay > 0:
            time.sleep(delay)

    return job_rows, warnings, cost


def job_to_csv(job_rows):
    """Job file text (CSV) for the queued searches."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=JOB_COLUMNS)
    writer.writeheader()
    writer.writerows(job_rows)
    return buffer.getvalue()


def job_from_csv(text):
    """Read a job file back, one row per keyword. Raises ValueError if it is not a job file."""
    reader = csv.DictReader(io.StringIO(text.lstrip('﻿')))
    missing = [c for c in ('task_id', 'keyword') if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError("This is not a job file: it has no " + " or ".join(missing) + " column. "
                         "Upload the job file the tool gave you when the searches were sent.")
    rows, seen = [], set()
    for row in reader:
        keyword = row.get('keyword') or ''
        if not (row.get('task_id') or '').strip() or keyword in seen:
            continue
        seen.add(keyword)
        rows.append(row)
    if not rows:
        raise ValueError("The job file has no searches in it.")
    return rows


def collect_results(job_rows, login, password, delay=0.5, timeout=60, progress_callback=None):
    """Fetch finished searches for a job.

    Returns a dict:
        ready:   {keyword: items} for finished searches (items = every SERP element, any type)
        pending: [keyword, ...] still being processed by DataForSEO
        errors:  ["keyword: message", ...]
        cost:    cost DataForSEO reported for the collection calls (normally 0)
    Raises QueueError on account errors.
    """
    headers = build_headers(login, password)
    result = {'ready': {}, 'pending': [], 'errors': [], 'cost': 0.0}

    for i, row in enumerate(job_rows):
        keyword, task_id = row['keyword'], row['task_id'].strip()
        if progress_callback:
            progress_callback(i, len(job_rows), keyword)
        try:
            response = requests.get(f"{BASE_URL}/task_get/advanced/{task_id}", headers=headers, timeout=timeout)
            data = _response_json(response)
            if data.get('status_code') != 20000:
                raise QueueError(f"API error: {data.get('status_message', 'Unknown error')}",
                                 data.get('status_code'))
            result['cost'] += data.get('cost') or 0.0
            task = (data.get('tasks') or [{}])[0]
            status = task.get('status_code')
            if status in PENDING_CODES:
                result['pending'].append(keyword)
            elif status == 20000:
                try:
                    items = task['result'][0]['items'] or []
                except (KeyError, TypeError, IndexError):
                    items = []
                result['ready'][keyword] = items
            else:
                raise QueueError(f"Task error: {task.get('status_message', 'Unknown error')}", status)
        except QueueError as e:
            if e.is_account_error:
                raise
            result['errors'].append(f"{keyword}: {e}")
        except requests.exceptions.RequestException as e:
            result['errors'].append(f"{keyword}: Request failed: {e}")

        if i < len(job_rows) - 1 and delay > 0:
            time.sleep(delay)

    if progress_callback:
        progress_callback(len(job_rows), len(job_rows), None)
    return result


def organic_items(items):
    """Organic results only."""
    return [item for item in items if item.get('type') == 'organic']


def is_unresolved(url):
    """True for a Google redirect link that DataForSEO did not resolve."""
    return 'google.com/goto' in (url or '')
