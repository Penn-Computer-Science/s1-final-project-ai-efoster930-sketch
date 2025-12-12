import json
from typing import Any, Iterable


def get_nested(obj: Any, path: Iterable[str], default=None):
    """Safely traverse nested dicts/lists by keys/indices.

    get_nested(data, ['team','metadata','displayName'])
    """
    cur = obj
    for p in path:
        if cur is None:
            return default
        # handle integer index for lists
        if isinstance(cur, list):
            try:
                idx = int(p)
            except Exception:
                return default
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return default
        elif isinstance(cur, dict):
            cur = cur.get(p, default)
        else:
            return default
    return cur


def safe_str_field(obj: Any, *paths, fallback=None):
    """Try multiple paths (each path is a list/tuple of keys) and return first non-empty string.

    Example:
      safe_str_field(resp_json, ['displayName'], ['display_name'], fallback='UNKNOWN')
    """
    for p in paths:
        val = get_nested(obj, p)
        if val is None:
            continue
        if isinstance(val, str) and val.strip() != "":
            return val
        # numbers -> str
        if isinstance(val, (int, float)):
            return str(val)
    return fallback


# Example helper to parse a typical API response for a team's display name
def parse_team_display_name(resp_json: Any):
    """Return the most likely display name from a response JSON, or None.

    This function checks several common locations and returns the first match.
    """
    # common single-key locations
    candidates = [
        ['displayName'],
        ['display_name'],
        ['team', 'displayName'],
        ['team', 'display_name'],
        ['data', 'team', 'displayName'],
        ['data', 'team', 'display_name'],
        ['result', 'team', 'displayName'],
        ['result', 'team', 'display_name'],
        ['teams', '0', 'displayName'],
        ['teams', '0', 'display_name'],
        ['teams', 0, 'displayName'] if False else ['teams','0','displayName']
    ]

    # try each candidate
    for path in candidates:
        val = get_nested(resp_json, path)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # fallback: if resp_json is a list and first element has name-like keys
    if isinstance(resp_json, list) and resp_json:
        first = resp_json[0]
        for k in ('displayName', 'display_name', 'name'):
            if isinstance(first, dict) and first.get(k):
                return first.get(k)
    return None


if __name__ == '__main__':
    # quick self-test
    samples = [
        {'displayName': 'Arizona Cardinals'},
        {'team': {'display_name': 'AZ Cardinals'}},
        {'data': {'team': {'displayName': 'Cards'}}},
        [{'displayName': 'ListName'}],
        {}
    ]
    for s in samples:
        print('->', s, '=>', parse_team_display_name(s))
