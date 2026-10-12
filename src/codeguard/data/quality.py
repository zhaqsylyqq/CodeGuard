"""Data-quality helpers: Python 2 detection and whitespace-insensitive hashing."""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
import warnings

# Syntax that is valid Python 2 but invalid (or a removed builtin) in Python 3.
# These are heuristics on raw text: a marker inside a string or comment is a
# false positive. Noisy candidates (backticks, ``<>``, octal ``0755``) are left
# out on purpose because they are common in docstrings, HTML and dates.
PY2_MARKERS: dict[str, re.Pattern[str]] = {
    "print_statement": re.compile(r"^\s*print\s+[^\s(=,.)\[]", re.MULTILINE),
    "exec_statement": re.compile(r"^\s*exec\s+[\"'\w]", re.MULTILINE),
    "long_literal": re.compile(r"\b(?:0[xX][0-9a-fA-F]+|\d+)[lL]\b"),
    "except_comma": re.compile(r"\bexcept\s+[\w.]+\s*,\s*\w+\s*:"),
    "raise_comma": re.compile(r"\braise\s+[\w.]+\s*,"),
    "ur_string": re.compile(r"(?<![\w'\"])[uU][rR]['\"]"),
    "has_key": re.compile(r"\.has_key\("),
    "xrange": re.compile(r"\bxrange\("),
    "raw_input": re.compile(r"\braw_input\("),
    "unicode_builtin": re.compile(r"(?<![\w.])unicode\("),
    "basestring": re.compile(r"\bbasestring\b"),
    "dict_iter_methods": re.compile(r"\.(?:iteritems|iterkeys|itervalues)\("),
}


def python2_markers(*snippets: str) -> list[str]:
    """Names of the Python 2 markers found in any of the given snippets."""
    text = "\n".join(snippets)
    return [name for name, pattern in PY2_MARKERS.items() if pattern.search(text)]


def parses_as_python3(code: str) -> bool:
    """Whether the dedented snippet is valid Python 3 on its own.

    Only a rough signal: diff snippets are fragments (an ``else:`` without its
    ``if``, a half-open call), so many valid Python 3 snippets fail too.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # e.g. "\d" escapes in old strings
        try:
            ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            return False
    return True


def normalize_code(code: str) -> str:
    """Collapse all whitespace runs so formatting-only differences compare equal."""
    return " ".join(code.split())


def code_hash(*parts: str) -> str:
    """Stable 16-hex-char hash of whitespace-normalised code (one or more parts)."""
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(normalize_code(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
