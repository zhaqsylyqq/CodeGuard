"""Rebuild the buggy and fixed code snippets from a unified diff.

A unified diff hunk interleaves three kinds of lines:

* ``" "`` context lines, present in both versions,
* ``"-"`` lines, present only in the old (buggy) version,
* ``"+"`` lines, present only in the new (fixed) version.

So ``buggy = context + "-" lines`` and ``fixed = context + "+" lines``, keeping
the original order and indentation, with the one-character marker removed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffPair:
    """Both sides of a diff plus the changed lines on each side."""

    buggy: str
    fixed: str
    removed: tuple[str, ...]  # lines only in the buggy version
    added: tuple[str, ...]  # lines only in the fixed version
    n_hunks: int

    @property
    def is_change(self) -> bool:
        """False when both sides are identical (e.g. a whitespace-only hunk)."""
        return self.buggy != self.fixed


def reconstruct(diff: str) -> DiffPair:
    """Split a unified diff into its buggy (old) and fixed (new) snippets.

    Handles the edge cases seen in mined diffs: ``\\r\\n`` line endings, file
    headers before the first hunk, the ``\\ No newline at end of file`` marker,
    blank context lines whose leading space was stripped, and several hunks
    (joined in order; callers usually filter those out via ``n_hunks``).
    """
    lines = diff.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline of the last line

    buggy: list[str] = []
    fixed: list[str] = []
    removed: list[str] = []
    added: list[str] = []
    n_hunks = 0

    for line in lines:
        if line.startswith("@@"):
            n_hunks += 1
            continue
        if n_hunks == 0:
            continue  # "--- a/file", "+++ b/file" or other preamble
        if line.startswith("\\"):
            continue  # "\ No newline at end of file"
        marker, body = line[:1], line[1:]
        if marker == "-":
            buggy.append(body)
            removed.append(body)
        elif marker == "+":
            fixed.append(body)
            added.append(body)
        else:
            # " " context line, or "" for a blank context line that lost its space
            text = body if marker == " " else line
            buggy.append(text)
            fixed.append(text)

    return DiffPair(
        buggy="\n".join(buggy),
        fixed="\n".join(fixed),
        removed=tuple(removed),
        added=tuple(added),
        n_hunks=n_hunks,
    )
