"""
tools.py -- the 3 tools for the Mini Coding Agent

    search_files(keyword)                       -> find which files are related to the bug
    search_in_file(path, keyword)               -> matching lines with line numbers
    replace_line(path, line_number, new_line)   -> replace one line

This module exports three things (newcodingagent.py imports them):
    TOOLS       -> the tool schema given to the LLM (OpenAI/Groq format)
    code_files  -> list of all C/C++ files in the workspace
    make_tools  -> dict of tool functions bound to a workspace folder
"""

from pathlib import Path

CODE_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"}


# ============================================================
# HELPERS
# ============================================================


def code_files(workspace: Path) -> list[Path]:
    """Return all C/C++ source and header files inside the workspace (sorted)."""
    workspace = Path(workspace)
    if not workspace.exists():
        return []
    return sorted(p for p in workspace.rglob("*") if p.is_file() and p.suffix.lower() in CODE_EXTENSIONS)


def _read_lines(path: Path) -> list[str]:
    """Read a file as a list of lines, keeping the line endings (\\n or \\r\\n)."""
    return path.read_bytes().decode("utf-8", errors="replace").splitlines(keepends=True)


def _resolve(workspace: Path, path: str) -> Path:
    """Match the path given by the LLM to a real file inside the workspace."""
    workspace = Path(workspace).resolve()
    path = (path or "").strip().replace("\\", "/")
    if not path:
        raise ValueError("path is empty. Use search_files to find the file path.")

    candidate = (workspace / path).resolve()

    # never touch files outside the workspace
    if workspace not in candidate.parents and candidate != workspace:
        raise ValueError(f"'{path}' is outside the project folder.")

    if candidate.is_file():
        return candidate

    # the LLM sometimes gives only the file name ("main.cpp"), so try matching by name
    name = Path(path).name
    matches = [f for f in code_files(workspace) if f.name == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ", ".join(str(m.relative_to(workspace)) for m in matches)
        raise ValueError(f"'{path}' is ambiguous. Use one of: {options}")

    raise ValueError(f"File '{path}' not found. Use search_files to see available files.")


# ============================================================
# TOOLS  (functions bound to the workspace)
# ============================================================


def make_tools(workspace: Path) -> dict:
    workspace = Path(workspace).resolve()

    def rel(p: Path) -> str:
        return p.relative_to(workspace).as_posix()

    def search_files(keyword: str = "") -> str:
        """Search file names and contents for the keyword. An empty keyword lists all files."""
        keyword = (keyword or "").strip().lower()
        files = code_files(workspace)
        if not files:
            return "No C++ files found in the project."

        found = []
        for f in files:
            lines = _read_lines(f)
            hits = sum(1 for line in lines if keyword in line.lower()) if keyword else 0
            name_match = keyword in rel(f).lower() if keyword else True
            if not keyword:
                found.append(f"{rel(f)}  ({len(lines)} lines)")
            elif name_match or hits:
                found.append(f"{rel(f)}  (name match: {name_match}, {hits} matching lines)")

        if not found:
            return f"No file matches '{keyword}'. Try a different keyword, or an empty keyword to list all files."
        return "\n".join(found)

    def search_in_file(path: str, keyword: str = "") -> str:
        """Return the lines of a file that contain the keyword, with line numbers. An empty keyword returns the whole file."""
        file = _resolve(workspace, path)
        keyword = (keyword or "").strip().lower()
        lines = _read_lines(file)

        out = []
        for number, line in enumerate(lines, start=1):
            if not keyword or keyword in line.lower():
                out.append(f"{number}: {line.rstrip()}")

        if not out:
            return f"No line in {rel(file)} contains '{keyword}'. File has {len(lines)} lines."
        return f"{rel(file)} ({len(lines)} lines)\n" + "\n".join(out)

    def replace_line(path: str, line_number: int, new_line: str) -> str:
        """Replace one whole line with new_line (line_number starts at 1)."""
        file = _resolve(workspace, path)
        lines = _read_lines(file)

        try:
            line_number = int(line_number)
        except (TypeError, ValueError):
            raise ValueError("line_number must be an integer.")

        if not 1 <= line_number <= len(lines):
            raise ValueError(f"line_number {line_number} is out of range. {rel(file)} has {len(lines)} lines.")

        old = lines[line_number - 1]
        # keep the original line ending (\n or \r\n) so Windows files stay intact
        ending = "\r\n" if old.endswith("\r\n") else "\n" if old.endswith("\n") else ""
        new_line = (new_line or "").rstrip("\r\n")

        lines[line_number - 1] = new_line + ending
        file.write_bytes("".join(lines).encode("utf-8"))

        return f"Replaced line {line_number} in {rel(file)}.\nOLD: {old.rstrip()}\nNEW: {new_line}"

    return {
        "search_files": search_files,
        "search_in_file": search_in_file,
        "replace_line": replace_line,
    }


# ============================================================
# TOOL SCHEMA  (what the LLM sees)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Find which C++ files are related to the bug. Matches the keyword in file names and file contents. An empty keyword lists all files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Word to look for, e.g. 'sum' or 'main'."},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Show lines of a file that contain the keyword, with line numbers. An empty keyword shows the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path as shown by search_files."},
                    "keyword": {"type": "string", "description": "Word to look for. Empty string shows the whole file."},
                },
                "required": ["path", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_line",
            "description": "Replace one whole line of a file with a new line. Take the line number from search_in_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path as shown by search_files."},
                    "line_number": {"type": "integer", "description": "1-based line number to replace."},
                    "new_line": {"type": "string", "description": "The complete new line of C++ (include indentation, no trailing newline)."},
                },
                "required": ["path", "line_number", "new_line"],
            },
        },
    },
]