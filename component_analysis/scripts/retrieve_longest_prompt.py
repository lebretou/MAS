#!/usr/bin/env python3
"""
Script to find the top k longest string literals in Python files within a folder.
Uses both AST parsing and regex to catch all strings.
"""

import argparse
import ast
import re
from pathlib import Path


class StringFinder(ast.NodeVisitor):
    """AST visitor to find all string constants."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.strings: list[tuple[str, int, str, Path, int]] = []  # (identifier, length, value, file_path, line_no)
    
    def visit_Assign(self, node: ast.Assign):
        """Visit assignment nodes to find string variables."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            string_value = node.value.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.strings.append((
                        target.id,
                        len(string_value),
                        string_value,
                        self.file_path,
                        node.lineno
                    ))
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment nodes (e.g., var: str = 'value')."""
        if node.value and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if isinstance(node.target, ast.Name):
                string_value = node.value.value
                self.strings.append((
                    node.target.id,
                    len(string_value),
                    string_value,
                    self.file_path,
                    node.lineno
                ))
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant):
        """Visit all constant nodes to find standalone strings."""
        if isinstance(node.value, str) and len(node.value) > 0:
            self.strings.append((
                f"<string@L{node.lineno}>",
                len(node.value),
                node.value,
                self.file_path,
                node.lineno
            ))
        self.generic_visit(node)


def find_strings_with_regex(content: str, file_path: Path) -> list[tuple[str, int, str, Path, int]]:
    """
    Find all string literals using regex patterns.
    Catches triple-quoted and regular quoted strings.
    
    Returns:
        List of tuples (identifier, length, value, file_path, line_no)
    """
    strings = []
    
    # Pattern for triple-quoted strings (""" or ''')
    triple_double = r'"""(.*?)"""'
    triple_single = r"'''(.*?)'''"
    
    # Pattern for regular quoted strings (handles escaped quotes)
    double_quoted = r'"((?:[^"\\]|\\.)*)"'
    single_quoted = r"'((?:[^'\\]|\\.)*)'"
    
    patterns = [
        (triple_double, 'triple_double'),
        (triple_single, 'triple_single'),
        (double_quoted, 'double'),
        (single_quoted, 'single'),
    ]
    
    for pattern, pattern_type in patterns:
        flags = re.DOTALL if pattern_type.startswith('triple') else 0
        for match in re.finditer(pattern, content, flags):
            string_value = match.group(1)
            # Calculate line number
            line_no = content[:match.start()].count('\n') + 1
            
            strings.append((
                f"<{pattern_type}@L{line_no}>",
                len(string_value),
                string_value,
                file_path,
                line_no
            ))
    
    return strings


def find_strings_in_file(file_path: Path, use_regex: bool = True) -> list[tuple[str, int, str, Path, int]]:
    """
    Parse a Python file and find all string literals.
    
    Args:
        file_path: Path to the Python file
        use_regex: Whether to also use regex-based search
    
    Returns:
        List of tuples (identifier, string_length, string_value, file_path, line_no)
    """
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    
    all_strings = []
    seen_values = set()  # To deduplicate
    
    # Parse with AST
    tree = ast.parse(content, filename=str(file_path))
    
    finder = StringFinder(file_path)
    finder.visit(tree)
    
    for item in finder.strings:
        value_hash = hash(item[2])  # Hash of string value
        if value_hash not in seen_values:
            seen_values.add(value_hash)
            all_strings.append(item)
    
    # Also use regex if enabled (catches strings AST might miss)
    if use_regex:
        regex_strings = find_strings_with_regex(content, file_path)
        for item in regex_strings:
            value_hash = hash(item[2])
            if value_hash not in seen_values:
                seen_values.add(value_hash)
                all_strings.append(item)
    
    return all_strings


def find_longest_strings(
    folder_path: str, 
    k: int = 10, 
    recursive: bool = False,
    use_regex: bool = True
) -> list[tuple[str, int, str, Path, int]]:
    """
    Find the top k longest string literals in Python files within the folder.
    
    Args:
        folder_path: Path to the folder to search
        k: Number of top results to return
        recursive: Whether to search subdirectories recursively
        use_regex: Whether to use regex-based search in addition to AST
        
    Returns:
        List of tuples (identifier, string_length, string_value, file_path, line_no) sorted by length descending
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")
    
    pattern = "**/*.py" if recursive else "*.py"
    all_strings: list[tuple[str, int, str, Path, int]] = []
    
    for file_path in folder.glob(pattern):
        if file_path.is_file():
            file_strings = find_strings_in_file(file_path, use_regex)
            all_strings.extend(file_strings)
    
    all_strings.sort(key=lambda x: x[1], reverse=True)
    
    return all_strings[:k]


def truncate_string(s: str, max_length: int = 50) -> str:
    """Truncate a string and add ellipsis if too long."""
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Find the top k longest string literals in Python files within a folder."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder to search"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search subdirectories recursively"
    )
    parser.add_argument(
        "-p", "--preview-length",
        type=int,
        default=50,
        help="Length of string preview to show (default: 50)"
    )
    parser.add_argument(
        "--no-regex",
        action="store_true",
        help="Disable regex-based search (use AST only)"
    )
    
    args = parser.parse_args()
    
    results = find_longest_strings(
        args.folder_path, 
        args.top_k, 
        args.recursive,
        use_regex=not args.no_regex
    )
    
    if not results:
        print("No string literals found in Python files in the specified folder.")
        return
    
    print(f"\nTop {len(results)} longest string literals in '{args.folder_path}':\n")
    print(f"{'Rank':<6}{'Identifier':<25}{'Length':<10}{'Line':<8}{'File Path'}")
    print("-" * 110)
    
    for rank, (identifier, length, value, file_path, line_no) in enumerate(results, 1):
        print(f"{rank:<6}{identifier:<25}{length:<10}{line_no:<8}{file_path}")
        preview = truncate_string(value, args.preview_length)
        print(f"       Preview: \"{preview}\"")
        print()


if __name__ == "__main__":
    main()
