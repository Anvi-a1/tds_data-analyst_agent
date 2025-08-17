import os
from typing import Optional, List
import base64
import magic
import PyPDF2


def read_files(file_paths: List[str], max_size_mb: float = 10.0) -> List[dict]:
    """
    Read the contents of multiple files, handling different file types.

    Args:
        file_paths: List of paths to the files to read
        max_size_mb: Maximum file size in MB for each file

    Returns:
        A list of dictionaries, where each dict contains file content and metadata
    """
    results = []
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                results.append(
                    {"error": f"File not found: {file_path}", "success": False}
                )
                continue

            if not os.path.isfile(file_path):
                results.append(
                    {"error": f"Path is not a file: {file_path}", "success": False}
                )
                continue

            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024

            if file_size > max_size_bytes:
                results.append(
                    {
                        "error": f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)",
                        "success": False,
                        "file_size": file_size,
                    }
                )
                continue

            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)

            content = None
            encoding = None

            if file_type.startswith("text"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    encoding = "utf-8"
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                    encoding = "latin-1"
            elif file_type.startswith("image"):
                with open(file_path, "rb") as f:
                    content = base64.b64encode(f.read()).decode("utf-8")
            elif file_type == "application/pdf":
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text()
            else:
                with open(file_path, "rb") as f:
                    content = base64.b64encode(f.read()).decode("utf-8")

            results.append(
                {
                    "content": content,
                    "success": True,
                    "file_path": file_path,
                    "file_size": file_size,
                    "file_type": file_type,
                    "encoding": encoding,
                }
            )

        except Exception as e:
            results.append(
                {"error": f"Error reading file {file_path}: {str(e)}", "success": False}
            )

    return results


def read_file(
    file_path: str, encoding: str = "utf-8", max_size_mb: float = 10.0, **kwargs
) -> dict:
    """
    Read the contents of a file safely with size limits.

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
        max_size_mb: Maximum file size in MB to prevent memory issues

    Returns:
        Dict with file contents, metadata, or error information
    """
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}

        if not os.path.isfile(file_path):
            return {"error": f"Path is not a file: {file_path}", "success": False}

        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024

        if file_size > max_size_bytes:
            return {
                "error": f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)",
                "success": False,
                "file_size": file_size,
            }

        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        return {
            "content": content,
            "success": True,
            "file_path": file_path,
            "file_size": file_size,
            "encoding": encoding,
            "line_count": content.count("\n") + 1 if content else 0,
        }

    except UnicodeDecodeError as e:
        return {
            "error": f"Unable to decode file with {encoding} encoding: {str(e)}",
            "success": False,
            "suggestion": "Try with a different encoding like 'latin-1' or 'cp1252'",
        }
    except PermissionError:
        return {
            "error": f"Permission denied reading file: {file_path}",
            "success": False,
        }
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}", "success": False}


def read_file_lines(
    file_path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
    encoding: str = "utf-8",
    **kwargs,
) -> dict:
    """
    Read specific lines from a file.

    Args:
        file_path: Path to the file to read
        start_line: Starting line number (1-based)
        end_line: Ending line number (1-based, inclusive). If None, read to end
        encoding: File encoding

    Returns:
        Dict with selected lines and metadata
    """
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}

        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()

        total_lines = len(lines)

        if start_line < 1:
            start_line = 1
        if start_line > total_lines:
            return {
                "error": f"Start line {start_line} exceeds file length ({total_lines} lines)",
                "success": False,
            }

        if end_line is None:
            end_line = total_lines
        elif end_line > total_lines:
            end_line = total_lines

        selected_lines = lines[start_line - 1 : end_line]
        content = "".join(selected_lines)

        return {
            "content": content,
            "success": True,
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "selected_line_count": len(selected_lines),
        }

    except Exception as e:
        return {"error": f"Error reading file lines: {str(e)}", "success": False}


def list_directory(dir_path: str, show_hidden: bool = False, **kwargs) -> dict:
    """
    List contents of a directory.

    Args:
        dir_path: Path to the directory
        show_hidden: Whether to show hidden files (starting with .)

    Returns:
        Dict with directory contents and metadata
    """
    try:
        if not os.path.exists(dir_path):
            return {"error": f"Directory not found: {dir_path}", "success": False}

        if not os.path.isdir(dir_path):
            return {"error": f"Path is not a directory: {dir_path}", "success": False}

        items = os.listdir(dir_path)

        if not show_hidden:
            items = [item for item in items if not item.startswith(".")]

        files = []
        directories = []

        for item in items:
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                files.append({"name": item, "size": size, "type": "file"})
            elif os.path.isdir(item_path):
                directories.append({"name": item, "type": "directory"})

        return {
            "success": True,
            "directory": dir_path,
            "files": files,
            "directories": directories,
            "total_items": len(files) + len(directories),
        }

    except PermissionError:
        return {
            "error": f"Permission denied accessing directory: {dir_path}",
            "success": False,
        }
    except Exception as e:
        return {"error": f"Error listing directory: {str(e)}", "success": False}
