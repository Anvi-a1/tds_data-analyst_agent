from typing import Dict, Any, List, Union


def _compare_structures(actual: Any, expected: Any, path: str) -> List[str]:
    errors = []
    if type(actual) != type(expected):
        errors.append(
            f"Type mismatch at '{path}': expected {type(expected).__name__}, got {type(actual).__name__}."
        )
        return errors

    if isinstance(expected, dict):
        for key in expected:
            if key not in actual:
                errors.append(f"Missing key at '{path}': '{key}'")
            else:
                errors.extend(
                    _compare_structures(actual[key], expected[key], f"{path}.{key}")
                )

    elif isinstance(expected, list):
        if not actual:
            return errors
        if not expected:
            return errors
        errors.extend(_compare_structures(actual[0], expected[0], f"{path}[0]"))

    return errors


def validate_response_format(
    actual_response: Dict[str, Any], expected_format: Dict[str, Any]
) -> Dict[str, Any]:
    validation_errors = _compare_structures(actual_response, expected_format, "root")

    if not validation_errors:
        return {"success": True, "message": "Response format is valid."}
    else:
        return {
            "success": False,
            "error": "Response validation failed. Please correct the output format.",
            "details": validation_errors,
            "expected_format": expected_format,
            "actual_response": actual_response,
        }
