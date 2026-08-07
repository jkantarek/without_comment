"""
Field Mapper — Transforms arbitrary JSON from different APIs into a normalized
internal event schema.

Uses JSONPath-style expressions to extract fields and applies transforms
to normalize values.

Internal Schema:
  - event_id: unique event identifier (string)
  - title: human-readable title (string)
  - full_url: URL to the event page (string, relative or absolute)
  - start_date_ms: start time in milliseconds since epoch (integer or None)
  - end_date_ms: end time in milliseconds since epoch (integer or None)
  - body_json: raw event JSON (dict, for future reference)
"""

import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Required internal fields for event normalization
REQUIRED_FIELDS = ["event_id", "title", "full_url", "start_date_ms", "end_date_ms"]

# Available transforms
TRANSFORM_MAP = {
    "identity": lambda v: v,
    "str": lambda v: str(v) if v is not None else None,
    "int": lambda v: int(v) if v is not None else None,
    "float": lambda v: float(v) if v is not None else None,
    "trim": lambda v: v.strip() if isinstance(v, str) else v,
    "multiply_1000": lambda v: int(v * 1000) if v is not None else None,
    "iso_to_ms": lambda v: int(datetime.datetime.fromisoformat(str(v)).timestamp() * 1000)
                 if v is not None else None,
    "date_to_ms": lambda v: int(v.timestamp() * 1000) if v is not None else None,
}


class FieldMappingError(Exception):
    """Raised when a field mapping fails to extract a value."""


def resolve_json_path(obj: Any, path: str) -> Any:
    """
    Resolve a JSONPath-style expression against a JSON object.

    Supports:
    - Dot notation: `.id`, `.location.addressTitle`
    - Bracket notation: `["key"]`, `["nested"]["key"]`
    - Array index: `[0]`, `[0].field`
    - Root access: just `id` or `id`

    Returns None if the path doesn't resolve.
    """
    if obj is None:
        return None

    path = path.strip()
    if not path:
        return None

    if path == ".":
        return obj

    # Strip leading dot if present
    if path.startswith("."):
        path = path[1:]

    # Tokenize: split on dots and brackets
    # Handle: a.b["c"].d[0].e
    segments = []

    # Split on dots, but preserve brackets
    # e.g., "a.b[0].c" -> ["a", "b[0]", "c"]
    remaining = path
    while remaining:
        # Find next dot that's not inside brackets
        depth = 0
        i = 0
        while i < len(remaining):
            c = remaining[i]
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
            elif c == '.' and depth == 0:
                break
            i += 1

        if i == len(remaining):
            # No more dots - take the rest
            segment = remaining
            remaining = ""
        else:
            segment = remaining[:i]
            remaining = remaining[i + 1:]

        if segment:
            segments.append(segment)

    # Now parse each segment
    current = obj
    for seg in segments:
        if not seg:
            continue

        # Check for bracket notation
        bracket_match = re.match(r'^\[', seg)
        if bracket_match:
            # Bracket notation: ["key"] or [0]
            bracket_end = seg.find(']')
            if bracket_end == -1:
                return None
            bracket_content = seg[1:bracket_end]
            # Remove quotes if present
            if bracket_content.startswith('"') and bracket_content.endswith('"'):
                bracket_content = bracket_content[1:-1]
            elif bracket_content.isdigit():
                bracket_content = int(bracket_content)

            if isinstance(current, dict):
                current = current.get(bracket_content)
            elif isinstance(current, list):
                current = current[bracket_content] if isinstance(bracket_content, int) and bracket_content < len(current) else None
            else:
                return None

            # Check for remaining part after bracket
            after_bracket = seg[bracket_end + 1:]
            if after_bracket.startswith('.'):
                after_bracket = after_bracket[1:]
                if after_bracket:
                    if isinstance(current, dict):
                        current = current.get(after_bracket)
                    elif isinstance(current, list):
                        idx = int(after_bracket) if after_bracket.isdigit() else None
                        current = current[idx] if idx is not None and idx < len(current) else None
            elif after_bracket:
                # No dot after bracket, treat as key
                if isinstance(current, dict):
                    current = current.get(after_bracket)
        else:
            # Simple key - check for embedded brackets
            key_part = seg.split('[')[0] if '[' in seg else seg
            if isinstance(current, dict):
                current = current.get(key_part)
            elif isinstance(current, list):
                idx = int(key_part) if key_part.isdigit() else None
                current = current[idx] if idx is not None and idx < len(current) else None
            else:
                return None

            # Handle bracket part after the key
            if '[' in seg:
                bracket_part = seg.split('[')[1]
                bracket_end = bracket_part.find(']')
                if bracket_end == -1:
                    return None
                bracket_content = bracket_part[:bracket_end]
                if bracket_content.startswith('"') and bracket_content.endswith('"'):
                    bracket_content = bracket_content[1:-1]
                elif bracket_content.isdigit():
                    bracket_content = int(bracket_content)

                if isinstance(current, dict):
                    current = current.get(bracket_content)
                elif isinstance(current, list):
                    current = current[bracket_content] if isinstance(bracket_content, int) and bracket_content < len(current) else None

                after_bracket = bracket_part[bracket_end + 1:]
                if after_bracket.startswith('.'):
                    after_bracket = after_bracket[1:]
                    if after_bracket:
                        if isinstance(current, dict):
                            current = current.get(after_bracket)
                        elif isinstance(current, list):
                            idx = int(after_bracket) if after_bracket.isdigit() else None
                            current = current[idx] if idx is not None and idx < len(current) else None
                elif after_bracket:
                    if isinstance(current, dict):
                        current = current.get(after_bracket)
        if current is None:
            break

    return current


def apply_transform(value: Any, transform: str) -> Any:
    """Apply a named transform to a value."""
    func = TRANSFORM_MAP.get(transform)
    if not func:
        logger.warning(f"Unknown transform: {transform}")
        return value
    try:
        return func(value)
    except Exception as e:
        logger.warning(f"Transform '{transform}' failed on {value}: {e}")
        return value


class FieldMapper:
    """
    Maps arbitrary JSON event data to our internal schema using configured
    field mappings.
    """

    def __init__(self, field_mappings: List[Dict]):
        """
        Initialize with field mappings.

        Each mapping is a dict with:
        - internal_field: one of REQUIRED_FIELDS
        - json_path: JSONPath expression to extract the value
        - transform: transform to apply (e.g., "identity", "int", "iso_to_ms")
        """
        self.field_mappings = field_mappings
        # Index by internal field
        self._by_field = {}
        for fm in field_mappings:
            field = fm.get("internal_field")
            if field:
                self._by_field[field] = fm

    def map_event(self, raw_event: Dict, source_base_url: str = "") -> Dict:
        """
        Map a raw JSON event to our internal schema.

        Returns a dict with:
        - event_id, title, full_url, start_date_ms, end_date_ms, body_json
        """
        result = {
            "event_id": None,
            "title": None,
            "full_url": None,
            "start_date_ms": None,
            "end_date_ms": None,
            "body_json": json.dumps(raw_event) if raw_event else None,
        }

        for internal_field in REQUIRED_FIELDS:
            fm = self._by_field.get(internal_field)
            if not fm:
                continue

            json_path = fm.get("json_path", "")
            transform = fm.get("transform", "identity")

            try:
                value = resolve_json_path(raw_event, json_path)
                if value is not None:
                    value = apply_transform(value, transform)
                result[internal_field] = value
            except Exception as e:
                logger.warning(
                    f"Failed to extract {internal_field} from event: {e}"
                )

        # Normalize full_url (make absolute if relative)
        full_url = result.get("full_url", "")
        if full_url and not full_url.startswith("http"):
            if source_base_url:
                result["full_url"] = f"{source_base_url}{full_url}"
            else:
                result["full_url"] = full_url

        return result

    def validate(self) -> List[str]:
        """
        Validate the field mappings.
        Returns a list of error messages.
        """
        errors = []
        mapped_fields = set(fm.get("internal_field") for fm in self.field_mappings if fm.get("internal_field"))

        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in mapped_fields:
                errors.append(f"Missing mapping for required field: {field}")

        # Check for duplicates
        field_counts = {}
        for fm in self.field_mappings:
            field = fm.get("internal_field")
            if field:
                field_counts[field] = field_counts.get(field, 0) + 1
            if field_counts.get(field, 0) > 1:
                errors.append(f"Duplicate mapping for field: {field}")

        # Check json paths
        for fm in self.field_mappings:
            if not fm.get("json_path"):
                errors.append(f"Mapping for {fm.get('internal_field')} has empty json_path")

        # Check transforms
        for fm in self.field_mappings:
            transform = fm.get("transform", "identity")
            if transform not in TRANSFORM_MAP:
                errors.append(f"Unknown transform '{transform}' for {fm.get('internal_field')}")

        return errors

    def get_required_fields(self) -> List[str]:
        """Return the list of required internal fields."""
        return REQUIRED_FIELDS

    def get_available_transforms(self) -> Dict[str, str]:
        """Return available transforms with descriptions."""
        return {
            "identity": "No transformation (raw value)",
            "str": "Convert to string",
            "int": "Convert to integer",
            "float": "Convert to float",
            "trim": "Trim whitespace from string",
            "multiply_1000": "Multiply by 1000 (e.g., seconds to milliseconds)",
            "iso_to_ms": "Convert ISO date string to milliseconds",
        }
