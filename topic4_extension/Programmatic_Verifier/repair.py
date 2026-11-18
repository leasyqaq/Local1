"""Restore an explicitly written exponent when a JSON answer retained only the mantissa."""

from decimal import Decimal
import json
import math
import re

from routing import CLAIM, normalize, reasoning_claim


class NumberLexeme(str):
    """Distinguish a JSON numeric literal from a quoted string."""


def answer_objects(text):
    decoder = json.JSONDecoder(parse_int=NumberLexeme, parse_float=NumberLexeme)
    for match in re.finditer(r"\{", text):
        try:
            obj, size = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            yield match.start(), match.start() + size, obj


def repair(text, unit):
    found = list(answer_objects(text))
    if len(found) != 1:
        return text, None
    start, end, obj = found[0]
    raw = obj["answer"]
    if (set(obj) != {"answer", "unit"} or not isinstance(raw, NumberLexeme)
            or "e" in raw.lower() or not isinstance(obj["unit"], str) or obj["unit"].strip() != unit):
        return text, None

    _, line = reasoning_claim(text, unit)
    if line is None:
        return text, None
    suffix = normalize(unit)
    normalized = normalize(line).rstrip(".,;")
    claim = CLAIM.search(normalized[:-len(suffix)])
    if claim is None or "e" not in claim.group(1).lower():
        return text, None
    mantissa_text, exponent_text = claim.group(1).lower().split("e")
    mantissa = Decimal(mantissa_text)
    exponent = int(exponent_text)
    if mantissa != Decimal(raw) or not mantissa or not exponent or abs(exponent) > 308:
        return text, None
    number = f"{mantissa:f}e{exponent}"
    numerical = float(number)
    if not math.isfinite(numerical) or not numerical:
        return text, None

    # Change only the top-level numeric lexeme, preserving whitespace, unit, and draft.
    obj_text = text[start:end]
    field = re.search(r'"answer"\s*:\s*(' + re.escape(raw) + r')(?=\s*[,}])', obj_text)
    if field is None:
        return text, None
    left, right = start + field.start(1), start + field.end(1)
    updated = text[:left] + number + text[right:]
    return updated, {"original_literal": str(raw), "restored_literal": number, "exponent": exponent,
                     "claim_line": line, "unit": unit, "span_start": left, "span_end": right}
