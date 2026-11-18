"""Route from the answer text and requested unit, without reference scores."""

import json
import math
import re


RTOL = 0.01
SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
CLAIM = re.compile(rf"(?:=|≈|:|is|equals)({NUMBER})$", re.I)


def normalize(text):
    text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:qquad|quad|cdot|[,!;:])|[~·]", " ", text)
    text = re.sub(r"\\[()\[\]]|[$`]", "", text)
    text = text.replace("−", "-").replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^" + m.group().translate(SUPERSCRIPT), text)
    text = re.sub(r"\s+", "", text)
    text = text.replace(r"\approx", "≈").replace(r"\times", "×")
    text = re.sub(r"\^\{([+-]?\d+)\}", r"^\1", text)
    return re.sub(r"[×*]10\^([+-]?\d+)", r"e\1", text)


def reasoning_claim(text, unit):
    """Find the last explicit numeric conclusion before the answer object."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            text = text[:match.start()]
            break
    suffix = normalize(unit)
    if not suffix:
        return None, None
    for line in reversed(text.splitlines()):
        compact = normalize(line).rstrip(".,;")
        if not compact.endswith(suffix):
            continue
        claim = CLAIM.search(compact[:-len(suffix)])
        if claim:
            value = float(claim.group(1))
            if math.isfinite(value):
                return value, line.strip()
    return None, None


def route(text, unit, parse_answer):
    answer, error = parse_answer(text)
    result = {"trigger": None, "final_value": None, "claim_value": None, "claim_line": None}
    if error:
        return {**result, "trigger": "unparseable"}
    result["final_value"] = answer["answer"]
    if answer["unit"].strip() != unit:
        return {**result, "trigger": "unit_mismatch"}
    value, line = reasoning_claim(text, unit)
    result.update(claim_value=value, claim_line=line)
    if value is not None and abs(value - answer["answer"]) > RTOL * max(abs(value), abs(answer["answer"])):
        result["trigger"] = "reasoning_final_mismatch"
    return result


def adopt(review, unit, parse_answer):
    answer, error = parse_answer(review)
    return not error and answer["unit"].strip() == unit
