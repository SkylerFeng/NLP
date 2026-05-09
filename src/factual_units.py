import re
from collections import Counter
from typing import Dict, Iterable, List

from src.reference_answer import clean_extracted_answer
from src.utils import normalize_text


MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
ERA_PATTERN = r"(?:BCE|BC|CE|AD)"
NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALE_WORD_VALUES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
NUMBER_WORD_PATTERN = "|".join(
    sorted([*NUMBER_WORD_VALUES, *SCALE_WORD_VALUES], key=len, reverse=True)
)
QUANTITY_UNIT_PATTERN = (
    r"%|percent|percentage|teams?|people|persons?|players?|members?|points?|"
    r"goals?|runs?|seats?|votes?|episodes?|seasons?|cylinders?|years?|"
    r"weeks?|days?|hours?|minutes?|seconds?|miles?|kilometers?|kilometres?|"
    r"km|meters?|metres?|feet|inches|pounds?|tons?|dollars?|acres?|ha|m2"
)
DATE_PATTERNS = [
    rf"\b(?:{MONTH_PATTERN})\s+\d{{1,2}},\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
    rf"\b\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
    rf"\b(?:early|mid|late)\s+(?:{MONTH_PATTERN})\b",
    rf"\b(?:{MONTH_PATTERN})\s+(?:or\s+early\s+)?(?:{MONTH_PATTERN})\b",
    rf"\b(?:{MONTH_PATTERN})\s+\d{{3,4}}(?:s)?(?:\s*{ERA_PATTERN})?\b",
    rf"\b(?:early|mid|late)\s+\d{{3,4}}s\b",
    rf"\b\d{{1,4}}\s*{ERA_PATTERN}\b",
    r"\b\d{3,4}\s*(?:-|–|to)\s*\d{2,4}\b",
    r"\b\d{3,4}s\b",
    r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b",
]
NUMERIC_PATTERNS = [
    r"\b\d+\s*/\s*\d+\b",
    rf"\b\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?(?:\s*(?:{QUANTITY_UNIT_PATTERN}))?\b",
    rf"\b\d+(?:\.\d+)?\s*(?:{QUANTITY_UNIT_PATTERN})\b",
    r"\b\d+(?:\.\d+)?\b",
]
NUMBER_WORD_WITH_UNIT_PATTERN = (
    rf"\b(?:{NUMBER_WORD_PATTERN})(?:[-\s](?:and\s+)?(?:{NUMBER_WORD_PATTERN})){{0,5}}"
    rf"(?:\s+(?:{QUANTITY_UNIT_PATTERN}))?\b"
)
ENTITY_TOKEN_PATTERN = (
    r"(?:[A-Z][A-Za-z.'-]*|[A-Z]{2,}(?:'[A-Z]+)?|[A-Z][A-Za-z]*\d+|[A-Z]\d+)"
)
ENTITY_PHRASE_PATTERN = (
    rf"{ENTITY_TOKEN_PATTERN}"
    rf"(?:\s+(?:of|the|and|&|de|del|la|le|van|von|{ENTITY_TOKEN_PATTERN}))*"
)
ENTITY_STOPWORDS = {
    "A",
    "An",
    "And",
    "Answer",
    "At",
    "By",
    "For",
    "From",
    "He",
    "Her",
    "His",
    "I",
    "In",
    "It",
    "No",
    "On",
    "Or",
    "She",
    "That",
    "The",
    "They",
    "This",
    "To",
    "With",
    "Yes",
}
FACTUAL_PENALTY_WEIGHT = 0.25


def clean_unit(text: str) -> str:
    text = clean_extracted_answer(str(text or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r,.;:")


def dedupe_units(units: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for unit in units:
        cleaned = clean_unit(unit)
        key = normalize_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def regex_units(patterns: Iterable[str], text: str, flags: int = 0) -> List[str]:
    units = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags):
            units.append(match.group(1) if match.groups() else match.group(0))
    return units


def extract_dates(text: str) -> List[str]:
    """
    Extract date-like factual units while preserving date ranges as units.
    """
    text = str(text or "")
    units = []
    occupied_ranges: List[tuple[int, int]] = []
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.I):
            if inside_any_range(match.start(), match.end(), occupied_ranges):
                continue
            units.append(match.group(1) if match.groups() else match.group(0))
            occupied_ranges.append((match.start(), match.end()))
    return dedupe_units(units)


def span_ranges(text: str, spans: Iterable[str]) -> List[tuple[int, int]]:
    ranges = []
    for span in spans:
        if not span:
            continue
        for match in re.finditer(re.escape(span), text, flags=re.I):
            ranges.append((match.start(), match.end()))
    return ranges


def inside_any_range(start: int, end: int, ranges: List[tuple[int, int]]) -> bool:
    return any(start >= range_start and end <= range_end for range_start, range_end in ranges)


def extract_numbers(text: str) -> List[str]:
    """
    Extract number and quantity units, excluding spans already captured as dates.
    """
    text = str(text or "")
    date_ranges = span_ranges(text, extract_dates(text))
    units = []
    for pattern in NUMERIC_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.I):
            if inside_any_range(match.start(), match.end(), date_ranges):
                continue
            units.append(match.group(0))
    for match in re.finditer(NUMBER_WORD_WITH_UNIT_PATTERN, text, flags=re.I):
        if inside_any_range(match.start(), match.end(), date_ranges):
            continue
        units.extend(split_simple_word_range(match.group(0)))
    return dedupe_units(units)


def split_simple_word_range(unit: str) -> List[str]:
    """
    Split simple word-number ranges like "two and six hours".
    """
    text = normalize_text(unit.replace("-", " "))
    tokens = re.findall(r"[a-z]+", text)
    number_tokens = [token for token in tokens if token in NUMBER_WORD_VALUES]
    has_scale = any(token in SCALE_WORD_VALUES for token in tokens)
    if len(number_tokens) == 2 and not has_scale and "and" in tokens:
        unit_match = re.search(rf"\b({QUANTITY_UNIT_PATTERN})\b$", text)
        suffix = f" {unit_match.group(1)}" if unit_match else ""
        return [number_tokens[0], f"{number_tokens[1]}{suffix}"]
    return [unit]


def extract_parenthetical_aliases(text: str) -> List[str]:
    aliases = []
    pattern = rf"\b({ENTITY_PHRASE_PATTERN})\s*\(\s*({ENTITY_TOKEN_PATTERN})\s*\)"
    for match in re.finditer(pattern, text):
        aliases.extend([match.group(1), match.group(2)])
    return aliases


def extract_entity_like_spans(text: str) -> List[str]:
    """
    Extract lightweight entity-like spans without external NER dependencies.
    """
    text = str(text or "")
    units = extract_parenthetical_aliases(text)
    units.extend(regex_units([ENTITY_PHRASE_PATTERN], text))
    cleaned = []
    for unit in units:
        unit = clean_unit(unit)
        if not unit or unit in ENTITY_STOPWORDS:
            continue
        if re.fullmatch(rf"(?:{MONTH_PATTERN})", unit, flags=re.I):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", unit):
            continue
        cleaned.append(unit)
    return dedupe_units(cleaned)


def number_word_value(text: str) -> float | None:
    total = 0
    current = 0
    matched = False
    for token in re.findall(r"[a-z]+", normalize_text(text.replace("-", " "))):
        if token == "and":
            continue
        if token in NUMBER_WORD_VALUES:
            current += NUMBER_WORD_VALUES[token]
            matched = True
        elif token in SCALE_WORD_VALUES:
            scale = SCALE_WORD_VALUES[token]
            current = max(current, 1) * scale
            if scale >= 1_000:
                total += current
                current = 0
            matched = True
    if not matched:
        return None
    return float(total + current)


def numeric_values(text: str) -> List[float]:
    text = normalize_text(text)
    if re.fullmatch(r"\d+\s*/\s*\d+", text):
        numerator, denominator = re.split(r"\s*/\s*", text)
        denominator_value = float(denominator)
        return [float(numerator) / denominator_value] if denominator_value else []

    values = []
    for raw_number in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", text):
        values.append(float(raw_number.replace(",", "")))

    if values:
        return values

    word_value = number_word_value(text)
    return [word_value] if word_value is not None else []


def canonical_number(unit: str) -> str:
    values = numeric_values(unit)
    if not values:
        return normalize_text(unit)
    return "|".join(f"{value:g}" for value in values)


def canonical_date(unit: str) -> str:
    text = normalize_text(unit)
    text = re.sub(r"\bbce\b", "bc", text)
    text = re.sub(r"\bce\b", "ad", text)
    years = re.findall(r"\d{1,4}", text)
    if re.search(r"\b(?:bc|ad)\b", text) and years:
        era = "bc" if "bc" in text else "ad"
        return "|".join(f"{year}:{era}" for year in years)
    if years:
        return "|".join(years)
    return text


def acronym(text: str) -> str:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", text)
    letters = [token[0] for token in tokens if token.lower() not in {"of", "the", "and"}]
    return "".join(letters).lower()


def entity_aliases(entity: str) -> set[str]:
    cleaned = clean_unit(entity)
    normalized = normalize_text(re.sub(r"[^A-Za-z0-9\s]", " ", cleaned))
    aliases = {normalized} if normalized else set()
    compact = normalized.replace(" ", "")
    if compact:
        aliases.add(compact)
    entity_acronym = acronym(cleaned)
    if len(entity_acronym) >= 2:
        aliases.add(entity_acronym)
    return aliases


def sets_have_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    return bool(set(left) & set(right))


def entity_sets_match(reference_entities: List[str], prediction_entities: List[str]) -> bool:
    reference_aliases = set()
    prediction_aliases = set()
    for entity in reference_entities:
        reference_aliases.update(entity_aliases(entity))
    for entity in prediction_entities:
        prediction_aliases.update(entity_aliases(entity))
    return sets_have_overlap(reference_aliases, prediction_aliases)


def f1_from_counters(reference_items: Iterable[str], prediction_items: Iterable[str]) -> float:
    ref_counter = Counter(reference_items)
    pred_counter = Counter(prediction_items)
    if not ref_counter and not pred_counter:
        return 1.0
    if not ref_counter or not pred_counter:
        return 0.0

    overlap = sum((ref_counter & pred_counter).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(ref_counter.values())
    return 2 * precision * recall / (precision + recall)


def canonical_items(units: Dict[str, List[str]]) -> List[str]:
    items = []
    items.extend(f"number:{canonical_number(unit)}" for unit in units["numbers"])
    items.extend(f"date:{canonical_date(unit)}" for unit in units["dates"])
    for entity in units["entities"]:
        aliases = sorted(entity_aliases(entity))
        if aliases:
            items.append(f"entity:{aliases[0]}")
    return items


def extract_factual_units(text: str) -> Dict[str, List[str]]:
    return {
        "numbers": extract_numbers(text),
        "dates": extract_dates(text),
        "entities": extract_entity_like_spans(text),
    }


def flag_match_and_conflict(
    reference_values: List[str],
    prediction_values: List[str],
) -> tuple[int, int]:
    if not reference_values or not prediction_values:
        return 0, 0
    reference_set = set(reference_values)
    prediction_set = set(prediction_values)
    match = int(bool(reference_set & prediction_set))
    conflict = int(reference_set != prediction_set)
    return match, conflict


def specificity_flag(
    reference_items: List[str],
    prediction_items: List[str],
    has_conflict: bool,
    list_item_f1: float,
) -> str:
    if has_conflict:
        return "conflict"
    if reference_items and not prediction_items:
        return "under_specific_prediction"
    if prediction_items and not reference_items:
        return "over_specific_prediction"
    if 0.0 < list_item_f1 < 1.0:
        return "partial_overlap"
    return "matched_or_neutral"


def conflict_penalty(
    number_conflict: int,
    date_conflict: int,
    entity_conflict: int,
    list_item_f1: float,
) -> float:
    penalty = (
        0.45 * number_conflict
        + 0.35 * date_conflict
        + 0.25 * entity_conflict
    )
    if 0.0 < list_item_f1 < 0.5:
        penalty += 0.15 * (1.0 - list_item_f1)
    return min(1.0, penalty)


def compare_factual_units(reference: str, prediction: str) -> Dict[str, object]:
    reference_units = extract_factual_units(reference)
    prediction_units = extract_factual_units(prediction)

    reference_numbers = [canonical_number(unit) for unit in reference_units["numbers"]]
    prediction_numbers = [canonical_number(unit) for unit in prediction_units["numbers"]]
    number_match, number_conflict = flag_match_and_conflict(
        reference_numbers,
        prediction_numbers,
    )

    reference_dates = [canonical_date(unit) for unit in reference_units["dates"]]
    prediction_dates = [canonical_date(unit) for unit in prediction_units["dates"]]
    date_match, date_conflict = flag_match_and_conflict(reference_dates, prediction_dates)

    entity_match = int(
        bool(reference_units["entities"])
        and bool(prediction_units["entities"])
        and entity_sets_match(reference_units["entities"], prediction_units["entities"])
    )
    entity_conflict = int(
        bool(reference_units["entities"])
        and bool(prediction_units["entities"])
        and not entity_match
    )

    reference_items = canonical_items(reference_units)
    prediction_items = canonical_items(prediction_units)
    list_item_f1 = f1_from_counters(reference_items, prediction_items)
    has_conflict = bool(number_conflict or date_conflict or entity_conflict)
    penalty = conflict_penalty(
        number_conflict=number_conflict,
        date_conflict=date_conflict,
        entity_conflict=entity_conflict,
        list_item_f1=list_item_f1,
    )

    return {
        "reference_numbers": reference_units["numbers"],
        "prediction_numbers": prediction_units["numbers"],
        "reference_dates": reference_units["dates"],
        "prediction_dates": prediction_units["dates"],
        "reference_entities": reference_units["entities"],
        "prediction_entities": prediction_units["entities"],
        "number_match": number_match,
        "number_conflict": number_conflict,
        "date_match": date_match,
        "date_conflict": date_conflict,
        "entity_match": entity_match,
        "entity_conflict": entity_conflict,
        "list_item_f1": list_item_f1,
        "specificity_flag": specificity_flag(
            reference_items=reference_items,
            prediction_items=prediction_items,
            has_conflict=has_conflict,
            list_item_f1=list_item_f1,
        ),
        "factual_conflict_penalty": penalty,
        "factual_unit_score": max(0.0, 1.0 - penalty),
    }


def add_factual_unit_features(
    records: List[Dict],
    reference_field: str = "reference_answer_v2",
    prediction_field: str = "prediction_answer_span",
) -> List[Dict]:
    output_records = []
    for record in records:
        new_record = dict(record)
        new_record.update(
            compare_factual_units(
                reference=str(record.get(reference_field, "")),
                prediction=str(record.get(prediction_field, "")),
            )
        )
        output_records.append(new_record)
    return output_records


def factual_conflict_adjusted_score(
    score: float,
    penalty: float,
    penalty_weight: float = FACTUAL_PENALTY_WEIGHT,
) -> float:
    if penalty_weight < 0.0:
        raise ValueError("penalty_weight must be non-negative.")
    return max(0.0, min(1.0, float(score) - penalty_weight * float(penalty)))


def build_factual_unit_report(records: List[Dict]) -> List[Dict[str, object]]:
    if not records or not any("factual_conflict_penalty" in record for record in records):
        return []

    rows: List[Dict[str, object]] = [
        {"metric": "num_records", "group": "all", "value": len(records)},
        {
            "metric": "number_conflict_count",
            "group": "all",
            "value": sum(int(record.get("number_conflict", 0)) for record in records),
        },
        {
            "metric": "date_conflict_count",
            "group": "all",
            "value": sum(int(record.get("date_conflict", 0)) for record in records),
        },
        {
            "metric": "entity_conflict_count",
            "group": "all",
            "value": sum(int(record.get("entity_conflict", 0)) for record in records),
        },
        {
            "metric": "any_conflict_count",
            "group": "all",
            "value": sum(
                int(float(record.get("factual_conflict_penalty", 0.0)) > 0.0)
                for record in records
            ),
        },
    ]

    source_counts = Counter(record.get("specificity_flag", "missing") for record in records)
    for flag, count in sorted(source_counts.items()):
        rows.append({"metric": "specificity_flag_count", "group": flag, "value": count})

    return rows
