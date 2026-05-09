import re
from collections import Counter
from typing import Dict, Iterable, List

from src.reference_answer import clean_extracted_answer
from src.utils import normalize_text


MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
MONTH_VALUES = {
    month.lower(): index
    for index, month in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
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
ORDINAL_WORD_VALUES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
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
ORDINAL_WORD_PATTERN = "|".join(
    sorted(ORDINAL_WORD_VALUES, key=len, reverse=True)
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
    r"\b\d+(?:st|nd|rd|th)\b",
    rf"\b\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?(?:\s*(?:{QUANTITY_UNIT_PATTERN}))?\b",
    rf"\b\d+(?:\.\d+)?\s*(?:{QUANTITY_UNIT_PATTERN})\b",
    r"\b\d+(?:\.\d+)?\b",
]
NUMBER_WORD_WITH_UNIT_PATTERN = (
    rf"\b(?:{NUMBER_WORD_PATTERN})(?:[-\s](?:and\s+)?(?:{NUMBER_WORD_PATTERN})){{0,5}}"
    rf"(?:\s+(?:{QUANTITY_UNIT_PATTERN}))?\b"
)
ORDINAL_NUMBER_WORD_PATTERN = (
    rf"\b(?:(?:{NUMBER_WORD_PATTERN})"
    rf"(?:[-\s](?:and\s+)?(?:{NUMBER_WORD_PATTERN}))*[-\s])?"
    rf"(?:{ORDINAL_WORD_PATTERN})\b"
)
ENTITY_TOKEN_PATTERN = (
    r"(?:[A-Z][A-Za-z.'-]*|[A-Z]{2,}(?:'[A-Z]+)?|[A-Z][A-Za-z]*\d+|[A-Z]\d+)"
)
ENTITY_CONNECTOR_PATTERN = r"(?:of|the|and|&|de|del|la|le|van|von)"
ENTITY_PHRASE_PATTERN = (
    rf"{ENTITY_TOKEN_PATTERN}"
    rf"(?:\s+(?:(?:{ENTITY_CONNECTOR_PATTERN})\s+)?{ENTITY_TOKEN_PATTERN})*"
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
    "However",
    "I",
    "In",
    "It",
    "No",
    "On",
    "Or",
    "She",
    "Sound",
    "That",
    "The",
    "They",
    "This",
    "To",
    "With",
    "You",
    "Yes",
}
ENTITY_DISALLOWED_TOKENS = {"However", "You"}
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
    for match in re.finditer(ORDINAL_NUMBER_WORD_PATTERN, text, flags=re.I):
        if inside_any_range(match.start(), match.end(), date_ranges):
            continue
        units.append(match.group(0))
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


def collapse_trailing_duplicate_token(text: str) -> str:
    tokens = text.split()
    if len(tokens) >= 3 and tokens[-1].casefold() == tokens[-2].casefold():
        return " ".join(tokens[:-1])
    return text


def contains_disallowed_entity_token(unit: str) -> bool:
    tokens = re.findall(r"\b[A-Z][A-Za-z.'-]*\b|[A-Z]{2,}\b", unit)
    return any(token in ENTITY_DISALLOWED_TOKENS for token in tokens)


def extract_entity_like_spans(text: str) -> List[str]:
    """
    Extract lightweight entity-like spans without external NER dependencies.
    """
    text = str(text or "")
    units = extract_parenthetical_aliases(text)
    units.extend(regex_units([ENTITY_PHRASE_PATTERN], text))
    cleaned = []
    for unit in units:
        unit = collapse_trailing_duplicate_token(clean_unit(unit))
        if not unit or unit in ENTITY_STOPWORDS:
            continue
        if contains_disallowed_entity_token(unit):
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
        elif token in ORDINAL_WORD_VALUES:
            current += ORDINAL_WORD_VALUES[token]
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


def month_number(text: str) -> int | None:
    match = re.search(rf"\b({MONTH_PATTERN})\b", text, flags=re.I)
    if not match:
        return None
    return MONTH_VALUES[match.group(1).lower()]


def expand_short_year(year: str, base_year: str) -> str:
    if len(year) == 2 and len(base_year) == 4:
        return f"{base_year[:2]}{year}"
    return year


def date_info(unit: str) -> Dict[str, object]:
    text = normalize_text(unit)
    text = re.sub(r"\bbce\b", "bc", text)
    text = re.sub(r"\bce\b", "ad", text)
    era_match = re.search(r"\b(?:bc|ad)\b", text)
    era = era_match.group(0) if era_match else ""
    month = month_number(text)

    range_match = re.search(r"\b(\d{3,4})\s*(?:-|–|to)\s*(\d{2,4})\b", text)
    if range_match:
        start_year = range_match.group(1)
        end_year = expand_short_year(range_match.group(2), start_year)
        return {
            "canonical": f"{start_year}|{end_year}{f':{era}' if era else ''}",
            "years": {start_year, end_year},
            "range": (int(start_year), int(end_year)),
            "month": None,
            "day": None,
            "era": era,
        }

    if month is not None:
        numbers = re.findall(r"\d{1,4}", text)
        year = next((number for number in reversed(numbers) if len(number) >= 3), "")
        day = next((int(number) for number in numbers if len(number) <= 2), None)
        if year and day:
            canonical = f"{year}-{month:02d}-{day:02d}"
        elif year:
            canonical = f"{year}-{month:02d}"
        else:
            canonical = f"{month:02d}"
        if era and year:
            canonical = f"{canonical}:{era}"
        return {
            "canonical": canonical,
            "years": {year} if year else set(),
            "range": None,
            "month": month,
            "day": day,
            "era": era,
        }

    years = re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2}|\d{1,4}(?=\s*(?:bc|ad)\b))\b", text)
    canonical_years = [f"{year}:{era}" if era else year for year in years]
    return {
        "canonical": "|".join(canonical_years) if canonical_years else text,
        "years": set(years),
        "range": None,
        "month": None,
        "day": None,
        "era": era,
    }


def canonical_date(unit: str) -> str:
    return str(date_info(unit)["canonical"])


def acronym(text: str) -> str:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", text)
    letters = [token[0] for token in tokens if token.lower() not in {"of", "the", "and"}]
    return "".join(letters).lower()


def normalize_entity_text(entity: str) -> str:
    cleaned = collapse_trailing_duplicate_token(clean_unit(entity))
    normalized = normalize_text(re.sub(r"[^A-Za-z0-9\s]", " ", cleaned))
    return re.sub(r"\s+", " ", normalized).strip()


def strip_leading_entity_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text).strip()


def entity_aliases(entity: str) -> set[str]:
    cleaned = collapse_trailing_duplicate_token(clean_unit(entity))
    normalized = normalize_entity_text(cleaned)
    aliases = {normalized} if normalized else set()
    without_article = strip_leading_entity_article(normalized)
    if without_article:
        aliases.add(without_article)
    compact = normalized.replace(" ", "")
    if compact:
        aliases.add(compact)
    articleless_compact = without_article.replace(" ", "")
    if articleless_compact:
        aliases.add(articleless_compact)
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
    if sets_have_overlap(reference_aliases, prediction_aliases):
        return True

    for reference_entity in reference_entities:
        for prediction_entity in prediction_entities:
            if entity_phrase_contains_alias(reference_entity, prediction_entity):
                return True
    return False


def entity_phrase_contains_alias(left: str, right: str) -> bool:
    left_normalized = strip_leading_entity_article(normalize_entity_text(left))
    right_normalized = strip_leading_entity_article(normalize_entity_text(right))
    if not left_normalized or not right_normalized:
        return False

    shorter, longer = sorted(
        [left_normalized, right_normalized],
        key=lambda value: len(value.split()),
    )
    if len(shorter.split()) < 2:
        return False
    return f" {shorter} " in f" {longer} "


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


def flag_date_match_conflict(
    reference_values: List[str],
    prediction_values: List[str],
) -> tuple[int, int, int]:
    if not reference_values or not prediction_values:
        return 0, 0, 0

    relation_matrix = [
        [
            date_relation(reference_value, prediction_value)
            for prediction_value in prediction_values
        ]
        for reference_value in reference_values
    ]
    relations = [relation for row in relation_matrix for relation in row]
    compatible_relations = {"exact", "partial"}
    match = int(any(relation in compatible_relations for relation in relations))
    partial = int(any(relation == "partial" for relation in relations))

    reference_covered = [
        any(relation in compatible_relations for relation in row)
        for row in relation_matrix
    ]
    prediction_covered = [
        any(
            relation_matrix[reference_index][prediction_index] in compatible_relations
            for reference_index in range(len(reference_values))
        )
        for prediction_index in range(len(prediction_values))
    ]
    conflict = int(not all(reference_covered) or not all(prediction_covered))
    return match, conflict, partial


def date_relation(left: str, right: str) -> str:
    left_info = date_info(left)
    right_info = date_info(right)
    if left_info["canonical"] == right_info["canonical"]:
        return "exact"
    if left_info["era"] != right_info["era"]:
        return "conflict"

    left_range = left_info["range"]
    right_range = right_info["range"]
    if left_range and right_range:
        return "partial" if range_contains(left_range, right_range) else "conflict"
    if left_range or right_range:
        date_range = left_range or right_range
        point_info = right_info if left_range else left_info
        point_years = point_info["years"]
        if date_range and any(
            date_range[0] <= int(year) <= date_range[1] for year in point_years
        ):
            return "partial"
        return "conflict"

    left_years = left_info["years"]
    right_years = right_info["years"]
    if left_years or right_years:
        if not left_years or not right_years or not left_years & right_years:
            return "conflict"
        if left_info["month"] and right_info["month"]:
            if left_info["month"] != right_info["month"]:
                return "conflict"
            if left_info["day"] and right_info["day"]:
                return "conflict" if left_info["day"] != right_info["day"] else "exact"
            return "partial"
        if left_info["month"] or right_info["month"]:
            return "partial"
        if left_years == right_years:
            return "partial"
        if left_years.issubset(right_years) or right_years.issubset(left_years):
            return "partial"
        return "conflict"

    return "conflict"


def range_contains(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return (
        left[0] <= right[0] <= right[1] <= left[1]
        or right[0] <= left[0] <= left[1] <= right[1]
    )


def specificity_flag(
    reference_items: List[str],
    prediction_items: List[str],
    has_conflict: bool,
    list_item_f1: float,
    partial_factual_overlap: bool,
) -> str:
    if has_conflict:
        return "conflict"
    if partial_factual_overlap:
        return "partial_overlap"
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

    date_match, date_conflict, partial_date_overlap = flag_date_match_conflict(
        reference_units["dates"],
        prediction_units["dates"],
    )

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
    partial_factual_overlap = bool(partial_date_overlap)
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
        "partial_factual_overlap": int(partial_factual_overlap),
        "specificity_flag": specificity_flag(
            reference_items=reference_items,
            prediction_items=prediction_items,
            has_conflict=has_conflict,
            list_item_f1=list_item_f1,
            partial_factual_overlap=partial_factual_overlap,
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
        {
            "metric": "partial_factual_overlap_count",
            "group": "all",
            "value": sum(
                int(record.get("partial_factual_overlap", 0)) for record in records
            ),
        },
    ]

    source_counts = Counter(record.get("specificity_flag", "missing") for record in records)
    for flag, count in sorted(source_counts.items()):
        rows.append({"metric": "specificity_flag_count", "group": flag, "value": count})

    return rows
