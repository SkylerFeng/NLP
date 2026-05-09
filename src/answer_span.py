import re
from collections import Counter
from typing import Dict, List

from src.reference_answer import (
    NUMBER_WORDS_PATTERN,
    clean_entity_answer,
    clean_extracted_answer,
    question_content_tokens,
    question_type_v2,
)
from src.sentence_level_similarity import split_into_sentences


MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
ERA_PATTERN = r"(?:BCE|BC|CE|AD)"
NUMBER_WORDS_EXTENDED_PATTERN = (
    f"{NUMBER_WORDS_PATTERN}|hundred|thousand|million|billion|half|quarter"
)
QUANTITY_UNIT_PATTERN = (
    r"%|percent|percentage|teams?|people|persons?|players?|members?|points?|"
    r"goals?|runs?|seats?|votes?|episodes?|seasons?|cylinders?|years?|"
    r"miles?|kilometers?|kilometres?|km|meters?|metres?|feet|inches|"
    r"pounds?|tons?|dollars?|million|billion|thousand"
)
ENTITY_TOKEN_PATTERN = (
    r"(?:[A-Z][A-Za-z.'-]*|[A-Z]{2,}(?:'[A-Z]+)?|"
    r"[A-Z][A-Za-z]*\d+)"
)
ENTITY_PHRASE_PATTERN = (
    rf"{ENTITY_TOKEN_PATTERN}"
    rf"(?:\s+(?:of|the|and|&|de|del|la|le|van|von|{ENTITY_TOKEN_PATTERN}))*"
)
UNCERTAIN_PATTERN = re.compile(
    r"\b(?:i do not know|i don't know|unknown|not sure|cannot determine|"
    r"can't determine|no specific|not specified|unclear)\b",
    re.I,
)
QUESTION_ENTITY_STOPWORDS = {
    "Answer",
    "The Answer",
    "I",
    "It",
    "He",
    "She",
    "They",
    "This",
    "That",
    "Yes",
    "No",
}


def trim_words(text: str, max_words: int = 24) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:")


def clean_span(text: str) -> str:
    text = clean_extracted_answer(str(text or ""))
    text = re.sub(
        r"^(?:answer|the answer)\s*(?:is|:)\s+",
        "",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def first_sentence(text: str) -> str:
    sentences = split_into_sentences(text)
    if sentences:
        return clean_span(sentences[0])
    return clean_span(text)


def result(span: str, source: str) -> Dict[str, str]:
    return {
        "prediction_answer_span": trim_words(clean_span(span)),
        "prediction_answer_span_source": source,
    }


def first_match(patterns: List[str], text: str, flags: int = 0) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return clean_span(match.group(1) if match.groups() else match.group(0))
    return ""


def extract_date_span(prediction: str) -> str:
    patterns = [
        rf"\b(?:{MONTH_PATTERN})\s+\d{{1,2}},\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
        rf"\b\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
        rf"\b(?:{MONTH_PATTERN})\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
        rf"\b\d{{1,4}}\s*{ERA_PATTERN}\b",
        r"\b\d{3,4}\s*(?:-|–|to)\s*\d{2,4}\b",
        r"\b\d{3,4}\b",
        r"\b(?:today|yesterday|tomorrow|tonight)\b",
        r"\b(?:last|next|this)\s+(?:day|week|month|year|season|spring|summer|fall|autumn|winter)\b",
        r"\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b",
    ]
    return first_match(patterns, prediction, re.I)


def extract_number_span(prediction: str) -> str:
    patterns = [
        r"\b\d+\s*/\s*\d+\b",
        rf"\b\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?(?:\s+(?:{QUANTITY_UNIT_PATTERN}))?\b",
        rf"\b\d+(?:\.\d+)?\s*(?:{QUANTITY_UNIT_PATTERN})\b",
        rf"\b(?:{NUMBER_WORDS_EXTENDED_PATTERN})(?:[-\s](?:{NUMBER_WORDS_EXTENDED_PATTERN}|{QUANTITY_UNIT_PATTERN})){{0,4}}\b",
        r"\b\d+(?:\.\d+)?\b",
    ]
    return first_match(patterns, prediction, re.I)


def entity_candidates(text: str, question: str) -> List[str]:
    q_tokens = question_content_tokens(question)
    candidates = []
    for match in re.finditer(ENTITY_PHRASE_PATTERN, text):
        candidate = clean_entity_answer(match.group(0))
        if not candidate or candidate in QUESTION_ENTITY_STOPWORDS:
            continue
        cand_tokens = question_content_tokens(candidate)
        if cand_tokens and cand_tokens <= q_tokens:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_who_span(question: str, prediction: str) -> tuple[str, str]:
    by_patterns = [
        rf"\b(?:played|voiced|portrayed|performed|written|composed|directed|produced|created)\s+by\s+({ENTITY_PHRASE_PATTERN})\b",
        rf"\bby\s+({ENTITY_PHRASE_PATTERN})\b",
    ]
    answer = first_match(by_patterns, prediction)
    if answer:
        return answer, "who_by_phrase"

    subject_patterns = [
        rf"\b({ENTITY_PHRASE_PATTERN})\s+(?:sang|sings|wrote|writes|played|plays|voiced|voices|portrayed|portrays|directed|created|founded|invented|produced|performed)\b",
    ]
    answer = first_match(subject_patterns, prediction)
    if answer:
        return answer, "who_verb_subject"

    copular_patterns = [
        rf"\b(?:is|was|are|were)\s+({ENTITY_PHRASE_PATTERN})\b",
    ]
    answer = first_match(copular_patterns, prediction)
    if answer:
        return answer, "who_copular_phrase"

    candidates = entity_candidates(prediction, question)
    if candidates:
        return candidates[0], "who_entity_fallback"
    return "", ""


def extract_where_span(question: str, prediction: str) -> tuple[str, str]:
    location_patterns = [
        rf"\b(?:in|at|near|from|inside|within|outside|around)\s+(?:the\s+)?({ENTITY_PHRASE_PATTERN}(?:,\s*{ENTITY_PHRASE_PATTERN})?)\b",
        rf"\bon\s+(?:the\s+)?({ENTITY_PHRASE_PATTERN}(?:\s+River)?)\b",
    ]
    answer = first_match(location_patterns, prediction)
    if answer:
        return answer, "where_prepositional_phrase"

    candidates = entity_candidates(prediction, question)
    if candidates:
        return candidates[0], "where_entity_fallback"
    return "", ""


def extract_yes_no_span(prediction: str) -> str:
    match = re.match(r"\s*(yes|no)\b[:,]?\s*(.*)", prediction, re.I)
    if match:
        polarity = match.group(1).lower()
        support = clean_span(match.group(2))
        if support:
            return f"{polarity}, {trim_words(support, 14)}"
        return polarity

    lowered = prediction.lower()
    if re.search(r"\b(?:not|never|no longer|false)\b", lowered):
        return "no, " + trim_words(first_sentence(prediction), 14)
    if re.search(r"\b(?:true|correct|does|is|are|was|were|has|have)\b", lowered):
        return "yes, " + trim_words(first_sentence(prediction), 14)
    return ""


def extract_list_span(prediction: str) -> str:
    entity_list_patterns = [
        rf"\b({ENTITY_PHRASE_PATTERN}(?:,\s*{ENTITY_PHRASE_PATTERN})+(?:,?\s+(?:and|or)\s+{ENTITY_PHRASE_PATTERN})?)\b",
        rf"\b({ENTITY_PHRASE_PATTERN}\s+(?:and|or)\s+{ENTITY_PHRASE_PATTERN})\b",
    ]
    answer = first_match(entity_list_patterns, prediction)
    if answer:
        return answer

    match = re.search(r"(?:include|includes|are|were|:)\s+(.+)", prediction, re.I)
    if match:
        return trim_words(clean_span(match.group(1)), 18)
    return ""


def extract_comparison_span(prediction: str) -> str:
    patterns = [
        r"\b([^.;]*\b(?:more|less|fewer|greater|larger|smaller|older|younger|higher|lower)\s+than\b[^.;]*)",
        r"\b([^.;]*\bbetween\b[^.;]*\band\b[^.;]*)",
        r"\b([^.;]*\b(?:first|second|last|largest|smallest|oldest|youngest|highest|lowest)\b[^.;]*)",
    ]
    return first_match(patterns, prediction, re.I)


def extract_definition_span(prediction: str) -> str:
    sentence = first_sentence(prediction)
    patterns = [
        r"\b(?:is|are|was|were)\s+(?:defined\s+as\s+)?(.+)",
        r"\b(?:means|refers\s+to)\s+(.+)",
    ]
    answer = first_match(patterns, sentence, re.I)
    if answer:
        return trim_words(answer, 18)
    return ""


def fallback_span(prediction: str) -> Dict[str, str]:
    source = "uncertain_prediction_fallback" if UNCERTAIN_PATTERN.search(prediction) else "prediction_fallback"
    return result(first_sentence(prediction), source)


def extract_prediction_answer_span(question: str, prediction: str) -> Dict[str, str]:
    """
    Extract a concise answer-like span from a model prediction.

    This is a deterministic Unit 2 heuristic. It does not judge correctness;
    it only creates a shorter prediction view for downstream embedding
    comparison against the frozen reference fields.
    """
    prediction = clean_span(prediction)
    if not prediction:
        return result("", "empty_prediction")

    qtype = question_type_v2(question)

    if qtype == "when":
        answer = extract_date_span(prediction)
        if answer:
            return result(answer, "when_date")
    elif qtype == "number":
        answer = extract_number_span(prediction)
        if answer:
            return result(answer, "number_quantity")
    elif qtype == "who":
        answer, source = extract_who_span(question, prediction)
        if answer:
            return result(answer, source)
    elif qtype == "where":
        answer, source = extract_where_span(question, prediction)
        if answer:
            return result(answer, source)
    elif qtype == "yes_no":
        answer = extract_yes_no_span(prediction)
        if answer:
            return result(answer, "yes_no_polarity")
    elif qtype == "list":
        answer = extract_list_span(prediction)
        if answer:
            return result(answer, "list_coordinated_items")
    elif qtype == "comparison":
        answer = extract_comparison_span(prediction)
        if answer:
            return result(answer, "comparison_relation")
    elif qtype == "definition":
        answer = extract_definition_span(prediction)
        if answer:
            return result(answer, "definition_predicate")

    return fallback_span(prediction)


def add_prediction_answer_spans(records: List[Dict]) -> List[Dict]:
    output_records = []
    for record in records:
        new_record = dict(record)
        new_record.update(
            extract_prediction_answer_span(
                question=str(record.get("question", "")),
                prediction=str(record.get("prediction", "")),
            )
        )
        output_records.append(new_record)
    return output_records


def build_prediction_span_report(records: List[Dict]) -> List[Dict[str, object]]:
    if not records or not any("prediction_answer_span" in record for record in records):
        return []

    rows: List[Dict[str, object]] = [
        {"metric": "num_records", "source": "all", "value": len(records)},
        {
            "metric": "empty_prediction_span_count",
            "source": "all",
            "value": sum(
                1 for record in records if not str(record.get("prediction_answer_span", "")).strip()
            ),
        },
        {
            "metric": "fallback_count",
            "source": "all",
            "value": sum(
                1
                for record in records
                if "fallback" in str(record.get("prediction_answer_span_source", ""))
            ),
        },
    ]

    source_counts = Counter(
        record.get("prediction_answer_span_source", "") or "missing"
        for record in records
    )
    for source, count in sorted(source_counts.items()):
        rows.append({"metric": "source_count", "source": source, "value": count})

    return rows
