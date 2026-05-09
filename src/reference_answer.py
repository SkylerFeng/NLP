import re
from collections import Counter
from typing import Dict, List

from src.sentence_level_similarity import split_into_sentences
from src.utils import normalize_text


QUESTION_STOPWORDS = {
    "a",
    "an",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

NUMBER_WORDS_PATTERN = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    "twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    "nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)

PRONOUN_REFERENCE_TOKENS = {
    "he",
    "she",
    "it",
    "they",
    "this",
    "that",
    "his",
    "her",
    "its",
    "their",
}

MONTH_TOKENS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

GENERIC_TITLE_FRAGMENTS = {
    "album",
    "airport",
    "book",
    "city",
    "club",
    "company",
    "country",
    "county",
    "episode",
    "film",
    "game",
    "league",
    "movie",
    "novel",
    "river",
    "school",
    "season",
    "series",
    "song",
    "state",
    "station",
    "team",
    "university",
    "village",
}

DEFAULT_MAX_EVIDENCE_FALLBACK_TOKENS = 36


def question_content_tokens(question: str) -> set[str]:
    text = re.sub(r"[^\w\s]", " ", normalize_text(question))
    return {
        token
        for token in text.split()
        if token not in QUESTION_STOPWORDS and len(token) > 2
    }


def question_type(question: str) -> str:
    q = normalize_text(question)
    if q.startswith("who"):
        return "who"
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("how many") or q.startswith("how much"):
        return "number"
    if q.startswith("which river"):
        return "where"
    if q.startswith("which country") or " which country " in f" {q} ":
        return "where"
    return "general"


def question_type_v2(question: str) -> str:
    """
    Shared Unit 1 question-type wrapper.

    Baseline extraction keeps using question_type() for reproducibility. V2
    features use this wrapper so later units can expand question types without
    mutating baseline reference_answer behavior.
    """
    q = normalize_text(question)
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
        return "yes_no"
    if q.startswith(("can ", "could ", "will ", "would ", "has ", "have ", "had ")):
        return "yes_no"
    if q.startswith("when") or re.match(r"^(what|which)\s+(year|date|month|day)\b", q):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("who"):
        return "who"
    if q.startswith("how many") or q.startswith("how much"):
        return "number"
    if re.search(r"\b(number|percentage|percent|amount|total)\b", q):
        return "number"
    if q.startswith("which river"):
        return "where"
    if q.startswith("which country") or " which country " in f" {q} ":
        return "where"
    if q.startswith(("list ", "name ")) or re.search(r"\b(names|types|kinds|examples)\b", q):
        return "list"
    if q.startswith("which") and " or " in q:
        return "comparison"
    if q.startswith(("what is", "what are", "define ")):
        return "definition"
    return "general"


def clean_extracted_answer(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.strip(" \t\n\r,.;:-")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_entity_answer(text: str) -> str:
    text = clean_extracted_answer(text)
    text = re.sub(
        r"\b(?:At|In|On|The|A|An|And|Or|By|For|With|From|To|Of)$",
        "",
        text,
    ).strip()
    parts = text.split()
    midpoint = len(parts) // 2
    if parts and len(parts) % 2 == 0 and parts[:midpoint] == parts[midpoint:]:
        text = " ".join(parts[:midpoint])
    return text


def truncate_words(text: str, max_words: int = 36) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def reference_token_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text)))


def normalized_answer_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def one_token_suspicious_reference(answer: str, question: str) -> bool:
    tokens = normalized_answer_tokens(answer)
    if len(tokens) != 1:
        return False

    token = tokens[0]
    qtype = question_type_v2(question)
    return (
        token in PRONOUN_REFERENCE_TOKENS
        or (token in MONTH_TOKENS and qtype != "when")
        or token in GENERIC_TITLE_FRAGMENTS
    )


def malformed_numeric_fragment(answer: str, evidence: str) -> bool:
    answer_digits = re.sub(r"\D", "", str(answer))
    if not answer_digits:
        return False

    if re.fullmatch(r"0{2,}", answer_digits) and re.search(
        rf"\b\d{{1,3}},{re.escape(answer_digits)}\b",
        str(evidence),
    ):
        return True

    return False


def candidate_sentences(passage: str) -> List[str]:
    sentences = split_into_sentences(passage)
    if sentences:
        return sentences
    passage = clean_extracted_answer(passage)
    return [passage] if passage else []


def score_sentence(question_tokens: set[str], sentence: str, qtype: str) -> float:
    sentence_norm = re.sub(r"[^\w\s]", " ", normalize_text(sentence))
    sentence_tokens = set(sentence_norm.split())
    overlap = len(question_tokens & sentence_tokens)
    score = overlap / max(len(question_tokens), 1)

    if qtype == "when" and re.search(r"\b(?:\d{3,4}|January|February|March|April|May|June|July|August|September|October|November|December)\b", sentence):
        score += 0.35
    elif qtype == "number" and re.search(rf"\b(?:\d+(?:\.\d+)?|{NUMBER_WORDS_PATTERN})\b", sentence, re.I):
        score += 0.35
    elif qtype in {"who", "where"} and re.search(r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4}\b", sentence):
        score += 0.25

    sentence_lower = sentence.lower()
    if qtype == "who":
        if {"play", "plays", "played", "voice", "voiced"} & question_tokens:
            if re.search(r"\b(played|plays|playing|voiced|voice|repris\w*|role|portray\w*|star\w*)\b", sentence_lower):
                score += 0.75
        if {"wrote", "write", "sang", "sing", "song", "composed"} & question_tokens:
            if re.search(r"\b(written|wrote|sung|sang|performed|composed|recorded)\b", sentence_lower):
                score += 0.75

    return score


def ranked_evidence_sentences(question: str, passage: str, qtype: str | None = None) -> List[str]:
    sentences = candidate_sentences(passage)
    if not sentences:
        return []

    qtype = qtype or question_type(question)
    q_tokens = question_content_tokens(question)
    return sorted(
        sentences,
        key=lambda sentence: (
            score_sentence(q_tokens, sentence, qtype),
            -abs(len(sentence.split()) - 24),
        ),
        reverse=True,
    )


def best_evidence_sentence(question: str, passage: str) -> str:
    sentences = ranked_evidence_sentences(question, passage)
    if not sentences:
        return ""
    return clean_extracted_answer(sentences[0])


def extract_date_answer(sentence: str) -> str:
    patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{3,4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{3,4}\b",
        r"\b\d{3,4}\s*(?:-|–|to)\s*\d{2,4}\b",
        r"\b\d{3,4}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if match:
            return clean_extracted_answer(match.group(0))
    return ""


def extract_number_answer(sentence: str) -> str:
    pattern = rf"\b(?:\d+(?:\.\d+)?|{NUMBER_WORDS_PATTERN})(?:[-\s][A-Za-z]+){{0,3}}\b"
    match = re.search(pattern, sentence, re.I)
    if match:
        return clean_extracted_answer(match.group(0))
    return ""


def extract_number_answer_v2(sentence: str) -> str:
    patterns = [
        r"\b\d{1,3}(?:,\d{3})+(?:[-\s][A-Za-z]+){0,3}\b",
        rf"\b(?:\d+(?:\.\d+)?|{NUMBER_WORDS_PATTERN})(?:[-\s][A-Za-z]+){{0,3}}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence, re.I)
        if match:
            return clean_extracted_answer(match.group(0))
    return ""


def validate_reference_answer(
    answer: str,
    question: str,
    evidence: str,
    source: str = "",
    max_evidence_fallback_tokens: int = DEFAULT_MAX_EVIDENCE_FALLBACK_TOKENS,
) -> Dict[str, object]:
    cleaned = clean_extracted_answer(str(answer or ""))
    tokens = normalized_answer_tokens(cleaned)

    if not cleaned:
        return {
            "reference_answer_valid": False,
            "reference_validation_reason": "empty",
        }

    if len(tokens) == 1 and tokens[0] in PRONOUN_REFERENCE_TOKENS:
        return {
            "reference_answer_valid": False,
            "reference_validation_reason": "pronoun_or_determiner",
        }

    if len(tokens) == 1 and tokens[0] in MONTH_TOKENS and question_type_v2(question) != "when":
        return {
            "reference_answer_valid": False,
            "reference_validation_reason": "one_token_month_non_date",
        }

    if len(tokens) == 1 and tokens[0] in GENERIC_TITLE_FRAGMENTS:
        return {
            "reference_answer_valid": False,
            "reference_validation_reason": "generic_title_fragment",
        }

    if malformed_numeric_fragment(cleaned, evidence):
        return {
            "reference_answer_valid": False,
            "reference_validation_reason": "malformed_numeric_fragment",
        }

    if (
        source == "nq_evidence_sentence"
        and reference_token_count(evidence) > max_evidence_fallback_tokens
    ):
        return {
            "reference_answer_valid": False,
            "reference_validation_reason": "long_evidence_fallback",
        }

    if reference_token_count(cleaned) > max_evidence_fallback_tokens:
        return {
            "reference_answer_valid": False,
            "reference_validation_reason": "long_reference_span",
        }

    return {
        "reference_answer_valid": True,
        "reference_validation_reason": "valid",
    }


def proper_noun_candidates(sentence: str) -> List[str]:
    candidates = re.findall(
        r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5}\b",
        sentence,
    )
    cleaned = []
    for candidate in candidates:
        candidate = clean_entity_answer(candidate)
        if candidate and candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned


def leading_title_entity(passage: str, question: str) -> str:
    first_chunk = clean_extracted_answer(passage[:160])
    candidates = proper_noun_candidates(first_chunk)
    q_tokens = question_content_tokens(question)
    for candidate in candidates:
        cand_tokens = question_content_tokens(candidate)
        if cand_tokens and not cand_tokens <= q_tokens:
            return candidate
    return ""


def extract_person_answer(sentence: str, question: str) -> str:
    patterns = [
        r"eclipsed\s+by\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5})",
        r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5})\s+is\s+(?:a|the)\s+(?:leading|largest|biggest|top|main|major)\s+(?:producer|holder|source|supplier)",
        r"(?:with|starring)\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5})\s+(?:repris\w*|playing|voicing|starring|as)",
        r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5})\s+(?:repris\w*|playing|voicing|starring|portray\w*)",
        r"(?:written|sung|performed|composed|directed|produced|created|founded|developed|invented|played|voiced)\s+by\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5})",
        r"by\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5})",
        r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5})\s+(?:wrote|sang|performed|composed|directed|produced|created|founded|developed|invented|played|voiced|holds|held|was|is)",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if match:
            return clean_entity_answer(match.group(1))

    q_tokens = question_content_tokens(question)
    for candidate in proper_noun_candidates(sentence):
        cand_tokens = question_content_tokens(candidate)
        if cand_tokens and not cand_tokens <= q_tokens:
            return candidate
    return ""


def extract_location_answer(sentence: str, question: str) -> str:
    patterns = [
        r"(?:located|based|situated)\s+in\s+([A-Z][A-Za-z.'-]+(?:,\s*[A-Z][A-Za-z.'-]+)?(?:\s+[A-Z][A-Za-z.'-]+){0,3})",
        r"(?:on|along)\s+the\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4}(?:\s+River)?)",
        r"in\s+([A-Z][A-Za-z.'-]+(?:,\s*[A-Z][A-Za-z.'-]+)?(?:\s+[A-Z][A-Za-z.'-]+){0,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if match:
            return clean_extracted_answer(match.group(1))

    q_tokens = question_content_tokens(question)
    for candidate in proper_noun_candidates(sentence):
        cand_tokens = question_content_tokens(candidate)
        if cand_tokens and not cand_tokens <= q_tokens:
            return candidate
    return ""


def heuristic_reference_candidate_v2(sentence: str, question: str, qtype: str) -> str:
    if qtype == "who":
        return extract_person_answer(sentence, question)
    if qtype == "when":
        return extract_date_answer(sentence)
    if qtype == "number":
        return extract_number_answer_v2(sentence)
    if qtype == "where":
        return extract_location_answer(sentence, question)
    return ""


def reference_v2_result(
    *,
    answer: str,
    source: str,
    evidence: str,
    question: str,
    max_evidence_fallback_tokens: int,
) -> Dict[str, object]:
    answer = clean_extracted_answer(answer)
    validation = validate_reference_answer(
        answer=answer,
        question=question,
        evidence=evidence,
        source=source,
        max_evidence_fallback_tokens=max_evidence_fallback_tokens,
    )
    return {
        "question_type_v2": question_type_v2(question),
        "reference_answer_v2": answer,
        "reference_answer_valid": validation["reference_answer_valid"],
        "reference_validation_reason": validation["reference_validation_reason"],
        "reference_answer_source_v2": source,
    }


def extract_nq_reference_answer_v2(
    question: str,
    passage: str,
    baseline_result: Dict[str, str] | None = None,
    max_evidence_fallback_tokens: int = DEFAULT_MAX_EVIDENCE_FALLBACK_TOKENS,
) -> Dict[str, object]:
    """
    Extract and validate a v2 NQ reference without mutating baseline fields.

    Invalid heuristic candidates are skipped so a pronoun, month fragment, or
    malformed numeric fragment does not become the final comparison target.
    """
    baseline_result = baseline_result or extract_nq_reference_answer(question, passage)
    baseline_answer = str(baseline_result.get("reference_answer", ""))
    baseline_source = str(baseline_result.get("reference_answer_source", ""))
    baseline_evidence = str(baseline_result.get("reference_evidence", passage))
    baseline_validation = validate_reference_answer(
        answer=baseline_answer,
        question=question,
        evidence=baseline_evidence,
        source=baseline_source,
        max_evidence_fallback_tokens=max_evidence_fallback_tokens,
    )
    if baseline_validation["reference_answer_valid"]:
        return {
            "question_type_v2": question_type_v2(question),
            "reference_answer_v2": baseline_answer,
            "reference_answer_valid": True,
            "reference_validation_reason": "valid",
            "reference_answer_source_v2": baseline_source,
        }

    qtype = question_type(question)
    sentences = ranked_evidence_sentences(question, passage, qtype=qtype)
    if not sentences:
        return reference_v2_result(
            answer="",
            source="empty",
            evidence="",
            question=question,
            max_evidence_fallback_tokens=max_evidence_fallback_tokens,
        )

    if qtype == "who" and {"sang", "sing"} & question_content_tokens(question):
        title_answer = leading_title_entity(passage, question)
        if title_answer:
            result = reference_v2_result(
                answer=title_answer,
                source="nq_who_title_heuristic",
                evidence=sentences[0],
                question=question,
                max_evidence_fallback_tokens=max_evidence_fallback_tokens,
            )
            if result["reference_answer_valid"]:
                return result

    for sentence in sentences:
        candidate = heuristic_reference_candidate_v2(sentence, question, qtype)
        if not candidate:
            continue

        result = reference_v2_result(
            answer=candidate,
            source=f"nq_{qtype}_heuristic",
            evidence=sentence,
            question=question,
            max_evidence_fallback_tokens=max_evidence_fallback_tokens,
        )
        if result["reference_answer_valid"]:
            return result

    evidence = baseline_evidence or clean_extracted_answer(sentences[0])
    fallback_answer = (
        baseline_answer
        if baseline_source == "nq_evidence_sentence"
        else truncate_words(evidence, max_evidence_fallback_tokens)
    )
    return reference_v2_result(
        answer=fallback_answer,
        source="nq_evidence_sentence",
        evidence=evidence,
        question=question,
        max_evidence_fallback_tokens=max_evidence_fallback_tokens,
    )


def extract_nq_reference_answer(question: str, passage: str) -> Dict[str, str]:
    """
    Extract a shorter answer-like reference from an NQ evidence passage.

    The teacher-provided NQ file stores Wikipedia evidence passages in
    correct_answer. Directly comparing a short prediction with the whole passage
    makes cosine similarity measure topic overlap instead of answer equivalence.
    This heuristic keeps the project offline and reproducible while reducing
    that representation mismatch.
    """
    evidence = best_evidence_sentence(question, passage)
    if not evidence:
        return {"reference_answer": "", "reference_answer_source": "empty"}

    qtype = question_type(question)
    extracted = ""

    if qtype == "who":
        q_tokens = question_content_tokens(question)
        if {"sang", "sing"} & q_tokens:
            title_answer = leading_title_entity(passage, question)
            if title_answer:
                return {
                    "reference_answer": title_answer,
                    "reference_answer_source": "nq_who_title_heuristic",
                    "reference_evidence": evidence,
                }
        extracted = extract_person_answer(evidence, question)
    elif qtype == "when":
        extracted = extract_date_answer(evidence)
    elif qtype == "number":
        extracted = extract_number_answer(evidence)
    elif qtype == "where":
        extracted = extract_location_answer(evidence, question)

    if extracted:
        return {
            "reference_answer": extracted,
            "reference_answer_source": f"nq_{qtype}_heuristic",
            "reference_evidence": evidence,
        }

    return {
        "reference_answer": truncate_words(evidence),
        "reference_answer_source": "nq_evidence_sentence",
        "reference_evidence": evidence,
    }


def prepare_reference_answers(records: List[Dict], dataset_name: str) -> List[Dict]:
    """
    Add evaluation reference fields.

    For NQ, reference_answer is an extracted answer/evidence sentence.
    For other datasets, reference_answer mirrors ground_truth so downstream code
    can use one field consistently.
    """
    output_records = []

    for record in records:
        new_record = dict(record)
        if dataset_name == "nq":
            extracted = extract_nq_reference_answer(
                question=str(record.get("question", "")),
                passage=str(record.get("ground_truth", "")),
            )
            new_record.update(extracted)
            new_record.update(
                extract_nq_reference_answer_v2(
                    question=str(record.get("question", "")),
                    passage=str(record.get("ground_truth", "")),
                    baseline_result=extracted,
                )
            )
        else:
            new_record["reference_answer"] = record.get("ground_truth", "")
            new_record["reference_answer_source"] = "ground_truth"
            new_record["reference_evidence"] = record.get("ground_truth", "")
            new_record["question_type_v2"] = question_type_v2(str(record.get("question", "")))
            new_record["reference_answer_v2"] = record.get("ground_truth", "")
            new_record["reference_answer_valid"] = True
            new_record["reference_validation_reason"] = "valid"
            new_record["reference_answer_source_v2"] = "ground_truth"
        output_records.append(new_record)

    return output_records


def source_distribution_rows(
    records: List[Dict],
    source_field: str,
    reference_field: str,
) -> List[Dict[str, object]]:
    rows = []
    counter = Counter(record.get(source_field, "") or "missing" for record in records)
    for source, count in sorted(counter.items()):
        rows.append(
            {
                "metric": "source_count",
                "reference_field": reference_field,
                "source": source,
                "reason": "",
                "value": count,
            }
        )
    return rows


def invalid_reference_count(
    records: List[Dict],
    reference_field: str,
    source_field: str,
    max_evidence_fallback_tokens: int,
) -> int:
    invalid = 0
    for record in records:
        if reference_field == "reference_answer_v2" and "reference_answer_valid" in record:
            invalid += int(not bool(record["reference_answer_valid"]))
            continue

        validation = validate_reference_answer(
            answer=str(record.get(reference_field, "")),
            question=str(record.get("question", "")),
            evidence=str(record.get("reference_evidence", record.get("ground_truth", ""))),
            source=str(record.get(source_field, "")),
            max_evidence_fallback_tokens=max_evidence_fallback_tokens,
        )
        invalid += int(not validation["reference_answer_valid"])
    return invalid


def long_evidence_fallback_count(
    records: List[Dict],
    source_field: str,
    max_evidence_fallback_tokens: int,
) -> int:
    return sum(
        1
        for record in records
        if record.get(source_field) == "nq_evidence_sentence"
        and reference_token_count(record.get("reference_evidence", record.get("ground_truth", "")))
        > max_evidence_fallback_tokens
    )


def quality_metric_row(metric: str, reference_field: str, value: int) -> Dict[str, object]:
    return {
        "metric": metric,
        "reference_field": reference_field,
        "source": "",
        "reason": "",
        "value": value,
    }


def build_reference_quality_report(
    records: List[Dict],
    max_evidence_fallback_tokens: int = DEFAULT_MAX_EVIDENCE_FALLBACK_TOKENS,
) -> List[Dict[str, object]]:
    if not records or not any("reference_answer_v2" in record for record in records):
        return []

    rows: List[Dict[str, object]] = [
        quality_metric_row("num_records", "all", len(records)),
    ]

    field_specs = [
        ("reference_answer", "reference_answer_source"),
        ("reference_answer_v2", "reference_answer_source_v2"),
    ]

    for reference_field, source_field in field_specs:
        if not any(reference_field in record for record in records):
            continue

        rows.extend(
            [
                quality_metric_row(
                    "pronoun_reference_count",
                    reference_field,
                    sum(
                        1
                        for record in records
                        if (
                            len(tokens := normalized_answer_tokens(record.get(reference_field, "")))
                            == 1
                            and tokens[0] in PRONOUN_REFERENCE_TOKENS
                        )
                    ),
                ),
                quality_metric_row(
                    "one_token_suspicious_reference_count",
                    reference_field,
                    sum(
                        1
                        for record in records
                        if one_token_suspicious_reference(
                            str(record.get(reference_field, "")),
                            str(record.get("question", "")),
                        )
                    ),
                ),
                quality_metric_row(
                    "long_evidence_fallback_count",
                    reference_field,
                    long_evidence_fallback_count(
                        records,
                        source_field,
                        max_evidence_fallback_tokens,
                    ),
                ),
                quality_metric_row(
                    "invalid_reference_count",
                    reference_field,
                    invalid_reference_count(
                        records,
                        reference_field,
                        source_field,
                        max_evidence_fallback_tokens,
                    ),
                ),
            ]
        )
        rows.extend(source_distribution_rows(records, source_field, reference_field))

    reason_counter = Counter(
        record.get("reference_validation_reason", "") or "missing"
        for record in records
    )
    for reason, count in sorted(reason_counter.items()):
        rows.append(
            {
                "metric": "validation_reason_count",
                "reference_field": "reference_answer_v2",
                "source": "",
                "reason": reason,
                "value": count,
            }
        )

    return rows


def resolve_reference_field(config: Dict) -> str:
    reference_field = config.get("evaluation", {}).get("reference_field", "auto")
    if reference_field and reference_field != "auto":
        return reference_field
    if config.get("data", {}).get("dataset") == "nq":
        return "reference_answer"
    return "ground_truth"
