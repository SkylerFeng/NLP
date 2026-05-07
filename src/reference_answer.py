import re
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


def best_evidence_sentence(question: str, passage: str) -> str:
    sentences = candidate_sentences(passage)
    if not sentences:
        return ""

    qtype = question_type(question)
    q_tokens = question_content_tokens(question)
    ranked = sorted(
        sentences,
        key=lambda sentence: (
            score_sentence(q_tokens, sentence, qtype),
            -abs(len(sentence.split()) - 24),
        ),
        reverse=True,
    )
    return clean_extracted_answer(ranked[0])


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
        else:
            new_record["reference_answer"] = record.get("ground_truth", "")
            new_record["reference_answer_source"] = "ground_truth"
            new_record["reference_evidence"] = record.get("ground_truth", "")
        output_records.append(new_record)

    return output_records


def resolve_reference_field(config: Dict) -> str:
    reference_field = config.get("evaluation", {}).get("reference_field", "auto")
    if reference_field and reference_field != "auto":
        return reference_field
    if config.get("data", {}).get("dataset") == "nq":
        return "reference_answer"
    return "ground_truth"
