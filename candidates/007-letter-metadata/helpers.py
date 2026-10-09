"""Helpers for the two notebooks: prompt assembly, one model call, scoring, correspDesc."""

import difflib
import json
import os
import re
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from lxml import etree
from openai import OpenAI

BASE_URL = os.environ.get("DHINFRA_BASE_URL", "https://api.dhinfra.uni-graz.at/v1")


def get_client():
    """OpenAI client for the DHinfra API. The token is read from DHINFRA_KEY (env or .env)."""
    # override=True: a token in .env wins over one already set in the environment.
    load_dotenv(find_dotenv(usecwd=True), override=True)
    key = os.environ.get("DHINFRA_KEY")
    if not key:
        raise SystemExit(
            "DHINFRA_KEY is not set. Put your project token in a .env file, "
            'as DHINFRA_KEY="...", or export it in your session.'
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def build_system_prompt(prompt_dir):
    """prompt.txt + encoding_rules.txt + few_shot_examples.txt, joined by blank lines."""
    parts = []
    for name in ("prompt.txt", "encoding_rules.txt", "few_shot_examples.txt"):
        parts.append((Path(prompt_dir) / name).read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def clean_output(raw):
    """Remove <think> blocks and a surrounding ```json fence, if the model added them."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n```\s*$", text, flags=re.DOTALL)
    return fence.group(1).strip() if fence else text


def ask_model(client, model, system_prompt, user_message, thinking=False,
              temperature=0.3, max_tokens=8000):
    """One request to the model. Returns (answer_text, usage)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        # vLLM chat-template switch for the thinking mode (models without one ignore it).
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    return clean_output(response.choices[0].message.content or ""), response.usage


def parse_json_answer(raw):
    """The answer should be a JSON object (notebook 1) or a list (notebook 2). None if it is neither."""
    text = clean_output(raw)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    if end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Notebook 1: writer, date, place and language from the letter text
# ---------------------------------------------------------------------------

UNKNOWN = {None, "", "unk", "unknown"}


def field_value(entry):
    """{"value": ..., "cert": ...} -> the value, or None for a missing or "unk" value."""
    value = entry.get("value") if isinstance(entry, dict) else entry
    return None if value in UNKNOWN else str(value).strip()


def writer_name(answer):
    """The structured writer of an answer as one string, "Lastname, Forename particles"; None if unknown."""
    writer = (answer or {}).get("writer") or {}
    last = field_value(writer.get("lastname"))
    fore = field_value(writer.get("forename"))
    particles = field_value(writer.get("particles"))
    if not last and not fore:
        return None
    first = " ".join(p for p in (fore, particles) if p)
    return f"{last or 'unk'}, {first}" if first else (last or "unk")


def place_name(answer):
    """"Vienna | AT" -> "Vienna"; None if unknown."""
    place = field_value((answer or {}).get("place"))
    return place.split("|")[0].strip() if place else None


def _same_surname(model_last, reference_name):
    """Rough check: does the model's surname match the surname in the reference name?

    Compares against each word of the reference surname, so that "Frankl" counts for
    "Frankl von Hochwart", and allows small spelling differences ("Theymer" / "Theimer")."""
    if not model_last or not reference_name:
        return False
    model_last = model_last.lower()
    ref_last = reference_name.split(",")[0].lower()
    words = [w for w in re.split(r"[\s\-]+", ref_last) if w] + [ref_last]
    return any(difflib.SequenceMatcher(None, model_last, w).ratio() >= 0.8 for w in words)


def _verdict(found, matches, in_letter):
    """One of five outcomes for a field.

    correct                  the model's value matches the reference
    correct (not in letter)  the model said "unk", and the reference says the letter does not give it
    missed                   the model said "unk", but the letter gives the value
    wrong                    the model gave a value that does not match
    check (not in letter)    the model gave a value the reference says is not in the letter:
                             maybe inferred from context, maybe made up - read it
    """
    if not found:
        return "missed" if in_letter else "correct (not in letter)"
    if matches:
        return "correct"
    return "wrong" if in_letter else "check (not in letter)"


def score_extraction(answer, reference):
    """Compare one answer of notebook 1 with the reference. Returns one row for a table.

    Date and language are compared exactly. The writer is compared by surname only (rough).
    The place is not scored: the model gives today's English name ("Munich"), the reference the
    German name the edition uses ("München"). Notebook 2 compares places by GeoNames ID instead."""
    answer = answer or {}
    date = field_value(answer.get("date"))
    language = field_value(answer.get("language"))
    last = field_value(((answer.get("writer") or {}).get("lastname")))
    ref_date = reference["date"]["value"]
    return {
        "letter": reference["letter"],
        "language": language,
        "language_ref": reference["language"],
        "language_ok": language == reference["language"],
        "date": date,
        "date_ref": ref_date,
        "date_result": _verdict(date, date == ref_date, reference["date"]["in_letter"]),
        "writer": writer_name(answer),
        "writer_ref": reference["writer"]["name"],
        "writer_result": _verdict(last, _same_surname(last, reference["writer"]["name"]),
                                  reference["writer"]["in_letter"]),
        "place": place_name(answer),
        "place_ref": reference["place"]["name"],
    }


# ---------------------------------------------------------------------------
# Notebook 2: pick the authority record, score the IDs, write correspDesc
# ---------------------------------------------------------------------------

CANDIDATE_KEY = {"writer": "gndCandidates", "place": "geonamesCandidates"}
ID_KEY = {"writer": "gnd", "place": "geonamesId"}


def build_choice_message(entry, field, letter_text):
    """The user message for one ambiguous field: the candidates and the letter, as JSON.

    Same shape as the examples in prompts/disambiguation/few_shot_examples.txt."""
    number = int(entry["letter"].removeprefix("letter"))
    message = {
        "id": number,
        field: entry[field]["query"],
        CANDIDATE_KEY[field]: entry[field]["candidates"],
        "date": field_value(entry["stage1"].get("date")) or "unk",
        "letterText": letter_text.strip(),
    }
    return json.dumps(message, ensure_ascii=False, indent=2)


def read_choice(answer, field):
    """From the model's answer (a list with one object): (chosen ID or "unclear", name, certainty)."""
    if isinstance(answer, dict):
        answer = [answer]
    for item in answer or []:
        if isinstance(item, dict) and ID_KEY[field] in item:
            cert = item.get("cert") or item.get("gnd_cert" if field == "writer" else "geo_cert")
            return str(item[ID_KEY[field]]), item.get(field), cert
    return None, None, None


def has_id(value):
    """True for an actual GND or GeoNames ID, False for None, "unk", "unclear" and the like."""
    return bool(value) and str(value).lower() not in {"unk", "unknown", "unclear", "none", "not_found"}


def score_id(found_id, ref_field):
    """Compare a final GND or GeoNames ID with the reference, the same five outcomes as notebook 1.

    A field without an ID in the reference (a person not in the GND, a place left open by the
    edition) counts as correct when the pipeline gives no ID either."""
    ref_id = ref_field.get("gnd", ref_field.get("geonames"))
    if not has_id(found_id):
        if not ref_id:
            return "correct (no ID)"
        return "missed" if ref_field["in_letter"] else "correct (not in letter)"
    if not ref_id:
        return "check (no ID in reference)"
    return "correct" if str(found_id) == str(ref_id) else "wrong"


TEI_NS = "http://www.tei-c.org/ns/1.0"


def corresp_desc(letter, writer, writer_gnd, place, place_geonames, date):
    """A CMIF <correspDesc> element with the sending side of one letter. Unknown parts are left out."""
    desc = etree.Element(f"{{{TEI_NS}}}correspDesc", nsmap={None: TEI_NS}, key=letter)
    sent = etree.SubElement(desc, f"{{{TEI_NS}}}correspAction", type="sent")
    if writer or has_id(writer_gnd):
        pers = etree.SubElement(sent, f"{{{TEI_NS}}}persName")
        pers.text = writer or "unknown"
        if has_id(writer_gnd):
            pers.set("ref", f"https://d-nb.info/gnd/{writer_gnd}")
    if place or has_id(place_geonames):
        pl = etree.SubElement(sent, f"{{{TEI_NS}}}placeName")
        pl.text = place or "unknown"
        if has_id(place_geonames):
            pl.set("ref", f"https://www.geonames.org/{place_geonames}")
    if date:
        etree.SubElement(sent, f"{{{TEI_NS}}}date", when=date)
    return desc


def profile_desc(descs):
    """Wrap <correspDesc> elements in a TEI <profileDesc>, where CMIF expects them. Returns XML text."""
    root = etree.Element(f"{{{TEI_NS}}}profileDesc", nsmap={None: TEI_NS})
    root.extend(descs)
    return etree.tostring(root, encoding="unicode", pretty_print=True)
