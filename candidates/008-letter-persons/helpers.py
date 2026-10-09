"""Helpers for the two notebooks: the letter text from TEI, prompts, one model call, the checks."""

import bisect
import hashlib
import json
import os
import re
import unicodedata
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
    """Remove <think> blocks and a surrounding ``` fence, if the model added them."""
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


# ---------------------------------------------------------------------------
# The letter text: plain text of the TEI letter body, without footnotes
# ---------------------------------------------------------------------------

TEI_NS = "http://www.tei-c.org/ns/1.0"

# Not part of the letter text: footnotes, and the editorial reading inside <choice>.
SKIP = {"note", "expan", "reg", "corr"}
# Elements that end a word: without this, "Diener</salute><signed>Böttiger" would become one word.
BLOCKS = {"div", "p", "ab", "opener", "closer", "salute", "signed", "dateline", "head", "lg", "l",
          "list", "item", "table", "row", "cell", "postscript", "note", "graphic", "figure"}
# Markup the edition may already have, listed for the model as "existing markup".
EXISTING = {"persName", "rs", "name", "orgName", "placeName"}
# Blocks that hold text themselves; a mention has to fit into one of them to be written into the TEI.
TEXT_BLOCKS = BLOCKS - {"div", "opener", "closer", "list", "table", "row"}
TOP = {"persName", "rs"}
PARTS = {"forename", "surname", "roleName", "nameLink", "genName", "addName", "name"}
GUARD = "\x1f"   # where a footnote was taken out: a space only if it stands between two words


def localname(el):
    return etree.QName(el.tag).localname


def read_letter(path):
    """The <div type="letter"> element of a letter file."""
    return etree.parse(str(path)).getroot()


def fragment_xml(div):
    """The letter as the TEI variant sends it to the model: as in the file, without the namespace."""
    xml = etree.tostring(div, encoding="unicode", with_tail=False)
    return re.sub(r'\sxmlns="[^"]*"', "", xml, count=1)


def letter_text(div):
    """Plain text of a letter body, the words numbered, the markup and footnotes it already has.

    Returns a dict: text, tokens [{i, t, start, end}], spans (every element with its character
    range and its path, so that nesting can be checked), notes [{after_token, text}].
    Footnotes are left out of the text, block elements end a word, whitespace is collapsed.
    """
    chars, spans, notes = [], [], []

    def walk(el, path):
        start = len(chars)
        chars.extend(el.text or "")
        counts = {}
        for child in el:
            if not isinstance(child.tag, str):   # comment or processing instruction
                chars.extend(child.tail or "")
                continue
            name = localname(child)
            counts[name] = counts.get(name, 0) + 1
            child_path = f"{path}/{name}[{counts[name]}]"
            anchor = len(chars)
            if name in SKIP:
                chars.append(GUARD)
                if name == "note":
                    notes.append({"raw": anchor, "node": child,
                                  "text": " ".join("".join(child.itertext()).split())})
            else:
                block = name in BLOCKS
                if block:
                    chars.append("\n")
                walk(child, child_path)
                if block:
                    chars.append("\n")
            chars.extend(child.tail or "")
        spans.append({"element": localname(el), "path": path, "raw": (start, len(chars)),
                      "type": el.get("type")})

    walk(div, "/div")

    # Collapse whitespace; keep for each output character its position in `chars`.
    out, nmap, prev_space, guard = [], [], True, False
    for i, ch in enumerate(chars):
        if ch == GUARD:
            guard = True
            continue
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                nmap.append(i)
                prev_space = True
            guard = False
            continue
        if guard and not prev_space and out[-1].isalnum() and ch.isalnum():
            out.append(" ")
            nmap.append(i)
        guard = False
        out.append(ch)
        nmap.append(i)
        prev_space = False
    while out and out[-1] == " ":
        out.pop()
        nmap.pop()
    text = "".join(out)

    tokens = [{"i": i, "t": m.group(), "start": m.start(), "end": m.end()}
              for i, m in enumerate(re.finditer(r"\S+", text))]
    mapped = []
    for s in spans:
        start = bisect.bisect_left(nmap, s["raw"][0])
        end = bisect.bisect_left(nmap, s["raw"][1])
        if end > start:
            mapped.append({"element": s["element"], "path": s["path"], "start": start,
                           "end": end, "text": text[start:end], "type": s["type"]})
    mapped.sort(key=lambda s: (s["start"], -s["end"]))
    for n in notes:
        anchor = min(bisect.bisect_left(nmap, n["raw"]), len(text))
        n["after_token"] = max((t["i"] for t in tokens if t["end"] <= anchor), default=None)
        del n["raw"]
    return {"text": text, "tokens": tokens, "spans": mapped, "notes": notes}


def token_range(tokens, start, end):
    """Character range -> (first token, last token)."""
    first = next(t["i"] for t in tokens if t["end"] > start)
    last = next(t["i"] for t in reversed(tokens) if t["start"] < end)
    return first, last


def existing_markup(letter):
    """The person (and place, org) markup the edition already has, with token ranges."""
    out = []
    for s in letter["spans"]:
        if s["element"] in EXISTING:
            first, last = token_range(letter["tokens"], s["start"], s["end"])
            out.append({**s, "tok_start": first, "tok_end": last})
    return out


def metadata_block(meta):
    """Writer, recipient, date and place of the letter, as both prompts show them."""
    w, r = meta["writer"], meta["recipient"]
    lines = [f"Letter ID: {meta['letter_id']}",
             f"Writer: {w['name']} (ID {w['id']})",
             f"Recipient: {r['name']} (ID {r['id']})"]
    if meta.get("date"):
        lines.append(f"Date: {meta['date']}")
    if meta.get("place"):
        lines.append(f"Place: {meta['place']}")
    return "\n".join(lines)


def positions_message(meta, letter):
    """User message for notebook 1: metadata, the numbered text, existing markup, footnotes."""
    numbered = " ".join(f"[{t['i']}]{t['t']}" for t in letter["tokens"])
    existing = []
    for s in existing_markup(letter):
        rng = str(s["tok_start"]) if s["tok_start"] == s["tok_end"] else f"{s['tok_start']}-{s['tok_end']}"
        existing.append("  %-9s Token %-8s %r" % (s["element"], rng, s["text"]))
    notes = [f"  [after token {n['after_token']}] {n['text']}" for n in letter["notes"]]
    return (f"=== LETTER METADATA ===\n{metadata_block(meta)}\n\n"
            f"=== LETTER TEXT, NUMBERED WORD BY WORD ===\n{numbered}\n\n"
            f"=== EXISTING MARKUP ===\n" + ("\n".join(existing) or "  (none)") + "\n\n"
            f"=== FOOTNOTES (context only, do not annotate) ===\n" + ("\n".join(notes) or "  (none)") + "\n")


def tei_message(meta, div):
    """User message for notebook 2: metadata and the TEI fragment itself."""
    return f"=== LETTER METADATA ===\n{metadata_block(meta)}\n\n=== TEI FRAGMENT ===\n{fragment_xml(div)}\n"


# ---------------------------------------------------------------------------
# Notebook 1: token positions -> character spans
# ---------------------------------------------------------------------------

def parse_json_answer(raw):
    """The answer should be a JSON object {"annotations": [...]}. Returns the list, or None."""
    text = clean_output(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data.get("annotations") if isinstance(data, dict) else None


# A closing bracket is kept when its opener is in the span: "v[on]" keeps its "]".
BRACKETS = {")": "(", "]": "[", "}": "{"}


def _trim_punct(text, start, end):
    """Pull the end in past trailing punctuation ("Joris," -> "Joris"). A full stop stays:
    it may abbreviate ("C.") - the model marks a sentence-final one with "suffix"."""
    e = end
    while e > start:
        ch = text[e - 1]
        if ch == "." or not unicodedata.category(ch).startswith("P"):
            break
        if ch in BRACKETS and BRACKETS[ch] in text[start:e]:
            break
        e -= 1
    return e if e > start else end


def _trim_suffix(entry, text, start, end, issues, where):
    """Cut the "suffix" the model named ("Wielands" -> "Wieland"), but only if it is really there."""
    suffix = entry.get("suffix")
    if not suffix:
        return end
    for e in dict.fromkeys((end, _trim_punct(text, start, end))):
        if text[start:e].endswith(suffix) and e - start > len(suffix):
            return e - len(suffix)
    issues.append(f"{where}: suffix {suffix!r} is not at the end of {text[start:end]!r}, not cut")
    return end


def _token_bounds(entry, n, issues, where):
    try:
        first, last = int(entry["tok_start"]), int(entry["tok_end"])
    except (KeyError, TypeError, ValueError):
        issues.append(f"{where}: tok_start/tok_end missing, rejected")
        return None
    if first > last:
        first, last = last, first
        issues.append(f"{where}: start and end swapped, corrected")
    if first < 0 or last >= n:
        issues.append(f"{where}: tokens {first}-{last} are outside 0-{n - 1}, rejected")
        return None
    return first, last


def positions_to_spans(answer, letter):
    """The model's token positions as character spans in the letter text. Returns (spans, issues).

    Checks what a model tends to get wrong: positions outside the text, parts outside their
    persName, a mention that runs from one text block into the next (from <salute> into <p>),
    spans that overlap (XML cannot hold them; the longer one wins). Nothing is repaired
    silently; every correction is listed in `issues`."""
    text, tokens = letter["text"], letter["tokens"]
    blocks = [s for s in letter["spans"] if s["element"] in TEXT_BLOCKS]
    issues, found = [], []
    for k, entry in enumerate(answer or [], 1):
        where = f"entry {k}"
        if not isinstance(entry, dict) or entry.get("element") not in TOP:
            issues.append(f"{where}: not a persName or rs, rejected")
            continue
        bounds = _token_bounds(entry, len(tokens), issues, where)
        if bounds is None:
            continue
        start, end = tokens[bounds[0]]["start"], tokens[bounds[1]]["end"]
        end = _trim_punct(text, start, _trim_suffix(entry, text, start, end, issues, where))
        parts = []
        for part in entry.get("parts") or []:
            if not isinstance(part, dict) or part.get("element") not in PARTS:
                issues.append(f"{where}: unknown part {part!r}, skipped")
                continue
            pb = _token_bounds(part, len(tokens), issues, where + " (part)")
            if pb is None:
                continue
            if pb[0] < bounds[0] or pb[1] > bounds[1]:
                issues.append(f"{where}: part {part['element']} sticks out of the {entry['element']}, skipped")
                continue
            ps, pe = tokens[pb[0]]["start"], tokens[pb[1]]["end"]
            pe = _trim_punct(text, ps, _trim_suffix(part, text, ps, pe, issues, where + " (part)"))
            if parts and ps < parts[-1]["end"]:
                issues.append(f"{where}: parts overlap, {part['element']} skipped")
                continue
            parts.append({"element": part["element"], "start": ps, "end": pe,
                          "text": text[ps:pe], "type": part.get("type")})
        if entry["element"] == "rs" and parts:
            issues.append(f"{where}: rs has parts, removed")
            parts = []
        if not any(b["start"] <= start and end <= b["end"] for b in blocks):
            issues.append(f"{where}: {entry['element']} {text[start:end]!r} runs over the end of a "
                          f"text block (e.g. from <salute> into <p>), rejected")
            continue
        found.append({"element": entry["element"], "start": start, "end": end,
                      "text": text[start:end], "parts": parts})

    # Same span twice: keep one. Overlapping spans: the longer one wins.
    accepted, seen = [], set()
    for span in sorted(found, key=lambda s: (-(s["end"] - s["start"]), s["start"])):
        key = (span["element"], span["start"], span["end"])
        if key in seen:
            issues.append(f"reported twice: {span['element']} {span['text']!r}")
            continue
        clash = next((a for a in accepted if span["start"] < a["end"] and a["start"] < span["end"]), None)
        if clash:
            issues.append(f"overlap: {span['element']} {span['text']!r} yields to {clash['text']!r}")
            continue
        seen.add(key)
        accepted.append(span)
    accepted.sort(key=lambda s: s["start"])
    return accepted, issues


def spans_to_tei(text, spans):
    """The letter text with the spans as inline TEI, inside an <ab>. The text itself is copied, never typed."""
    events = []   # (position, order, markup); closing tags sort before opening tags at the same position
    for depth, items in ((0, spans), (1, [p for s in spans for p in s.get("parts", [])])):
        for s in items:
            attr = f' type="{s["type"]}"' if s.get("type") else ""
            events.append((s["start"], 2 + depth, f"<{s['element']}{attr}>"))
            events.append((s["end"], 1 - depth, f"</{s['element']}>"))
    out, pos = [], 0
    for at, _, tag in sorted(events, key=lambda e: (e[0], e[1])):
        out.append(_escape(text[pos:at]))
        out.append(tag)
        pos = at
    out.append(_escape(text[pos:]))
    return f'<ab xmlns="{TEI_NS}">' + "".join(out) + "</ab>"


def _escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_of(xml):
    """The plain text of an XML string, for the check that the letter text is unchanged."""
    return "".join(etree.fromstring(xml.encode("utf-8")).itertext())


def compare_with_edition(letter, spans):
    """What the model did with the persName markup the edition already had.

    kept: same span. extended: overlaps one old span, different boundaries. merged: covers
    several old spans. removed: an old span the model did not keep. added: no old span there."""
    old = [s for s in existing_markup(letter) if s["element"] == "persName"]
    new = [s for s in spans if s["element"] == "persName"]
    overlap = lambda a, b: a["start"] < b["end"] and b["start"] < a["end"]   # noqa: E731
    result = {"kept": 0, "extended": 0, "merged": 0, "removed": 0, "added": 0}
    for s in new:
        hits = [o for o in old if overlap(o, s)]
        if not hits:
            result["added"] += 1
        elif len(hits) > 1:
            result["merged"] += 1
        elif (hits[0]["start"], hits[0]["end"]) == (s["start"], s["end"]):
            result["kept"] += 1
        else:
            result["extended"] += 1
    result["removed"] = sum(1 for o in old if not any(overlap(o, s) for s in new))
    return result


# ---------------------------------------------------------------------------
# Notebook 2: the annotated TEI fragment
# ---------------------------------------------------------------------------

def parse_fragment(raw):
    """The <div> the model returned. Returns (element, how): "strict", "recovered" or "error".

    "recovered" means the XML only parsed with lxml's error recovery: not well-formed."""
    text = clean_output(raw)
    start, end = text.find("<div"), text.rfind("</div>")
    if start == -1 or end < start:
        return None, "error"
    data = text[start:end + len("</div>")].encode("utf-8")
    try:
        return etree.fromstring(data), "strict"
    except etree.XMLSyntaxError:
        pass
    try:
        div = etree.fromstring(data, etree.XMLParser(recover=True, resolve_entities=False))
    except etree.XMLSyntaxError:
        div = None
    return (div, "recovered") if div is not None else (None, "error")


def compare_text(returned, original):
    """identical / whitespace (same characters, different spacing) / changed."""
    if returned == original:
        return "identical"
    if "".join(returned.split()) == "".join(original.split()):
        return "whitespace"
    return "changed"


def first_difference(returned, original, context=60):
    """Where two texts first differ, with some context on both sides. None if they are equal."""
    if returned == original:
        return None
    i = next((k for k, (a, b) in enumerate(zip(returned, original)) if a != b),
             min(len(returned), len(original)))
    lo = max(0, i - context)
    return {"position": i, "original": original[lo:i + context], "returned": returned[lo:i + context]}


def compare_notes(returned, original):
    """What happened to the footnotes: intact / lost / added / changed / annotated."""
    got = [n["text"] for n in returned["notes"]]
    want = [n["text"] for n in original["notes"]]
    if len(got) != len(want):
        return "lost" if len(got) < len(want) else "added"
    if got != want:
        return "changed"
    markup = lambda notes: sum(1 for n in notes for el in n["node"].iter()   # noqa: E731
                               if el is not n["node"] and isinstance(el.tag, str)
                               and localname(el) in TOP | PARTS)
    return "annotated" if markup(returned["notes"]) > markup(original["notes"]) else "intact"


def tei_to_spans(returned, original):
    """persName and rs in the returned fragment, as spans in the ORIGINAL letter text.

    Works when the text is identical or only differs in whitespace: the n-th non-space
    character of one text is the n-th non-space character of the other."""
    src = [i for i, ch in enumerate(returned["text"]) if not ch.isspace()]
    dst = [i for i, ch in enumerate(original["text"]) if not ch.isspace()]

    def project(start, end):
        first, last = bisect.bisect_left(src, start), bisect.bisect_left(src, end) - 1
        return (dst[first], dst[last] + 1) if 0 <= first <= last < len(dst) else None

    tops = [s for s in returned["spans"] if s["element"] in TOP]
    tops = [s for s in tops if not any(s["path"].startswith(o["path"] + "/") for o in tops)]
    spans = []
    for s in tops:
        rng = project(s["start"], s["end"])
        if rng is None:
            continue
        parts = []
        for p in returned["spans"]:
            if p["element"] in PARTS and p["path"].startswith(s["path"] + "/"):
                prng = project(p["start"], p["end"])
                if prng:
                    parts.append({"element": p["element"], "start": prng[0], "end": prng[1],
                                  "text": original["text"][prng[0]:prng[1]], "type": p["type"]})
        spans.append({"element": s["element"], "start": rng[0], "end": rng[1],
                      "text": original["text"][rng[0]:rng[1]], "parts": parts})
    return sorted(spans, key=lambda s: s["start"])
