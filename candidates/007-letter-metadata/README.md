# Letter metadata with an LLM: extraction and authority IDs

Two steps towards [CMIF](https://github.com/TEI-Correspondence-SIG/CMIF) metadata for
transcribed 19th-century letters (German, French, English, Italian) from the Hammer-Purgstall
correspondence (1794-1856), solved with an LLM through the cluster's API. Each has its own
notebook:

1. **Extraction.** Plain-text letters go in. The writer, the date, the place of writing and the
   language come out as normalized JSON and are compared with the metadata of the edition.
2. **Authority IDs.** Where a lookup in the GND or in GeoNames returned several candidates, the
   model picks one, using the letter text. The final IDs are compared with the reference and
   written as CMIF `<correspDesc>` elements.

The pipeline and its prompts are described in a paper that is not yet published.

## A short example

Notebook 1 gets a letter (here `letter1385`, written in Munich and signed "Dr. Frank") and
returns, shortened:

```json
{
  "place":    {"value": "Munich | DE", "cert": "high"},
  "date":     {"value": "1812-12-07", "cert": "high"},
  "writer":   {"forename": {"value": "unk", "cert": "high"},
               "lastname": {"value": "Frank", "cert": "high"}},
  "language": {"value": "de", "cert": "high"}
}
```

The GND lookup returned fourteen candidates named Frank, all alive in 1812. Notebook 2 gives the
model the candidates and the letter, and the model answers:

```json
[{"id": 1385, "writer": "Frank, Othmar", "gnd": "116704128", "cert": "high"}]
```

Both answers come from the original run of the pipeline with `qwen3.5-397b` on the cluster, not
from the notebooks' default model.

## What you need

- **Session:** JupyterHub, CPU session, or any Python environment that can reach the API. No GPU
  is needed: the model runs on the cluster and the notebooks only send requests to it.
- **Resources:** minimal RAM and disk. Notebook 1 sends eleven requests (one letter on its own,
  then all ten), notebook 2 seven (one choice on its own, then all six).
- **Software:** Python 3.9+. The first notebook cell installs `openai`, `python-dotenv`,
  `lxml` and `pandas` from `requirements.txt`.
- **Access:** a project token from [console.dhinfra.uni-graz.at](https://console.dhinfra.uni-graz.at).

**Tested:** not yet run against the cluster's API.

The API is an OpenAI-compatible endpoint, which is why the standard `openai` package works.

## Quickstart

```bash
cd candidates/007-letter-metadata
echo 'DHINFRA_KEY="your-token"' > .env      # stays out of git, see below
```

Then open `01_extract_metadata.ipynb` and run all cells, then `02_link_authorities.ipynb`. The
second notebook does not need the output of the first: it starts from saved results (see Data).

## What's inside

| File | What it is |
|---|---|
| `01_extract_metadata.ipynb` | notebook 1: writer, date, place and language from the letter text |
| `02_link_authorities.ipynb` | notebook 2: choose GND and GeoNames records, write CMIF |
| `helpers.py` | code both notebooks import: prompt assembly, the model call, scoring, `correspDesc` |
| `prompts/extraction/` | task, rules, four examples for notebook 1 |
| `prompts/disambiguation/` | task, rules, examples for notebook 2 |
| `data/letters/` | ten sample letters as plain text |
| `data/reference.json` | the edition's metadata for the ten letters, with GND and GeoNames IDs |
| `data/lookups.json` | saved results of the first step and of the GND and GeoNames lookups |
| `output/` | results, one folder per model |
| `requirements.txt` | Python packages |

## Settings to change before you run

- `DHINFRA_KEY` in `.env` (or the environment): your token. Do not commit the `.env` file.
- `MODEL` in the settings cell of each notebook, if you want another model than
  `deepseek-v4.1-flash`. The notebook prints the models your token can use; the available models
  change, so pick one from that list.
- `THINKING`: `True` lets the model reason before it answers. Slower, more tokens.
- `DHINFRA_BASE_URL` (optional environment variable): defaults to
  `https://api.dhinfra.uni-graz.at/v1`.

Nothing else points at the author's setup; all paths are relative to the notebooks.

## What it produces

- Notebook 1: `output/<model>/extraction.json`, the answer for each letter. The notebook shows a
  table that compares language and date exactly and the writer by surname with the reference.
  The place is shown but not scored, because the model writes today's English name ("Munich")
  and the edition the German one ("München").
- Notebook 2: `output/<model>/choices.json`, the model's answer for each of the six ambiguous
  fields, and `output/<model>/correspDesc.xml`, a TEI `<profileDesc>` with one `<correspDesc>`
  per letter. The notebook shows a table that compares the final GND and GeoNames IDs with the
  reference.

Both notebooks sort each result into one of a few outcomes: correct, correct `unk` (the letter
does not give the value), missed, wrong, and "check" for a value the reference cannot confirm.
Ten letters are a demonstration, not an evaluation.

## Data

**Edition.** All texts and the reference metadata come from: Höflechner, Walter; Scholger,
Martina; Wagner, Alexandra; Strutz, Sabrina; Steiner, Elisabeth (eds.) 2025: Joseph von
Hammer-Purgstall. Korrespondenz. Graz. <https://gams.uni-graz.at/hpe>,
<http://hdl.handle.net/11471/559.20> (GAMS 559.20).

**Letters.** Ten letters from the sample of 100 that was drawn from the edition to evaluate the
pipeline (90 by stratified random sampling, 10 chosen for special cases). They were chosen to
cover four languages, letters with and without dateline and signature, and the cases where the
lookups returned several candidates.

**Reference.** `data/reference.json` has, for each letter, the writer with GND number, the
place with GeoNames ID, the date and the language from the edition's metadata. Two flags were
added during the evaluation: whether the letter text itself gives the writer and the date
(`in_letter`). For the place, the flag is `false` when the edition leaves the place open or gives
it only in square brackets, as inferred by the editors.

**Lookups.** `data/lookups.json` has the result of the first step from the original run
(`qwen3.5-397b`, thinking mode, July 2026) and what the GND (queried through
[lobid-gnd](https://lobid.org/gnd)) and [GeoNames](https://www.geonames.org) returned for
those values.

## Licence

- Code (`helpers.py`, the notebooks): Apache-2.0, the repository default.
- Letters, reference and prompts: CC BY-NC 4.0 (non-commercial use), see `LICENSE.txt`. This is
  the licence of the edition the letters come from (GAMS 559.20).
- GND records in `data/lookups.json`: Deutsche Nationalbibliothek, CC0 1.0, retrieved through
  lobid-gnd.
- GeoNames records in `data/lookups.json`: GeoNames, <https://www.geonames.org>, CC BY 4.0.

## Credits

Letters and reference: Höflechner et al. (edition). Authority data: GND (Deutsche
Nationalbibliothek, via lobid-gnd, hbz) and GeoNames. Model for the default settings: DeepSeek
V4.1 Flash; model for the saved first step and the examples above: Qwen3.5-397B. Both served by
the DHinfra cluster.

The model was served by the cluster of Digital Humanities Infrastructure Austria (DHinfra.at). The
GPU systems are described in:

> Atzenhofer-Baumgartner, Florian, David Fleischhacker, Max Resch, Lukas Waldhofer, and
> Michael Otto. 2026. "Design and Operation of a Federated GPU Cluster for Digital
> Humanities within DHinfra.at." arXiv:2609.10552 [cs.DC].
> <https://doi.org/10.48550/arXiv.2609.10552>
