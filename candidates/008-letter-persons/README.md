# Person mentions in letters with an LLM: word positions or TEI

Marking every person in transcribed 19th-century letters (German, English, Italian) from the
TEI edition of the Hammer-Purgstall correspondence (1794-1856), with an LLM through the
cluster's API. The edition marks persons only in part: often only the surname is a
`<persName>`, some names are split over two elements, and pronouns are not marked at all. The
task is to mark every mention of a real person with its name parts (`roleName`, `forename`,
`nameLink`, `surname` ...), and every pronoun or phrase that refers to one as `<rs>`.

The same task is asked in two ways, one notebook each:

1. **Word positions.** The model gets the letter text numbered word by word and answers with word
   numbers only. A script sets the tags, so the letter text cannot change.
2. **TEI.** The model gets the TEI fragment and returns it annotated. The notebook checks whether
   the letter text and the footnotes came back unchanged, and compares the mentions with those of
   notebook 1.

The pipeline and its prompts are described in a paper that is not yet published.

## A short example

Notebook 1 sends `letter7368` (here the start and the end) numbered word by word:

```
[0]Mein [1]guter [2]Baron [3]Hammer, [4]wissend [5]wie [6]sehr [7]Sie [8]die [9]Russen ...
... [48]Ihr [49]Freund [50][sic] [51]Anna [52]Iwanowna [53]Bland [54]geb. [55]Schmidt [56][Smith]
```

The edition marks `Hammer`, `Bland ` and ` Schmidt [Smith]` as three `<persName>`. The model
answers with positions, shortened:

```json
{"annotations": [
  {"element": "rs", "tok_start": 0, "tok_end": 0},
  {"element": "persName", "tok_start": 2, "tok_end": 3, "suffix": ",",
   "parts": [{"element": "roleName", "tok_start": 2, "tok_end": 2},
             {"element": "surname", "tok_start": 3, "tok_end": 3, "suffix": ","}]},
  {"element": "persName", "tok_start": 51, "tok_end": 55,
   "parts": [{"element": "forename", "tok_start": 51, "tok_end": 51},
             {"element": "forename", "tok_start": 52, "tok_end": 52},
             {"element": "surname", "tok_start": 53, "tok_end": 53},
             {"element": "surname", "tok_start": 55, "tok_end": 55, "type": "maiden"}]}
]}
```

and the script writes:

```xml
<rs>Mein</rs> guter <persName><roleName>Baron</roleName> <surname>Hammer</surname></persName>, ...
<persName><forename>Anna</forename> <forename>Iwanowna</forename> <surname>Bland</surname>
geb. <surname type="maiden">Schmidt</surname></persName> [Smith]
```

The answer comes from a run of the pipeline with `qwen3.5-397b` on the cluster, not from the
notebooks' default model. The same answer also marks "Russen" and "Mongolen" as `<rs>`, which
the rules exclude (collective references): the checks in the notebooks catch formal errors, not
this kind.

## What you need

- **Session:** JupyterHub, CPU session, or any Python environment that can reach the API. No GPU
  is needed: the model runs on the cluster and the notebooks only send requests to it.
- **Resources:** minimal RAM and disk. Each notebook sends eleven requests: one letter on its own,
  then all ten.
- **Software:** Python 3.9+. The first notebook cell installs `openai`, `python-dotenv`,
  `lxml` and `pandas` from `requirements.txt`.
- **Access:** a project token from [console.dhinfra.uni-graz.at](https://console.dhinfra.uni-graz.at).

**Tested:** not yet run against the cluster's API.

The API is an OpenAI-compatible endpoint, which is why the standard `openai` package works.

## Quickstart

```bash
cd candidates/008-letter-persons
echo 'DHINFRA_KEY="your-token"' > .env      # stays out of git, see below
```

Then open `01_persons_by_position.ipynb` and run all cells, then `02_persons_in_tei.ipynb` with
the same `MODEL`. Section 6 of notebook 2 compares the two.

## What's inside

| File | What it is |
|---|---|
| `01_persons_by_position.ipynb` | notebook 1: the model answers with word positions |
| `02_persons_in_tei.ipynb` | notebook 2: the model returns annotated TEI; comparison with notebook 1 |
| `helpers.py` | code both notebooks import: letter text from TEI, prompt assembly, the model call, the checks |
| `prompts/positions/` | task, encoding rules, answer format and example for notebook 1 |
| `prompts/tei/` | the same for notebook 2 |
| `data/letters/` | ten letter bodies (`<div type="letter">`) as they are in the edition |
| `data/metadata.json` | writer, recipient (with the IDs of the edition's person index), date and place per letter |
| `output/` | results, one folder per model |
| `requirements.txt` | Python packages |

## Settings to change before you run

- `DHINFRA_KEY` in `.env` (or the environment): your token. Do not commit the `.env` file.
- `MODEL` in the settings cell of each notebook, if you want another model than
  `deepseek-v4.1-flash`. The notebook prints the models your token can use; the available models
  change, so pick one from that list. Use the same model in both notebooks to compare them.
- `THINKING`: `True` lets the model reason before it answers. Slower, more tokens.
- `DHINFRA_BASE_URL` (optional environment variable): defaults to
  `https://api.dhinfra.uni-graz.at/v1`.

Nothing else points at the author's setup; all paths are relative to the notebooks.

## What it produces

- Notebook 1: `output/<model>/positions/<letter>.json` (the mentions as character positions in the
  letter text, the corrections the script made, the token counts) and `<letter>.xml` (the letter
  text with the mentions as inline TEI). The notebook shows a table: how many mentions, how many
  corrections, and what happened to the edition's `persName` markup (kept, extended, merged,
  removed, added).
- Notebook 2: `output/<model>/tei/<letter>.xml` (the fragment as the model returned it) and
  `<letter>.json` (the checks and the mentions). The notebook shows whether each answer is
  well-formed, whether the letter text and the footnotes are unchanged, the same comparison with
  the edition's markup, and, per letter, which mentions both notebooks found and the output tokens
  of both.

There is no reference annotation in this example. The checks say whether an answer is usable and
what it changed, not whether it is right. Read the results as well.

The full pipeline goes further: it writes the mentions back into the edition's TEI files and links
each one to the person index or the GND. Neither step is part of this example.

## Data

**Edition.** All texts and metadata come from: Höflechner, Walter; Scholger, Martina; Wagner,
Alexandra; Strutz, Sabrina; Steiner, Elisabeth (eds.) 2025: Joseph von Hammer-Purgstall.
Korrespondenz. Graz. <https://gams.uni-graz.at/hpe>, <http://hdl.handle.net/11471/559.20>
(GAMS 559.20).

**Letters.** Ten letters from a sample drawn from the edition to evaluate the pipeline, chosen to
cover three languages and letters by and to Hammer-Purgstall. `data/letters/` holds the letter
body of each (`<div type="letter" subtype="original">`) as it is in the edition, with its
footnotes and its existing markup. `data/metadata.json` holds the correspondence metadata of the
edition for these letters.

**Prompts.** `prompts/positions/` and `prompts/tei/` are the prompts for the two variants of the
tagging step in the author's pipeline.

## Licence

- Code (`helpers.py`, the notebooks): Apache-2.0, the repository default.
- Letters, metadata and prompts: CC BY-NC 4.0 (non-commercial use), see `LICENSE.txt`. This is the
  licence of the edition the letters come from (GAMS 559.20).

## Credits

Letters and metadata: Höflechner et al. (edition). Model for the default settings: DeepSeek V4.1
Flash; model for the example above: Qwen3.5-397B. Both served by the DHinfra cluster.

The model was served by the cluster of Digital Humanities Infrastructure Austria (DHinfra.at). The
GPU systems are described in:

> Atzenhofer-Baumgartner, Florian, David Fleischhacker, Max Resch, Lukas Waldhofer, and
> Michael Otto. 2026. "Design and Operation of a Federated GPU Cluster for Digital
> Humanities within DHinfra.at." arXiv:2609.10552 [cs.DC].
> <https://doi.org/10.48550/arXiv.2609.10552>
