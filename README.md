# Pipeline

Five scripts, run in order from inside this folder. Each validates itself
before printing its report and exits with a non-zero status if a check fails.

| Script | What it produces | Checks |
|---|---|---|
| `build_phase1.py` | validated object catalogue, five operational classes, 40 shells of 35 km from 200 to 1600 km, satellite-year panel | 9 |
| `build_phase2.py` | launch cadence by country and year, active payload density over time, operator concentration | 6 |
| `build_history.py` | 1957 to 2015 satellite and debris series, checked against Rao, Burgess and Kaffine | 6 |
| `build_phase3.py` | collision risk per shell, satellite value from the OPUS cost function, Pigouvian charge split into congestion and pollution | 8 |
| `export_web_data.py` | CSV exports already sit in `data/processed/`; this writes the compact JSON payload the website reads to `web/data.json` | 0, export only |

29 checks in total. Three of them, two in `build_phase1.py` and one in
`build_history.py`, compare against small external reference files that are
not included in this repository, a replication series from Rao, Burgess and
Kaffine and a set of OPUS reference bins. Both checks are wrapped so they skip
gracefully when the file is absent, which means a fresh clone runs 26 of the
29 checks out of the box.

## How the scripts find their own project root

Each script computes `ROOT = Path(__file__).resolve().parents[1]`, the
grandparent folder of the script itself. Since these scripts live in
`pipeline/`, that resolves to the repository root, one level up. From there
the scripts expect:

```
<repo root>/
├── src/gcat.py, src/physics.py     imported directly, must exist
├── data/raw/                       the six GCAT TSV files, you provide these
├── data/processed/                 every CSV, created automatically
└── web/data.json                   created automatically by export_web_data.py
```

None of `data/`, `web/`, or their contents are committed to this repository.
They are created the first time you run the pipeline locally.

## Inputs

Six GCAT TSV files, `currentcat.tsv`, `satcat.tsv`, `psatcat.tsv`,
`launchlog.tsv`, `orgs.tsv`, `active.tsv`. Place them in `data/raw/` at the
repository root before running `build_phase1.py`. Not committed here, archived
with the Zenodo record instead. Source https://planet4589.org/space/gcat,
CC BY 4.0.

## Publishing an update

`export_web_data.py` writes to `web/data.json`, a local build folder, not the
live site. To publish a refreshed snapshot, copy `web/data.json` together with
the relevant files from `data/processed/`, `risk_by_shell.csv`,
`tax_by_operator.csv`, `shell_population.csv`, over the copies sitting at the
root of this repository, then commit.

## Notes that cost time to learn

- GCAT `Active` encodes `A` for an operational payload and `P` for a derelict
  one. It is not a payload type field.
- `Control` in the payload catalogue is the ground control centre. Country of
  operator comes from `SatState` in the launch log.
- For debris fragments `SDate` is the separation date and `LDate` is the parent
  launch date. Conflating them inflates historical debris counts by 2.6.
- The MOCAT 4S untracked debris multiplier `delta = 10` must be included.
  Leaving it out lowers the weight attributed to an inert object by a factor
  of about fifty.
- The published OPUS `buildCostFunction.m` mixes km/s and m/s. Follow Table 2
  of the paper, metres per second throughout.
