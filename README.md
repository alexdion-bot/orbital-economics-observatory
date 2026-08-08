# Orbital Economics Observatory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21854228.svg)](https://doi.org/10.5281/zenodo.21854228)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

**Website — https://alexdion-bot.github.io/orbital-economics-observatory/**

An open estimate of the external cost imposed by every active satellite in low
Earth orbit, computed from the public catalogue of tracked objects and checked
against the published economic literature.

Every satellite in low Earth orbit raises the probability that every other one
is destroyed. That cost is borne by other operators rather than by the operator
who creates it. This project measures it, prices it, and attributes it.

---

## Headline results

| Quantity | Value |
|---|---|
| Total external cost imposed each year | $184,605,464 |
| Active satellites, 200 to 1600 km | 15,493 |
| Inert objects alongside them | 12,720 |
| Average cost per satellite-year | $11,915 |
| Highest cost shell, 445 to 480 km | $20,440 per satellite-year |
| Operators liable | 748 |

Three findings, in order of how much they revise the published picture.

1. **The cost is almost entirely pollution, not congestion.** The split is
   about $4 against $11,911 per satellite-year. A fee charged for occupying an
   orbital position would miss almost all of the damage. What has to be priced
   is the inert matter left behind.
2. **Three independent estimates agree within a factor of three.** A kinetic
   per-shell calculation, the Rao–Burgess–Kaffine coefficients applied to the
   present orbit, and their published fee trajectory extrapolated to 2026.
3. **The published aggregate model diverges outside its calibration range.**
   Fitted on data ending in 2015, it returns an annual collision probability
   more than three hundred times the physical calculation when applied to
   today's orbit. This is the empirical case for recalibrating on current data.

---

## Repository layout

```
.
├── index.html              the website, payload embedded, no build step
├── data.json               everything the site reads
├── risk_by_shell.csv       per shell population, risk, satellite value, cost by channel
├── tax_by_operator.csv     per operator fleet size, mean altitude, annual liability
├── shell_population.csv    object counts by shell and operational class
├── social-card.png         link preview image
│
├── src/                    the two modules the pipeline is built on
│   ├── gcat.py             reading and normalising the raw GCAT files
│   └── physics.py          the collision risk model and the OPUS cost function
│
├── pipeline/               the five scripts that run in sequence
│   ├── requirements.txt
│   ├── README.md
│   ├── build_phase1.py            catalogue, 40 shells, satellite-year panel
│   ├── build_phase2.py            launch cadence, density, concentration
│   ├── build_history.py           1957-2015 series against Rao, Burgess and Kaffine
│   ├── build_phase3.py            collision risk and the Pigouvian charge
│   └── export_web_data.py         writes data.json for the website
│
├── data/                   created locally when the pipeline runs, not committed
│   ├── raw/                the six downloaded GCAT TSV files
│   └── processed/          every intermediate and final CSV
│
├── web/                    created locally by export_web_data.py, not committed
│   └── data.json           freshly generated payload, copied to the repo root to publish
│
├── CITATION.cff
├── AI-USE.md
└── LICENSE
```

The website itself, the five files at the root plus `social-card.png`, is
fully static. It needs no server, no database and no build step, which is why
it runs on GitHub Pages unchanged. The `src/`, `pipeline/`, `data/` and `web/`
folders exist to make the numbers reproducible, not to run the site.

A detail worth knowing if you run the pipeline yourself: the scripts locate
their own project root as the parent of the folder they sit in, so
`pipeline/build_phase1.py` resolves the project root as this repository's
root, and expects `src/`, `data/` and `web/` as siblings of `pipeline/` at
that same root, exactly as shown above.

---

## Data

Every object comes from **GCAT**, the General Catalog of Artificial Space
Objects compiled by Jonathan C. McDowell, under a CC BY 4.0 licence.

- Source — https://planet4589.org/space/gcat
- Snapshot used — **2 August 2026**
- Files used — current catalogue, full historical catalogue, payload
  catalogue, launch log, organisation table, and the active object list used
  as an independent cross-check

GCAT is updated continuously, so the snapshot date is part of the result. The
raw TSV files are not committed here. They are archived with the Zenodo record
so that a given DOI always resolves to the exact inputs that produced a given
set of outputs.

---

## Method in brief

Low Earth orbit is divided into **40 shells of 35 km** between 200 and 1600 km.
That division is not arbitrary. It is the discretisation used by MOCAT 4S and
by the OPUS integrated assessment model, which keeps every count directly
comparable to the published literature.

Objects are sorted into five operational classes. The decisive distinction
separates an active payload from a derelict one, because a satellite that no
longer operates cannot manoeuvre and behaves like debris.

**Collision rate** in shell *k*, from kinetic theory:

```
lambda_k = [ A * (S_k - 1) + B * D_k ] / V_k
P_k      = 1 - exp(-lambda_k)
```

where `S_k` is the active satellites, `D_k` the inert objects, `V_k` the
geometric shell volume, `A = sigma * v * alpha` for an active satellite that
can manoeuvre, and `B = sigma * v * (1 + delta)` for an inert object that
cannot. The untracked debris multiplier `delta = 10` is taken from MOCAT 4S.
The resulting weight of one inert object relative to one active satellite is
about **320 to 1**.

**Satellite value** comes from the OPUS cost function, reimplemented from its
published specification. It combines launch price, the delta-v budget needed to
hold station against atmospheric drag, and the operating life given up to the
end-of-mission deorbit burn. It is altitude dependent, which is what makes a
price per shell possible.

**The charge** is the external cost a satellite imposes without bearing it.
Both channels are strictly proportional to the number of active satellites
already present, so the cost per satellite grows linearly with occupancy and
the total cost in a shell grows with its square.

```
congestion_k(S) = kc_k * S          kc_k = (A / V_k) * cost_k
pollution_k(S)  = kp_k * S          kp_k = nu * (B / V_k) * cost_k
total_k(S)      = (kc_k + kp_k) * S^2
```

Full detail, including every limitation, is on the Method and Limits pages of
the website.

---

## Reproducing the results

Requires Python 3.11 or later. Tested on Python 3.14 on Windows 10.

```bash
git clone https://github.com/alexdion-bot/orbital-economics-observatory.git
cd orbital-economics-observatory

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS and Linux

pip install -r pipeline/requirements.txt
```

Create `data/raw/` at the repository root and download the six GCAT TSV files
into it, or fetch the exact snapshot from the Zenodo record. Then run the five
scripts from inside `pipeline/`, in order:

```bash
cd pipeline
python build_phase1.py       # validated object catalogue, 40 shells, satellite-year panel
python build_phase2.py       # launch cadence, payload density, operator concentration
python build_history.py      # 1957 to 2015 series against Rao, Burgess and Kaffine
python build_phase3.py       # collision risk, satellite value, Pigouvian charge
python export_web_data.py    # writes web/data.json for the website
```

Each script validates itself before printing its report and exits with a
non-zero status if a check fails. Across the five scripts, **29 automated
checks** run in total, 6 in `build_history.py`, 9 in `build_phase1.py`, 6 in
`build_phase2.py`, 8 in `build_phase3.py`, and none in `export_web_data.py`,
which only exports. Three of the 29 depend on external reference files not
included in this repository, a replication series from Rao, Burgess and
Kaffine and a set of OPUS reference bins, and are skipped gracefully when
those files are absent, so a fresh clone runs 26 of the 29 out of the box.

The checks are not decorative. Four substantive errors were caught this way,
including one that placed the 1,895 derelict payloads in this altitude range on
the wrong side of the risk equation, and one that dated debris fragments to the
launch of their parent object rather than to the fragmentation event that
produced them, inflating historical debris counts by a factor of 2.6.

`export_web_data.py` writes its output to `web/data.json`. To publish an
update, copy that file, together with the relevant CSVs from `data/processed/`,
over the copies sitting at the root of this repository.

The historical reconstruction is validated against the series published by Rao,
Burgess and Kaffine for 1957 to 2015. Median agreement over 1990 to 2015 is
0.96 for payloads in orbit and 1.26 for debris fragments.

---

## Known limitations

Stated in full on the website, summarised here.

- The estimate is a **lower bound**. The pollution persistence channel of Rao
  and Rondina (2025), covering fragments generated by future debris-on-debris
  collisions, is not priced.
- The calculation covers a **single date**. It cannot reproduce the 2040
  projections in the literature, which model how launch behaviour responds to
  risk.
- Objects are assigned to the shell containing their **mean altitude**. About a
  third of tracked objects are eccentric enough to cross more than one shell.
  No residence time weighting is applied.
- The untracked debris factor is **assumed, not measured**. The charge varies
  linearly with it.
- The cost function is calibrated on a **representative 223 kg satellite**, so
  a three-tonne spacecraft and a cubesat are treated identically.
- The comparison with Rao, Burgess and Kaffine is not exact. Their model covers
  100 to 2000 km against 200 to 1600 km here, and their two published anchors,
  $14,900 for 2020 and $235,000 for 2040, do not lie on a single 14 per cent
  path, which places the 2026 reference value between $32,700 and $37,500.

---

## References

- Rao, A., Burgess, M. G., & Kaffine, D. (2020). Orbital-use fees could more
  than quadruple the value of the space industry. *PNAS*, 117(23), 12756–12762.
  https://doi.org/10.1073/pnas.1921260117
- Rao, A., & Rondina, G. (2025). The Economics of Orbit Use: Open Access,
  External Costs, and Runaway Debris Growth. *JAERE*, 12(2), 353–388.
  https://doi.org/10.1086/730695
- McDowell, J. C. GCAT, General Catalog of Artificial Space Objects.
  https://planet4589.org/space/gcat
- MOCAT 4S, MIT ARCLab. Source-sink orbital debris model.
- OPUS, Orbital Debris Propagators Unified with Economic Systems.

---

## Licence and attribution

Code is MIT. See [LICENSE](LICENSE).

Data derived from GCAT inherits **CC BY 4.0**. If you reuse the CSV files or
`data.json`, credit GCAT (J. C. McDowell, planet4589.org/space/gcat).

AI assistance is disclosed in [AI-USE.md](AI-USE.md).

## Citing this work

Use the DOI at the top of this page, or the "Cite this repository" button in
the GitHub sidebar, which reads [CITATION.cff](CITATION.cff).

---

Independent research project by Alex Dion.
Comments, corrections and questions are welcome through
[LinkedIn](https://www.linkedin.com/in/alex-dion-889812331/) or the
repository issues.
