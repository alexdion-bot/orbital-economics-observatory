"""
Phase 1 — Pipeline de donnees GCAT.

Produit dans data/processed/ :
  shells.csv            table de reference des coquilles orbitales (40 x 35 km, 200-1600 km)
  objects_in_orbit.csv  population actuellement en orbite, objet par objet, avec classe et coquille
  shell_population.csv  comptages par coquille x classe -> entree du modele de risque (phase 3)
  payload_panel.csv     panel charge utile x annee (operateur, etat, categorie)
  launches_by_year.csv  cadence de lancement annuelle par etat lanceur

Chaque etape est verifiee ; les controles sont imprimes en fin d'execution et le
script sort en code 1 si l'un d'eux echoue.

Source : GCAT (J. McDowell, planet4589.org/space/gcat), licence CC-BY.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gcat  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2026
CLASSES = [gcat.ACTIVE_PAYLOAD, gcat.DERELICT_PAYLOAD,
           gcat.ROCKET_BODY, gcat.DEBRIS, gcat.COMPONENT]

checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))


def fmt(n) -> str:
    return f"{int(n):,}".replace(",", " ")


# ---------------------------------------------------------------------------
# 0. Provenance
# ---------------------------------------------------------------------------
FILES = ["currentcat.tsv", "satcat.tsv", "psatcat.tsv", "launchlog.tsv", "orgs.tsv", "active.tsv"]
print("Dates de mise a jour GCAT declarees :")
for f in FILES:
    print(f"  {f:16s} {gcat.gcat_updated_at(RAW / f)}")
print()

# ---------------------------------------------------------------------------
# 1. Coquilles orbitales
# ---------------------------------------------------------------------------
shells = gcat.shell_table()
shells.to_csv(OUT / "shells.csv", index=False)
check("40 coquilles de 35 km construites",
      len(shells) == 40 and abs(shells.alt_upper_km.iloc[-1] - 1600.0) < 1e-9,
      f"{len(shells)} coquilles, {shells.alt_lower_km.iloc[0]:.0f}-{shells.alt_upper_km.iloc[-1]:.0f} km")

# ---------------------------------------------------------------------------
# 2. Population actuellement en orbite
# ---------------------------------------------------------------------------
cur = gcat.read_gcat(RAW / "currentcat.tsv")
cur["op_class"] = gcat.operational_class(cur["Active"], cur["Type"])
cur["perigee_km"] = gcat.to_num(cur["Perigee"])
cur["apogee_km"] = gcat.to_num(cur["Apogee"])
cur["inc_deg"] = gcat.to_num(cur["Inc"])
cur["launch_year"] = gcat.to_year(cur["LDate"])
cur["in_earth_orbit"] = cur["ExpandedStatus"].str.lower().str.startswith("in earth orbit")

orbit = cur[cur["in_earth_orbit"]].copy()
orbit["alt_mean_km"] = gcat.mean_altitude(orbit["perigee_km"], orbit["apogee_km"])
orbit["shell"] = gcat.assign_shell(orbit["alt_mean_km"])
orbit["crosses_shells"] = gcat.eccentricity_flag(orbit["perigee_km"], orbit["apogee_km"])
orbit["in_leo_band"] = orbit["shell"].notna()
orbit["maneuverable"] = orbit["op_class"].isin(gcat.MANEUVERABLE)

objects_in_orbit = orbit[[
    "JCAT", "Satcat", "Name", "op_class", "Type", "Owner", "State", "launch_year",
    "perigee_km", "apogee_km", "alt_mean_km", "inc_deg", "shell",
    "crosses_shells", "in_leo_band", "maneuverable", "OpOrbit", "ODate",
]].reset_index(drop=True)
objects_in_orbit.to_csv(OUT / "objects_in_orbit.csv", index=False)

n_classed = int(objects_in_orbit["op_class"].notna().sum())
check("classe operationnelle resolue pour tous les objets en orbite",
      n_classed == len(objects_in_orbit), f"{n_classed}/{len(objects_in_orbit)}")

leo = objects_in_orbit[objects_in_orbit["in_leo_band"]]
check("elements orbitaux presents sur toute la bande 200-1600 km",
      bool(leo[["perigee_km", "apogee_km", "inc_deg"]].notna().all().all()),
      f"{fmt(len(leo))} objets")

# ---------------------------------------------------------------------------
# 3. Population par coquille : l'entree directe du modele de risque
# ---------------------------------------------------------------------------
shell_pop = (leo.groupby(["shell", "op_class"], observed=True)
                .size().unstack(fill_value=0)
                .reindex(range(1, 41), fill_value=0)
                .rename_axis("shell").reset_index())
for c in CLASSES:
    if c not in shell_pop.columns:
        shell_pop[c] = 0
shell_pop["S_maneuverable"] = shell_pop[gcat.MANEUVERABLE].sum(axis=1)
shell_pop["D_non_maneuverable"] = shell_pop[gcat.NON_MANEUVERABLE].sum(axis=1)
shell_pop["total"] = shell_pop["S_maneuverable"] + shell_pop["D_non_maneuverable"]
shell_pop = shells[["shell", "alt_lower_km", "alt_upper_km", "alt_mid_km"]].merge(shell_pop, on="shell")
shell_pop.to_csv(OUT / "shell_population.csv", index=False)

check("aucun objet perdu dans l'agregation par coquille",
      int(shell_pop["total"].sum()) == len(leo),
      f"{fmt(shell_pop['total'].sum())} = {fmt(len(leo))}")

# ---------------------------------------------------------------------------
# 4. Panel charge utile x annee
# ---------------------------------------------------------------------------
psat = gcat.read_gcat(RAW / "psatcat.tsv")
psat["launch_year"] = gcat.to_year(psat["LDate"])
psat["op_end_year"] = gcat.to_year(psat["TOp"])

# Une charge utile sans fin d'exploitation renseignee n'est prolongee jusqu'a
# aujourd'hui que si GCAT la declare encore active ; sinon on ne lui invente pas
# une duree de vie.
active_jcats = set(cur.loc[cur["op_class"] == gcat.ACTIVE_PAYLOAD, "JCAT"])
psat["still_active"] = psat["JCAT"].isin(active_jcats)
psat["end_year"] = psat["op_end_year"]
psat.loc[psat["still_active"], "end_year"] = (
    psat.loc[psat["still_active"], "end_year"].fillna(CURRENT_YEAR).clip(lower=CURRENT_YEAR))

src = psat[psat["launch_year"].notna() & psat["end_year"].notna()].copy()
src["end_year"] = src[["launch_year", "end_year"]].max(axis=1)
src["n_years"] = (src["end_year"] - src["launch_year"] + 1).astype(int)

rep = src.loc[src.index.repeat(src["n_years"])].copy()
rep["year"] = rep["launch_year"].astype(int) + rep.groupby(level=0).cumcount()

# L'operateur du satellite (SatOwner) et son pays (SatState) ne sont PAS dans
# psatcat : le champ 'Control' de psatcat designe le centre de controle au sol
# (ex. TsUP, JSC), pas l'organisation proprietaire. La bonne source est
# launchlog, ou SatOwner/SatState sont renseignes pour la quasi-totalite des
# lancements et se joignent un-a-un sur JCAT (verifie : aucun doublon).
owner_lookup = (gcat.read_gcat(RAW / "launchlog.tsv")[["JCAT", "SatOwner", "SatState"]]
                .rename(columns={"SatOwner": "operator_code", "SatState": "operator_state"})
                .drop_duplicates("JCAT"))

orgs = (gcat.read_gcat(RAW / "orgs.tsv")[["Code", "ShortEName", "EName"]]
        .rename(columns={"Code": "operator_state", "ShortEName": "operator_state_short",
                         "EName": "operator_state_name"})
        .drop_duplicates("operator_state"))

panel = (rep[["JCAT", "Name", "year", "launch_year", "end_year", "still_active",
              "UNState", "Class", "Category", "Program"]]
         .rename(columns={"Name": "payload_name", "UNState": "un_registration_state",
                          "Class": "payload_class", "Category": "category", "Program": "program"})
         .merge(owner_lookup, on="JCAT", how="left")
         .merge(orgs, on="operator_state", how="left"))
panel.to_csv(OUT / "payload_panel.csv", index=False)

check("panel : aucune annee anterieure au lancement",
      bool((panel["year"] >= panel["launch_year"]).all()),
      f"{fmt(len(panel))} lignes charge utile-annee, "
      f"{fmt(panel['JCAT'].nunique())} charges utiles")

n_no_state = int(panel.drop_duplicates("JCAT")["operator_state"].isna().sum())
n_payloads = panel["JCAT"].nunique()
check("pays de l'operateur renseigne pour la quasi-totalite des charges utiles",
      n_no_state / n_payloads < 0.02,
      f"{fmt(n_no_state)}/{fmt(n_payloads)} charges utiles sans pays operateur")

# ---------------------------------------------------------------------------
# 5. Cadence de lancement
# ---------------------------------------------------------------------------
ll = gcat.read_gcat(RAW / "launchlog.tsv")
ll["launch_year"] = gcat.to_year(ll["Launch_Date"])
ll["orbital_success"] = ll["Launch_Code"].str.strip().str.upper().str.startswith("O")

# launchlog compte une ligne par charge utile : dedoublonner pour compter les tirs.
launches = ll.drop_duplicates("Launch_Tag")
by_year = (launches.groupby(["launch_year", "LVState"], observed=True)
           .agg(n_launches=("Launch_Tag", "size"), n_orbital_success=("orbital_success", "sum"))
           .reset_index().rename(columns={"LVState": "launch_state"}))
by_year.to_csv(OUT / "launches_by_year.csv", index=False)

# n_payload_slots (charges utiles deployees) est une quantite au niveau ANNEE,
# pas par pays lanceur : on la publie a part pour eviter tout risque de
# double comptage si quelqu'un somme n_payload_slots en regroupant par annee
# apres avoir agrege par pays (c'est l'erreur qui s'est produite en phase 2
# la premiere fois -- gardee en memoire ici).
payload_slots_by_year = (ll.groupby("launch_year", observed=True).size()
                          .rename("n_payload_slots").reset_index())
payload_slots_by_year.to_csv(OUT / "payload_slots_by_year.csv", index=False)

check("launchlog dedoublonne par tir",
      launches.shape[0] < ll.shape[0],
      f"{fmt(launches.shape[0])} tirs pour {fmt(ll.shape[0])} lignes charge utile")

# ---------------------------------------------------------------------------
# 6. Controles croises externes
# ---------------------------------------------------------------------------
act = gcat.read_gcat(RAW / "active.tsv")
n_act_cur = int((orbit["op_class"] == gcat.ACTIVE_PAYLOAD).sum())
check("charges utiles actives coherentes entre currentcat et active.tsv",
      abs(n_act_cur - len(act)) / len(act) < 0.02,
      f"currentcat {fmt(n_act_cur)} vs active.tsv {fmt(len(act))}")

opus = Path("/home/claude/refs/OPUS/x0_TLE")
if opus.exists():
    tot = lambda f: int(pd.read_csv(opus / f)["Count"].sum())  # noqa: E731
    o_pay = tot("Counts_PAYLOADslot_bins_35.csv") + tot("Counts_PAYLOADunslot_bins_35.csv")
    o_non = tot("Counts_DEBRIS_bins_35.csv") + tot("Counts_DERELICT_bins_35.csv") + tot("Counts_ROCKET BODY_bins_35.csv")
    n_pay = int(shell_pop[gcat.ACTIVE_PAYLOAD].sum())
    n_non = int(shell_pop["D_non_maneuverable"].sum())
    print("Reference OPUS (Space-Track, oct. 2022) vs pipeline (GCAT, aout 2026) :")
    print(f"  charges utiles actives   OPUS {o_pay:6d}   ici {n_pay:6d}   x{n_pay/o_pay:.2f}")
    print(f"  objets non manoeuvrants  OPUS {o_non:6d}   ici {n_non:6d}   x{n_non/o_non:.2f}\n")
    check("ordre de grandeur des non-manoeuvrants coherent avec OPUS 2022",
          0.5 < n_non / o_non < 2.0, f"ratio {n_non / o_non:.2f} sur ~4 ans d'ecart")

# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
print(f"Objets en orbite terrestre : {fmt(len(objects_in_orbit))}")
vc = objects_in_orbit["op_class"].value_counts()
for c in CLASSES:
    print(f"  {c:18s} {fmt(vc.get(c, 0)):>8s}")
print(f"\nBande 200-1600 km : {fmt(len(leo))} objets "
      f"({leo['crosses_shells'].mean():.1%} traversent plusieurs coquilles)")
print(f"  S manoeuvrant     {fmt(shell_pop['S_maneuverable'].sum()):>8s}")
print(f"  D non manoeuvrant {fmt(shell_pop['D_non_maneuverable'].sum()):>8s}")

print("\nCONTROLES")
all_ok = True
for label, ok, detail in checks:
    print(f"  [{'OK   ' if ok else 'ECHEC'}] {label}" + (f" — {detail}" if detail else ""))
    all_ok &= ok
print("\nPhase 1 :", "pipeline valide" if all_ok else "ECHEC — corriger avant de continuer")
sys.exit(0 if all_ok else 1)
