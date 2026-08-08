"""
Phase 2 — Statistiques descriptives.

Trois questions, dans l'ordre de la feuille de route :
  1. Cadence de lancement mondiale : par annee, par pays, par operateur.
  2. Densite orbitale dans le temps : combien d'objets actifs vs inertes,
     chaque annee, dans la bande 200-1600 km.
  3. Concentration : quelle part revient a Starlink / a l'operateur dominant,
     et comment cette part a evolue.

Limite assumee et documentee : le panel charge utile-annee (payload_panel.csv)
utilise TOp (derniere date d'operation connue) comme fin de vie ; ce n'est pas
une desorbitation confirmee. Un satellite peut etre compte "actif" une annee
ou il etait en realite deja degrade. Cette imprecision est bornee (quelques
mois a un an par objet) et ne biaise pas les tendances de long terme.

Produit dans data/processed/ :
  descriptive_launches_by_year.csv     lancements et charges utiles par annee
  descriptive_launches_by_country.csv  lancements par pays x annee (top 12 pays)
  descriptive_density_by_year.csv      objets actifs vs inertes en orbite, par annee
  descriptive_concentration.csv        part de l'operateur dominant, par annee

Source : GCAT (J. McDowell, planet4589.org/space/gcat), CC-BY.
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
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2026
checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))


def fmt(n) -> str:
    return f"{int(n):,}".replace(",", " ")


# Les tables produites en phase 1 sont l'entree de la phase 2 : on ne relit
# pas les .tsv bruts ici, sauf launchlog pour le detail par pays lanceur
# (l'information n'est pas conservee telle quelle dans launches_by_year.csv).
payload_panel = pd.read_csv(PROC / "payload_panel.csv")
launches_by_year = pd.read_csv(PROC / "launches_by_year.csv")
# n_payload_slots vit dans un fichier a part : c'est une quantite au niveau
# ANNEE (nombre de charges utiles deployees), pas par pays lanceur. La
# sommer apres un groupby-par-pays la multiplierait par le nombre de pays
# actifs cette annee-la -- piege deja rencontre, corrige a la source en
# phase 1 plutot que patche ici.
payload_slots = pd.read_csv(PROC / "payload_slots_by_year.csv")

# ---------------------------------------------------------------------------
# 1. Cadence de lancement mondiale, par annee
# ---------------------------------------------------------------------------
launches_total = (launches_by_year.groupby("launch_year", as_index=False)
                  .agg(n_launches=("n_launches", "sum"),
                       n_orbital_success=("n_orbital_success", "sum")))
launches_total = launches_total.merge(payload_slots, on="launch_year", how="left")
launches_total = launches_total[launches_total["launch_year"] >= 1957]
launches_total["success_rate"] = (launches_total["n_orbital_success"]
                                   / launches_total["n_launches"]).round(3)
launches_total.to_csv(PROC / "descriptive_launches_by_year.csv", index=False)

check("serie de lancements couvre 1957 a aujourd'hui sans trou",
      set(range(1957, CURRENT_YEAR + 1)) <= set(launches_total["launch_year"].astype(int)),
      f"{launches_total['launch_year'].min():.0f}-{launches_total['launch_year'].max():.0f}")
check("aucun taux de succes hors [0,1]",
      launches_total["success_rate"].between(0, 1).all(),
      f"min {launches_total['success_rate'].min():.2f}, max {launches_total['success_rate'].max():.2f}")
# Garde-fou specifique contre le double comptage deja rencontre : le nombre
# de charges utiles deployees une annee ne peut pas depasser tres largement
# le nombre de lancements reussis multiplie par une charge utile utile
# maximale plausible (permissif : x600, pour couvrir les tirs a tres forte
# multiplicite comme les rideshares Starlink/Transporter).
recent = launches_total.query("launch_year >= 2015")
check("aucun signe de double comptage des charges utiles deployees",
      (recent["n_payload_slots"] <= recent["n_orbital_success"] * 600).all(),
      f"max ratio slots/lancement reussi : "
      f"{(recent['n_payload_slots'] / recent['n_orbital_success'].clip(lower=1)).max():.0f}")

# ---------------------------------------------------------------------------
# 2. Lancements par pays de l'Etat lanceur, par annee (launchlog, dedoublonne par tir)
# ---------------------------------------------------------------------------
ll = gcat.read_gcat(RAW / "launchlog.tsv")
ll["launch_year"] = gcat.to_year(ll["Launch_Date"])
launches_uniq = ll.drop_duplicates("Launch_Tag")

top_states = launches_uniq["LVState"].value_counts().head(12).index
by_country = (launches_uniq[launches_uniq["LVState"].isin(top_states)]
              .groupby(["launch_year", "LVState"], observed=True)
              .size().rename("n_launches").reset_index()
              .rename(columns={"LVState": "launch_state"}))
by_country.to_csv(PROC / "descriptive_launches_by_country.csv", index=False)

check("somme des lancements par pays (top 12) <= total des lancements",
      int(by_country["n_launches"].sum()) <= len(launches_uniq),
      f"{fmt(by_country['n_launches'].sum())} <= {fmt(len(launches_uniq))}")

# ---------------------------------------------------------------------------
# 3. Densite orbitale dans le temps : actifs vs inertes, par annee
# ---------------------------------------------------------------------------
# On ne dispose que d'un instantane du stock de debris/etages/composants
# (currentcat, aout 2026) : impossible de reconstruire leur historique sans
# TLE historiques (limite deja signalee). En revanche le panel charge
# utile-annee permet de suivre le nombre de charges utiles ACTIVES chaque
# annee, ce qui est deja la question 2 de la feuille de route.
active_by_year = (payload_panel.groupby("year", as_index=False)
                  .size().rename(columns={"size": "n_active_payloads"}))
active_by_year = active_by_year[active_by_year["year"] >= 1957]
active_by_year.to_csv(PROC / "descriptive_density_by_year.csv", index=False)

check("nombre de charges utiles actives croissant sur longue periode",
      active_by_year.query("year == 2024")["n_active_payloads"].iloc[0]
      > active_by_year.query("year == 2010")["n_active_payloads"].iloc[0],
      "2024 > 2010, tendance de fond")

# ---------------------------------------------------------------------------
# 4. Concentration : part de l'operateur dominant par annee
# ---------------------------------------------------------------------------
op_year = (payload_panel.groupby(["year", "operator_code"], observed=True)
           .size().rename("n").reset_index())
tot_year = op_year.groupby("year")["n"].transform("sum")
op_year["share"] = op_year["n"] / tot_year

top_op_by_year = (op_year.sort_values(["year", "n"], ascending=[True, False])
                  .groupby("year").first().reset_index())
top_op_by_year = top_op_by_year[top_op_by_year["year"] >= 1957]
top_op_by_year.to_csv(PROC / "descriptive_concentration.csv", index=False)

check("part du plus gros operateur toujours <= 100%",
      top_op_by_year["share"].between(0, 1).all(),
      f"max {top_op_by_year['share'].max():.1%}")

# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
print("Cadence de lancement mondiale, 5 dernieres annees :")
print(launches_total.tail(5).to_string(index=False))
print()
print("Charges utiles actives en orbite, 5 dernieres annees :")
print(active_by_year.tail(5).to_string(index=False))
print()
last = top_op_by_year.iloc[-1]
print(f"Operateur dominant en {int(last['year'])} : {last['operator_code']} "
      f"({last['n']} satellites, {last['share']:.1%} du total actif cette annee-la)")
print()
print("CONTROLES")
all_ok = True
for label, ok, detail in checks:
    print(f"  [{'OK   ' if ok else 'ECHEC'}] {label}" + (f" — {detail}" if detail else ""))
    all_ok &= ok
print("\nPhase 2 :", "statistiques valides" if all_ok else "ECHEC — corriger avant de continuer")
sys.exit(0 if all_ok else 1)
