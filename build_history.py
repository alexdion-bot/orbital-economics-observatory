"""
Reconstruction historique du stock orbital, 1957-2026.

Principe : satcat.tsv donne pour chaque objet jamais catalogue sa date de
lancement, sa date de retombee (si elle a eu lieu) et ses elements orbitaux.
Un objet est donc compte "en orbite" pour chaque annee comprise entre son
lancement et sa retombee, et affecte a la coquille correspondant a son
altitude moyenne enregistree.

LIMITE CENTRALE, A NE JAMAIS OMETTRE
Les elements orbitaux de satcat sont un instantane par objet, pas une
trajectoire. Un objet est donc maintenu dans une coquille fixe toute sa vie,
alors qu'en realite il perd de l'altitude sous l'effet de la trainee. La
reconstruction est fiable en agregat et pour les tendances ; elle est
approximative dans la repartition fine par coquille, surtout sous 500 km.
La section de validation ci-dessous quantifie l'erreur commise en comparant
a la serie publiee de Rao, Burgess & Kaffine (2020) sur 1957-2015.

Produit dans data/processed/ :
  history_stock_by_year.csv        stock annuel par classe, orbite basse
  history_shell_by_year.csv        stock annuel par coquille et par classe
  history_validation_vs_rao.csv    comparaison a la serie publiee

Source : GCAT (J. McDowell, planet4589.org/space/gcat), CC-BY.
Reference de validation : Rao, Burgess & Kaffine (2020), PNAS 117(23), depot
de replication github.com/akhilrao/tragedy-space-commons.
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

RAO_SERIES = Path("/home/claude/refs/tragedy-space-commons/data/stock_series.csv")

FIRST_YEAR, LAST_YEAR = 1957, 2026
# Rao et al. definissent l'orbite basse comme 100-2000 km ; on reprend cette
# bande pour la validation, distincte de la bande 200-1600 km des coquilles.
LEO_VALIDATION_MIN, LEO_VALIDATION_MAX = 100.0, 2000.0

checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))


def fmt(n) -> str:
    return f"{int(n):,}".replace(",", " ")


# ---------------------------------------------------------------------------
# 1. Preparation du catalogue complet
# ---------------------------------------------------------------------------
sc = gcat.read_gcat(RAW / "satcat.tsv")
# DATE D'ENTREE EN ORBITE : SDate, pas LDate.
# GCAT distingue la date de lancement du vehicule parent (LDate) de la date a
# laquelle l'objet a commence a exister comme objet distinct (SDate). Pour une
# charge utile les deux coincident, mais pour un fragment de debris SDate est
# la date de la fragmentation qui l'a cree : 64 % des debris ont un ecart
# non nul entre les deux, median de 2 ans et pouvant atteindre 55 ans.
# Utiliser LDate ferait exister un fragment des le lancement de son parent,
# gonflant massivement les stocks historiques de debris.
sc["separation_year"] = gcat.to_year(sc["SDate"])
sc["launch_year"] = gcat.to_year(sc["LDate"])
sc["origin_year"] = sc["separation_year"].fillna(sc["launch_year"])
sc["decay_year"] = gcat.to_year(sc["DDate"])
sc["perigee_km"] = gcat.to_num(sc["Perigee"])
sc["apogee_km"] = gcat.to_num(sc["Apogee"])
sc["obj_class"] = gcat.object_class(sc["Type"])
sc["alt_mean_km"] = gcat.mean_altitude(sc["perigee_km"], sc["apogee_km"])

usable = sc[sc["origin_year"].notna() & sc["alt_mean_km"].notna() & sc["obj_class"].notna()].copy()
check("catalogue exploitable pour la reconstruction",
      len(usable) / len(sc) > 0.95,
      f"{fmt(len(usable))}/{fmt(len(sc))} objets ({len(usable)/len(sc):.1%})")

# Fin de presence en orbite : annee de retombee si connue, sinon toujours present.
usable["end_year"] = usable["decay_year"].fillna(LAST_YEAR)
usable = usable[usable["end_year"] >= usable["origin_year"]]

# PORTEE DE LA RECONSTRUCTION : on reconstruit la presence en orbite par
# classe d'objet, PAS le statut operationnel. satcat ne date pas la fin
# d'exploitation d'une charge utile, et psatcat ne la renseigne que pour 42 %
# d'entre elles : distinguer historiquement satellite actif et epave n'est donc
# pas possible de facon fiable. La distinction actif/inerte n'est etablie que
# sur l'instantane courant (currentcat), ou GCAT la fournit explicitement.
# C'est cet instantane, et non l'historique, qui alimente le calcul de risque.

# ---------------------------------------------------------------------------
# 2. Expansion en panel objet x annee
# ---------------------------------------------------------------------------
usable["n_years"] = (usable["end_year"] - usable["origin_year"] + 1).astype(int)
rep = usable.loc[usable.index.repeat(usable["n_years"])].copy()
rep["year"] = rep["origin_year"].astype(int) + rep.groupby(level=0).cumcount()
rep = rep[(rep["year"] >= FIRST_YEAR) & (rep["year"] <= LAST_YEAR)]

rep["is_payload"] = rep["obj_class"] == "P"
rep["is_debris"] = rep["obj_class"] == "D"
rep["is_rocket_body"] = rep["obj_class"] == "R"
rep["is_component"] = rep["obj_class"] == "C"

check("panel historique coherent (pas d'annee hors bornes)",
      bool((rep["year"] >= FIRST_YEAR).all() and (rep["year"] <= LAST_YEAR).all()),
      f"{fmt(len(rep))} lignes objet-annee")

# ---------------------------------------------------------------------------
# 3. Stock annuel sur la bande de validation (100-2000 km)
# ---------------------------------------------------------------------------
leo = rep[(rep["alt_mean_km"] >= LEO_VALIDATION_MIN)
          & (rep["alt_mean_km"] <= LEO_VALIDATION_MAX)]

stock = (leo.groupby("year")
         .agg(payloads=("is_payload", "sum"),
              debris=("is_debris", "sum"),
              rocket_bodies=("is_rocket_body", "sum"),
              components=("is_component", "sum"),
              total_objects=("is_payload", "size"))
         .reset_index())
stock["non_payload"] = stock[["debris", "rocket_bodies", "components"]].sum(axis=1)
stock.to_csv(OUT / "history_stock_by_year.csv", index=False)

check("stock reconstruit croissant sur longue periode",
      int(stock.query("year == 2020")["total_objects"].iloc[0])
      > int(stock.query("year == 1990")["total_objects"].iloc[0]),
      "2020 > 1990")

# ---------------------------------------------------------------------------
# 4. Stock annuel par coquille (bande 200-1600 km)
# ---------------------------------------------------------------------------
shells = gcat.shell_table()
band = rep[(rep["alt_mean_km"] >= gcat.SHELL_MIN_KM) & (rep["alt_mean_km"] < gcat.SHELL_MAX_KM)].copy()
band["shell"] = gcat.assign_shell(band["alt_mean_km"])

shell_hist = (band.groupby(["year", "shell"])
              .agg(payloads=("is_payload", "sum"),
                   debris=("is_debris", "sum"),
                   rocket_bodies=("is_rocket_body", "sum"),
                   components=("is_component", "sum"))
              .reset_index())
shell_hist["non_payload"] = shell_hist[["debris", "rocket_bodies", "components"]].sum(axis=1)
shell_hist = shell_hist.merge(shells[["shell", "alt_lower_km", "alt_upper_km", "alt_mid_km"]], on="shell")
shell_hist.to_csv(OUT / "history_shell_by_year.csv", index=False)

_tot = int(shell_hist[["payloads", "debris", "rocket_bodies", "components"]].sum().sum())
check("stock par coquille : totaux coherents avec le panel de bande",
      _tot == len(band), f"{fmt(_tot)} = {fmt(len(band))}")

# ---------------------------------------------------------------------------
# 5. Validation contre la serie publiee de Rao et al. (1957-2015)
# ---------------------------------------------------------------------------
if RAO_SERIES.exists():
    rao = pd.read_csv(RAO_SERIES)[["year", "payloads_in_orbit", "debris"]]
    rao = rao[(rao["year"] >= 1957) & (rao["year"] <= 2015)]
    rao = rao.rename(columns={"payloads_in_orbit": "rao_active_payloads", "debris": "rao_debris"})

    # Rao compte les charges utiles en orbite d'une part, les fragments de
    # debris d'autre part. On compare classe a classe : les etages de lanceur
    # et composants largues ne relevent d'aucune des deux categories publiees.
    comp = stock.merge(rao, on="year", how="inner")
    comp["ratio_payloads"] = comp["payloads"] / comp["rao_active_payloads"].replace(0, np.nan)
    comp["ratio_debris"] = comp["debris"] / comp["rao_debris"].replace(0, np.nan)
    comp.to_csv(OUT / "history_validation_vs_rao.csv", index=False)

    recent = comp[comp["year"] >= 1990]
    med_pay = float(recent["ratio_payloads"].median())
    med_deb = float(recent["ratio_debris"].median())

    print("Validation contre Rao, Burgess & Kaffine (2020), 1990-2015 :")
    print(f"  charges utiles en orbite : ratio median reconstruction/publie = {med_pay:.2f}")
    print(f"  fragments de debris      : ratio median reconstruction/publie = {med_deb:.2f}")
    print()
    print(comp[comp["year"].isin([1990, 2000, 2010, 2015])][
        ["year", "payloads", "rao_active_payloads", "debris", "rao_debris"]
    ].to_string(index=False))
    print()

    check("charges utiles : reconstruction coherente avec Rao",
          0.7 < med_pay < 1.4, f"ratio median {med_pay:.2f}")
    check("fragments de debris : reconstruction coherente avec Rao",
          0.7 < med_deb < 1.7, f"ratio median {med_deb:.2f}")
else:
    print("Serie de reference Rao absente : validation externe sautee.\n")

# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
print("Stock orbital reconstruit (bande 100-2000 km) :")
print(stock[stock["year"].isin([1970, 1990, 2000, 2010, 2015, 2020, 2026])][
    ["year", "payloads", "debris", "rocket_bodies", "components", "total_objects"]].to_string(index=False))
print()
print("CONTROLES")
all_ok = True
for label, ok, detail in checks:
    print(f"  [{'OK   ' if ok else 'ECHEC'}] {label}" + (f" — {detail}" if detail else ""))
    all_ok &= ok
print("\nReconstruction historique :", "valide" if all_ok else "ECHEC — corriger avant de continuer")
sys.exit(0 if all_ok else 1)
