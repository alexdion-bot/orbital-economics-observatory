"""
Phase 3 — Risque de collision et taxe pigouvienne implicite.

Calcule, coquille par coquille puis operateur par operateur, sur l'etat de
l'orbite au 2 aout 2026 :
  1. la probabilite annuelle de collision subie par un satellite actif ;
  2. le risque marginal qu'un satellite supplementaire impose aux autres ;
  3. la taxe pigouvienne qui internaliserait ce cout externe, en dollars par
     satellite-annee.

Deux approches sont menees en parallele et confrontees :
  (A) cinetique par coquille, parametree sur MOCAT-4S ;
  (B) agregee, coefficients calibres par Rao, Burgess & Kaffine (2020).
La (B) sert de controle externe : si les deux convergent en ordre de grandeur
sur l'agregat, la (A) est credible dans sa ventilation par coquille, que la
(B) est structurellement incapable de produire.

Produit dans data/processed/ :
  risk_by_shell.csv        risque, cout et taxe par coquille
  tax_by_operator.csv      exposition et charge fiscale implicite par operateur
  risk_validation.csv      confrontation des deux approches et aux reperes publies

Sources : GCAT (J. McDowell, planet4589.org/space/gcat), CC-BY, pour l'etat de
l'orbite ; MOCAT-4S / OPUS (Rao et al. 2023) pour les parametres physiques et
la fonction de cout ; Rao, Burgess & Kaffine (2020) pour les coefficients
calibres et les reperes de redevance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gcat      # noqa: E402
import physics as ph  # noqa: E402

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

SNAPSHOT = "2026-08-02"
checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))


def fmt(n) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def usd(x) -> str:
    return "$" + fmt(x)


# ---------------------------------------------------------------------------
# 1. Etat de l'orbite par coquille
# ---------------------------------------------------------------------------
shell = pd.read_csv(PROC / "shell_population.csv")
shell["volume_m3"] = ph.shell_volume_m3(shell["alt_lower_km"], shell["alt_upper_km"])

S = shell["S_maneuverable"].to_numpy(dtype=float)
D = shell["D_non_maneuverable"].to_numpy(dtype=float)
V = shell["volume_m3"].to_numpy(dtype=float)

check("volumes de coquille strictement croissants avec l'altitude",
      bool(np.all(np.diff(V) > 0)),
      f"{V[0]:.3e} a {V[-1]:.3e} m3")

# ---------------------------------------------------------------------------
# 2. (A) Risque cinetique par coquille
# ---------------------------------------------------------------------------
shell["collision_rate_per_year"] = ph.collision_rate_per_satellite(D, S, V)
shell["collision_prob_annual"] = ph.rate_to_probability(shell["collision_rate_per_year"])

# Densite spatiale, utile a l'interpretation
shell["density_per_km3"] = (S + D) / (V / 1e9)

check("probabilites de collision dans [0,1]",
      bool(shell["collision_prob_annual"].between(0, 1).all()),
      f"max {shell['collision_prob_annual'].max():.4%}")

# ---------------------------------------------------------------------------
# 3. Valeur d'un satellite, par coquille
# ---------------------------------------------------------------------------
shell["satellite_cost_usd"] = ph.satellite_cost_usd(shell["alt_mid_km"].to_numpy())

check("cout d'un satellite dans une plage plausible sur la bande peuplee",
      bool(shell.loc[shell["alt_mid_km"] >= 400, "satellite_cost_usd"].between(1e6, 3e6).all()),
      f"{usd(shell.loc[shell['alt_mid_km'] >= 400, 'satellite_cost_usd'].min())} a "
      f"{usd(shell.loc[shell['alt_mid_km'] >= 400, 'satellite_cost_usd'].max())}")

# ---------------------------------------------------------------------------
# 4. Taxe pigouvienne par coquille
# ---------------------------------------------------------------------------
cost = shell["satellite_cost_usd"].to_numpy()
shell["marginal_risk_imposed"] = ph.marginal_risk_imposed(S, V)
shell["tax_congestion_usd_year"] = ph.pigouvian_tax_usd(S, V, cost)
shell["tax_pollution_usd_year"] = ph.pigouvian_tax_pollution_usd(S, V, cost)
shell["pigouvian_tax_usd_year"] = shell["tax_congestion_usd_year"] + shell["tax_pollution_usd_year"]

# Le cout externe total impose par le parc en place, coquille par coquille
shell["total_external_cost_usd_year"] = shell["pigouvian_tax_usd_year"] * S

cols = ["shell", "alt_lower_km", "alt_upper_km", "alt_mid_km", "S_maneuverable",
        "D_non_maneuverable", "volume_m3", "density_per_km3",
        "collision_rate_per_year", "collision_prob_annual", "satellite_cost_usd",
        "marginal_risk_imposed", "tax_congestion_usd_year", "tax_pollution_usd_year",
        "pigouvian_tax_usd_year", "total_external_cost_usd_year"]
shell[cols].to_csv(PROC / "risk_by_shell.csv", index=False)

check("taxe pigouvienne positive partout ou il y a des satellites",
      bool((shell.loc[shell["S_maneuverable"] > 0, "pigouvian_tax_usd_year"] > 0).all()),
      f"{int((shell['S_maneuverable'] > 0).sum())} coquilles peuplees")

# ---------------------------------------------------------------------------
# 5. (B) Controle externe : approche agregee calibree Rao
# ---------------------------------------------------------------------------
S_tot, D_tot = float(S.sum()), float(D.sum())
rao_prob = float(ph.rao_collision_probability(S_tot, D_tot))
rao_marg = float(ph.rao_marginal_risk(S_tot, D_tot))
# valeur moyenne d'un satellite, ponderee par la localisation reelle du parc
mean_value = float(np.average(shell["satellite_cost_usd"], weights=np.maximum(S, 1e-9)))
rao_tax = S_tot * rao_marg * mean_value

# Approche cinetique, ramenee a un equivalent agrege comparable :
# probabilite moyenne ponderee par le nombre de satellites exposes
kin_prob = float(np.average(shell["collision_prob_annual"], weights=np.maximum(S, 1e-9)))
kin_tax = float(np.average(shell["pigouvian_tax_usd_year"], weights=np.maximum(S, 1e-9)))

# Repere publie : redevance optimale de Rao et al., extrapolee a la date du snapshot
years_since_2020 = 2026 - 2020
rao_ouf_2026 = ph.RAO_OUF_2020_USD * (1 + ph.RAO_OUF_GROWTH) ** years_since_2020

validation = pd.DataFrame([
    {"grandeur": "probabilite annuelle de collision par satellite",
     "cinetique_par_coquille": kin_prob, "agrege_calibre_rao": rao_prob,
     "repere_publie": np.nan},
    {"grandeur": "taxe pigouvienne ($/satellite-annee)",
     "cinetique_par_coquille": kin_tax, "agrege_calibre_rao": rao_tax,
     "repere_publie": rao_ouf_2026},
])
validation.to_csv(PROC / "risk_validation.csv", index=False)

# La probabilite cinetique est confrontee a un repere PHYSIQUE externe, non au
# modele agrege de Rao : celui-ci a ete calibre sur un parc de ~2 000 satellites
# et l'extrapoler a 15 500 le fait exploser (voir rapport). L'ESA situe le
# risque annuel de collision catastrophique d'un satellite en orbite basse
# autour de 1e-5 a 1e-3 selon l'altitude et l'epoque.
check("probabilite cinetique dans la plage physique publiee (1e-5 a 1e-3/an)",
      1e-5 < kin_prob < 1e-3,
      f"{kin_prob:.5%} par satellite-annee")

check("taxe cinetique et taxe agregee Rao concordent a un facteur 5 pres",
      0.2 < kin_tax / rao_tax < 5.0,
      f"cinetique {usd(kin_tax)} vs agrege Rao {usd(rao_tax)}, ratio {kin_tax/rao_tax:.2f}")

check("taxe agregee coherente avec la trajectoire publiee de Rao et al.",
      0.2 < rao_tax / rao_ouf_2026 < 5.0,
      f"{usd(rao_tax)} vs repere extrapole {usd(rao_ouf_2026)}, ratio {rao_tax/rao_ouf_2026:.2f}")

# ---------------------------------------------------------------------------
# 6. Ventilation par operateur
# ---------------------------------------------------------------------------
obj = pd.read_csv(PROC / "objects_in_orbit.csv")
active_leo = obj[(obj["op_class"] == gcat.ACTIVE_PAYLOAD) & obj["in_leo_band"]].copy()

tax_lookup = shell.set_index("shell")["pigouvian_tax_usd_year"]
prob_lookup = shell.set_index("shell")["collision_prob_annual"]
active_leo["tax_usd_year"] = active_leo["shell"].map(tax_lookup)
active_leo["collision_prob"] = active_leo["shell"].map(prob_lookup)

orgs = (gcat.read_gcat(RAW / "orgs.tsv")[["Code", "ShortEName", "EName"]]
        .rename(columns={"Code": "State", "ShortEName": "state_short", "EName": "state_name"})
        .drop_duplicates("State"))

by_op = (active_leo.groupby("Owner")
         .agg(n_satellites=("JCAT", "size"),
              mean_altitude_km=("alt_mean_km", "mean"),
              mean_collision_prob=("collision_prob", "mean"),
              total_tax_usd_year=("tax_usd_year", "sum"),
              state=("State", "first"))
         .reset_index()
         .rename(columns={"Owner": "operator_code"}))
by_op["tax_per_satellite_usd_year"] = by_op["total_tax_usd_year"] / by_op["n_satellites"]
by_op["expected_losses_usd_year"] = by_op["mean_collision_prob"] * by_op["n_satellites"] * mean_value
by_op = by_op.merge(orgs.rename(columns={"State": "state"}), on="state", how="left")
by_op = by_op.sort_values("total_tax_usd_year", ascending=False).reset_index(drop=True)
by_op.to_csv(PROC / "tax_by_operator.csv", index=False)

check("charge fiscale totale coherente entre ventilation et agregation",
      abs(by_op["total_tax_usd_year"].sum() - shell["total_external_cost_usd_year"].sum())
      / shell["total_external_cost_usd_year"].sum() < 0.01,
      f"{usd(by_op['total_tax_usd_year'].sum())} vs {usd(shell['total_external_cost_usd_year'].sum())}")

# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
print(f"ETAT DE L'ORBITE AU {SNAPSHOT}")
print(f"  satellites actifs (S)     {fmt(S_tot)}")
print(f"  objets inertes (D)        {fmt(D_tot)}")
print(f"  valeur moyenne satellite  {usd(mean_value)}")
print()

print("RISQUE ET TAXE — coquilles les plus taxees")
top = shell.nlargest(8, "pigouvian_tax_usd_year")
for _, r in top.iterrows():
    print(f"  {r['alt_lower_km']:6.0f}-{r['alt_upper_km']:<6.0f} km  "
          f"S={fmt(r['S_maneuverable']):>6s}  D={fmt(r['D_non_maneuverable']):>6s}  "
          f"P(collision)={r['collision_prob_annual']:7.4%}  "
          f"taxe={usd(r['pigouvian_tax_usd_year']):>12s}/sat/an")
print()

print("DECOMPOSITION DE LA TAXE (moyenne ponderee par le parc)")
w = np.maximum(S, 1e-9)
print(f"  canal congestion  {usd(np.average(shell['tax_congestion_usd_year'], weights=w)):>12s}/sat/an")
print(f"  canal pollution   {usd(np.average(shell['tax_pollution_usd_year'], weights=w)):>12s}/sat/an")
print(f"  total             {usd(kin_tax):>12s}/sat/an")
print()
print("CONFRONTATION DES DEUX APPROCHES")
print(f"  probabilite annuelle    cinetique {kin_prob:.5%}   agrege Rao {rao_prob:.4%}")
print(f"  taxe $/sat/an           cinetique {usd(kin_tax):>10s}   agrege Rao {usd(rao_tax):>10s}   "
      f"ratio {kin_tax/rao_tax:.2f}")
print(f"  repere Rao et al. 2020 extrapole a 2026 (+14%/an) : {usd(rao_ouf_2026)}")
print()

print("CHARGE FISCALE IMPLICITE — 8 premiers operateurs")
for _, r in by_op.head(8).iterrows():
    name = r["state_short"] if isinstance(r["state_short"], str) else r["state"]
    print(f"  {r['operator_code']:<10s} {str(name)[:14]:<14s} "
          f"{fmt(r['n_satellites']):>6s} sat  "
          f"alt.moy {r['mean_altitude_km']:6.0f} km  "
          f"{usd(r['total_tax_usd_year']):>14s}/an")
print()
print(f"  cout externe total du parc en orbite basse : "
      f"{usd(shell['total_external_cost_usd_year'].sum())} par an")
print()

print("CONTROLES")
all_ok = True
for label, ok, detail in checks:
    print(f"  [{'OK   ' if ok else 'ECHEC'}] {label}" + (f" — {detail}" if detail else ""))
    all_ok &= ok
print("\nPhase 3 :", "modele valide" if all_ok else "ECHEC — corriger avant d'interpreter")
sys.exit(0 if all_ok else 1)
