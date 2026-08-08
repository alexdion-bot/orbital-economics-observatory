"""
Export des resultats vers un jeu de donnees compact destine au site web.

Produit web/data.json, contenant :
  - les totaux d'entete (cout externe total, nombre de satellites, snapshot)
  - la population par coquille, avec risque et taxe
  - la fiche de chaque operateur (748), avec sa charge fiscale
  - la liste des satellites actifs en orbite basse, encodee de facon compacte

Encodage des satellites : trois tableaux paralleles (altitude arrondie au km,
inclinaison arrondie au dixieme de degre, index de l'operateur) plutot qu'une
liste d'objets, ce qui divise la taille du fichier par environ six.

Le nom de chaque satellite est conserve pour l'affichage au clic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gcat  # noqa: E402

PROC = ROOT / "data" / "processed"
WEB = ROOT / "web"
WEB.mkdir(parents=True, exist_ok=True)

SNAPSHOT = "2 August 2026"

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
shells = pd.read_csv(PROC / "risk_by_shell.csv")
operators = pd.read_csv(PROC / "tax_by_operator.csv")
objects = pd.read_csv(PROC / "objects_in_orbit.csv")
history = pd.read_csv(PROC / "history_stock_by_year.csv")
launches = pd.read_csv(PROC / "descriptive_launches_by_year.csv")
concentration = pd.read_csv(PROC / "descriptive_concentration.csv")
validation = pd.read_csv(PROC / "risk_validation.csv")

sats = objects[(objects["op_class"] == gcat.ACTIVE_PAYLOAD)
               & objects["in_leo_band"]].copy()

# Nom lisible de l'operateur : on prefere le nom complet quand GCAT le fournit
orgs = gcat.read_gcat(ROOT / "data" / "raw" / "orgs.tsv")
org_names = (orgs[["Code", "Name", "ShortEName"]]
             .drop_duplicates("Code").set_index("Code"))
name_map = org_names["Name"].to_dict()
state_map = org_names["ShortEName"].to_dict()

# ---------------------------------------------------------------------------
# Fiches operateurs
# ---------------------------------------------------------------------------
operators = operators.sort_values("total_tax_usd_year", ascending=False).reset_index(drop=True)
op_index = {code: i for i, code in enumerate(operators["operator_code"])}

operators_out = []
for _, r in operators.iterrows():
    code = r["operator_code"]
    operators_out.append({
        "code": code,
        "name": str(name_map.get(code, code))[:48],
        "country": r["state_short"] if isinstance(r["state_short"], str) else r["state"],
        "n": int(r["n_satellites"]),
        "altMean": round(float(r["mean_altitude_km"]), 1),
        "prob": float(r["mean_collision_prob"]),
        "taxPerSat": round(float(r["tax_per_satellite_usd_year"]), 2),
        "taxTotal": round(float(r["total_tax_usd_year"]), 2),
        "losses": round(float(r["expected_losses_usd_year"]), 2),
    })

# ---------------------------------------------------------------------------
# Satellites : tableaux paralleles compacts
# ---------------------------------------------------------------------------
sats = sats[sats["Owner"].isin(op_index)].copy()
sats["opIdx"] = sats["Owner"].map(op_index)
sats["shellIdx"] = sats["shell"].astype(int) - 1
sats = sats.sort_values("alt_mean_km").reset_index(drop=True)

shell_tax = shells.set_index("shell")["pigouvian_tax_usd_year"].to_dict()
shell_prob = shells.set_index("shell")["collision_prob_annual"].to_dict()

# Compression des noms : un petit nombre de prefixes couvre l'essentiel du
# catalogue (Starlink, OneWeb, Kuiper...). On les remplace par un jeton court
# et on reconstitue cote client. Le reste du nom est conserve tel quel.
raw_names = [str(n)[:40] for n in sats["Name"]]
prefix_counts: dict[str, int] = {}
for n in raw_names:
    for cut in range(4, min(len(n), 16) + 1):
        prefix_counts[n[:cut]] = prefix_counts.get(n[:cut], 0) + 1
# on retient les prefixes qui font gagner le plus de caracteres
scored = sorted(prefix_counts.items(), key=lambda kv: -(len(kv[0]) - 2) * kv[1])
chosen: list[str] = []
for pref, cnt in scored:
    if cnt < 20 or len(pref) < 5:
        continue
    if any(pref.startswith(c) or c.startswith(pref) for c in chosen):
        continue
    chosen.append(pref)
    if len(chosen) >= 24:
        break
chosen.sort(key=len, reverse=True)

def encode_name(n: str) -> str:
    for i, pref in enumerate(chosen):
        if n.startswith(pref):
            return f"~{i:x}{n[len(pref):]}"
    return n

satellites_out = {
    "alt": [int(round(v)) for v in sats["alt_mean_km"]],
    "inc": [int(round(float(v))) for v in sats["inc_deg"]],
    "op": [int(v) for v in sats["opIdx"]],
    "yr": [int(v) - 1957 if pd.notna(v) and v > 0 else -1 for v in sats["launch_year"]],
    "nm": "|".join(encode_name(n) for n in raw_names),
    "prefixes": chosen,
}

# ---------------------------------------------------------------------------
# Coquilles
# ---------------------------------------------------------------------------
shells_out = [{
    "k": int(r["shell"]),
    "lo": int(r["alt_lower_km"]),
    "hi": int(r["alt_upper_km"]),
    "S": int(r["S_maneuverable"]),
    "D": int(r["D_non_maneuverable"]),
    "prob": float(r["collision_prob_annual"]),
    "tax": round(float(r["pigouvian_tax_usd_year"]), 2),
    "taxCongestion": round(float(r["tax_congestion_usd_year"]), 2),
    "taxPollution": round(float(r["tax_pollution_usd_year"]), 2),
    "cost": round(float(r["satellite_cost_usd"]), 0),
    "density": float(r["density_per_km3"]),
} for _, r in shells.iterrows()]

# ---------------------------------------------------------------------------
# Series historiques (allegees : une valeur par an)
# ---------------------------------------------------------------------------
hist = history[history["year"] >= 1960]
history_out = {
    "year": [int(v) for v in hist["year"]],
    "payloads": [int(v) for v in hist["payloads"]],
    "debris": [int(v) for v in hist["debris"]],
    "rocketBodies": [int(v) for v in hist["rocket_bodies"]],
    "components": [int(v) for v in hist["components"]],
}

lau = launches[launches["launch_year"] >= 1960]
launches_out = {
    "year": [int(v) for v in lau["launch_year"]],
    "launches": [int(v) for v in lau["n_launches"]],
    "payloads": [int(v) for v in lau["n_payload_slots"]],
}

conc = concentration[concentration["year"] >= 1960]
concentration_out = {
    "year": [int(v) for v in conc["year"]],
    "op": [str(v) for v in conc["operator_code"]],
    "share": [round(float(v), 4) for v in conc["share"]],
}

# ---------------------------------------------------------------------------
# Entete
# ---------------------------------------------------------------------------
total_tax = float(shells["total_external_cost_usd_year"].sum())
top = operators_out[0]
kin_tax = float(validation.loc[1, "cinetique_par_coquille"])
rao_tax = float(validation.loc[1, "agrege_calibre_rao"])
rao_ref = float(validation.loc[1, "repere_publie"])
kin_prob = float(validation.loc[0, "cinetique_par_coquille"])
rao_prob = float(validation.loc[0, "agrege_calibre_rao"])

payload = {
    "snapshot": SNAPSHOT,
    "totals": {
        "externalCostUsdYear": round(total_tax, 0),
        "activeSatellites": int(shells["S_maneuverable"].sum()),
        "inertObjects": int(shells["D_non_maneuverable"].sum()),
        "operators": len(operators_out),
        "topOperator": top["name"],
        "topOperatorCode": top["code"],
        "topOperatorShare": round(top["taxTotal"] / total_tax, 4),
        "meanTaxPerSat": round(kin_tax, 0),
        "maxShellTax": round(float(shells["pigouvian_tax_usd_year"].max()), 0),
        "maxShellAlt": int(shells.loc[shells["pigouvian_tax_usd_year"].idxmax(), "alt_lower_km"]),
    },
    "validation": {
        "kineticTax": round(kin_tax, 0),
        "raoTax": round(rao_tax, 0),
        "raoPublishedExtrapolated": round(rao_ref, 0),
        "kineticProb": kin_prob,
        "raoProb": rao_prob,
        "congestionShare": round(
            float(np.average(shells["tax_congestion_usd_year"],
                             weights=np.maximum(shells["S_maneuverable"], 1e-9))) / kin_tax, 6),
    },
    "shells": shells_out,
    "operators": operators_out,
    "satellites": satellites_out,
    "history": history_out,
    "launches": launches_out,
    "concentration": concentration_out,
}

out_path = WEB / "data.json"
out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

size_kb = out_path.stat().st_size / 1024
print(f"ecrit : {out_path}  ({size_kb:.0f} Ko)")
print(f"  satellites   {len(satellites_out['alt'])}")
print(f"  operateurs   {len(operators_out)}")
print(f"  coquilles    {len(shells_out)}")
print(f"  cout externe total  ${total_tax:,.0f}")
