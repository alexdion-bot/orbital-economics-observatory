"""
Lecture et normalisation des fichiers GCAT (General Catalog of Artificial Space Objects).

Source : GCAT (J. McDowell, planet4589.org/space/gcat), licence CC-BY.
Toute publication dérivée doit citer : "data from GCAT (J. McDowell, planet4589.org/space/gcat)".

Particularites du format GCAT gerees ici :
  - ligne 1 = en-tete commencant par '#', ligne 2 = commentaire '# Updated ...', donnees a partir de la ligne 3
  - champs rembourres par des espaces (format fixe converti en TSV) -> strip systematique
  - '-' signifie "non renseigne", pas zero
  - suffixe '?' sur les valeurs incertaines (dates, parfois numeriques)
  - dates a precision variable : "1957 Oct  4 1928:34", "1964 Mar", "1962?", "-"
"""

from __future__ import annotations

import csv
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes physiques et conventions de discretisation
# ---------------------------------------------------------------------------

R_EARTH_KM = 6378.1366  # rayon equatorial terrestre (WGS-84), convention GCAT/MOCAT

# Coquilles orbitales : 40 bandes de 35 km entre 200 et 1600 km d'altitude.
# Cette discretisation reprend exactement celle de MOCAT-4S / OPUS
# (Lifson et al. 2022 ; Rao et al. 2023), ce qui rend nos comptages
# directement comparables aux populations initiales publiees dans OPUS.
SHELL_MIN_KM = 200.0
SHELL_MAX_KM = 1600.0
SHELL_WIDTH_KM = 35.0
N_SHELLS = int(round((SHELL_MAX_KM - SHELL_MIN_KM) / SHELL_WIDTH_KM))  # 40

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Valeurs qui signifient "absent" dans GCAT
_NA_TOKENS = {"", "-", "*", "?", "n/a", "na", "nan", "none"}


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def read_gcat(path: str, strip_strings: bool = True) -> pd.DataFrame:
    """Lit un TSV GCAT et renvoie un DataFrame de chaines nettoyees.

    Toutes les colonnes sont lues en str : la conversion de type est explicite
    et se fait ensuite via to_num / to_year, pour eviter que pandas ne devine
    mal sur des colonnes ou '-' cohabite avec des nombres.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        skiprows=[1],          # ligne '# Updated ...'
        keep_default_na=False,  # on gere les manquants nous-memes
        na_values=[],
        engine="python",
        quoting=csv.QUOTE_NONE,   # GCAT n'echappe pas les guillemets : les traiter
                                  # comme des delimiteurs ferait perdre des lignes
        on_bad_lines="error",     # echouer bruyamment plutot que perdre des donnees
    )
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    if strip_strings:
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def gcat_updated_at(path: str) -> str:
    """Renvoie la date de mise a jour declaree en ligne 2 du fichier."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        line = fh.readline()
    return line.lstrip("#").replace("Updated", "").strip()


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def is_missing(s: pd.Series) -> pd.Series:
    """True la ou la valeur GCAT signifie 'non renseigne'."""
    return s.astype(str).str.strip().str.lower().isin(_NA_TOKENS)


def to_num(s: pd.Series) -> pd.Series:
    """Convertit une colonne GCAT en float, en tolerant les suffixes '?' et '+'."""
    cleaned = (
        s.astype(str)
         .str.strip()
         .str.replace(r"[?+*]", "", regex=True)
         .str.strip()
    )
    cleaned = cleaned.mask(cleaned.str.lower().isin(_NA_TOKENS))
    return pd.to_numeric(cleaned, errors="coerce")


_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")


def to_year(s: pd.Series) -> pd.Series:
    """Extrait l'annee d'une date GCAT a precision variable. Renvoie un Int64 nullable."""
    txt = s.astype(str).str.strip()
    years = txt.str.extract(_YEAR_RE, expand=False)
    return pd.to_numeric(years, errors="coerce").astype("Int64")


def to_date(s: pd.Series) -> pd.Series:
    """Convertit une date GCAT en datetime a la precision disponible.

    Les dates partielles sont ramenees au debut de la periode connue :
    "1964 Mar" -> 1964-03-01, "1962?" -> 1962-01-01. Le champ associe
    <col>_precision permet de savoir ce qui a ete suppose (cf. add_date_precision).
    """
    txt = s.astype(str).str.strip().str.replace("?", "", regex=False).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    parts = txt.str.split(r"\s+", regex=True)
    year = pd.to_numeric(parts.str[0], errors="coerce")
    month_txt = parts.str[1].fillna("").str.lower().str[:3]
    month = month_txt.map(MONTHS)
    day = pd.to_numeric(parts.str[2], errors="coerce")

    ok = year.notna() & (year >= 1900) & (year <= 2100)
    frame = pd.DataFrame({
        "year": year.where(ok),
        "month": month.where(ok).fillna(1),
        "day": day.where(ok).fillna(1),
    })
    # jours invalides (ex. 31 fevrier issu d'une coquille) -> ramenes au 1er
    frame.loc[(frame["day"] < 1) | (frame["day"] > 31), "day"] = 1
    out.loc[ok] = pd.to_datetime(frame.loc[ok], errors="coerce")
    return out


def date_precision(s: pd.Series) -> pd.Series:
    """Renvoie la precision reelle de chaque date GCAT : 'day', 'month', 'year' ou NA."""
    txt = s.astype(str).str.strip().str.replace("?", "", regex=False).str.strip()
    parts = txt.str.split(r"\s+", regex=True)
    has_year = pd.to_numeric(parts.str[0], errors="coerce").notna()
    has_month = parts.str[1].fillna("").str.lower().str[:3].isin(MONTHS).fillna(False)
    has_day = pd.to_numeric(parts.str[2], errors="coerce").notna()

    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out[has_year] = "year"
    out[has_year & has_month] = "month"
    out[has_year & has_month & has_day] = "day"
    return out


# ---------------------------------------------------------------------------
# Coquilles orbitales
# ---------------------------------------------------------------------------

def shell_table() -> pd.DataFrame:
    """Table de reference des coquilles orbitales (40 bandes de 35 km, 200-1600 km)."""
    lower = SHELL_MIN_KM + SHELL_WIDTH_KM * np.arange(N_SHELLS)
    upper = lower + SHELL_WIDTH_KM
    return pd.DataFrame({
        "shell": np.arange(1, N_SHELLS + 1, dtype=int),
        "alt_lower_km": lower,
        "alt_upper_km": upper,
        "alt_mid_km": (lower + upper) / 2.0,
        "radius_lower_km": R_EARTH_KM + lower,
        "radius_upper_km": R_EARTH_KM + upper,
    })


def mean_altitude(perigee_km: pd.Series, apogee_km: pd.Series) -> pd.Series:
    """Altitude moyenne = demi-grand axe moins rayon terrestre = (perigee + apogee)/2.

    Exact au sens keplerien : a = (r_p + r_a)/2, donc a - R_E = (h_p + h_a)/2.
    """
    return (perigee_km + apogee_km) / 2.0


def assign_shell(alt_km: pd.Series) -> pd.Series:
    """Assigne chaque altitude a une coquille (1 a 40). Hors 200-1600 km -> NA.

    LIMITE ASSUMEE : l'assignation se fait par altitude moyenne, donc un objet
    excentrique est compte dans une seule coquille alors qu'il en traverse
    plusieurs. La ponderation par temps de residence n'est pas implementee ici ;
    la part d'objets concernes est mesuree et reportee (cf. eccentricity_flag).
    """
    idx = np.floor((alt_km - SHELL_MIN_KM) / SHELL_WIDTH_KM) + 1
    idx = idx.where((alt_km >= SHELL_MIN_KM) & (alt_km < SHELL_MAX_KM))
    return idx.astype("Int64")


def eccentricity_flag(perigee_km: pd.Series, apogee_km: pd.Series,
                      tol_km: float = SHELL_WIDTH_KM) -> pd.Series:
    """True si l'objet traverse plus d'une coquille (apogee - perigee > largeur de coquille)."""
    return (apogee_km - perigee_km) > tol_km


# ---------------------------------------------------------------------------
# Typologie des objets
# ---------------------------------------------------------------------------

def object_class(type_field: pd.Series) -> pd.Series:
    """Reduit le champ Type de GCAT a quatre classes exploitables.

    GCAT encode le type sur plusieurs caracteres (ex. 'P A', 'R2', 'D  P', 'C  V').
    Seule la premiere lettre porte la nature de l'objet :
      P = charge utile, R = etage de lanceur, D = debris de fragmentation,
      C = composant largue (coiffe, adaptateur, masselotte...).
    Les classes R, D et C constituent ensemble le stock de debris au sens
    economique : des objets non manoeuvrants qui imposent un risque de collision.
    """
    first = type_field.astype(str).str.strip().str[0].str.upper()
    return first.where(first.isin(["P", "R", "D", "C"]))


def operational_class(active_field: pd.Series, type_field: pd.Series) -> pd.Series:
    """Classe operationnelle, en croisant le champ Active et le champ Type de currentcat.

    Le champ Active de GCAT vaut 'A' pour une charge utile encore operationnelle
    et 'P' pour une charge utile devenue inerte ; pour les non-charges utiles il
    reprend simplement le type. La distinction est decisive pour la modelisation :
    seule une charge utile active peut manoeuvrer pour eviter une collision, une
    epave subit exactement le meme regime qu'un debris.

    Renvoie : ACTIVE_PAYLOAD, DERELICT_PAYLOAD, ROCKET_BODY, DEBRIS, COMPONENT.
    """
    a = active_field.astype(str).str.strip().str.upper()
    t = object_class(type_field)
    out = pd.Series(pd.NA, index=a.index, dtype="object")
    out[(t == "P") & (a == "A")] = ACTIVE_PAYLOAD
    out[(t == "P") & (a != "A")] = DERELICT_PAYLOAD
    out[t == "R"] = ROCKET_BODY
    out[t == "D"] = DEBRIS
    out[t == "C"] = COMPONENT
    return out


ACTIVE_PAYLOAD = "active_payload"
DERELICT_PAYLOAD = "derelict_payload"
ROCKET_BODY = "rocket_body"
DEBRIS = "debris"
COMPONENT = "component"

# Stock manoeuvrant (S dans les modeles Rao) vs stock inerte (D).
MANEUVERABLE = [ACTIVE_PAYLOAD]
NON_MANEUVERABLE = [DERELICT_PAYLOAD, ROCKET_BODY, DEBRIS, COMPONENT]
