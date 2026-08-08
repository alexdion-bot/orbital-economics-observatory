"""
Modele physico-economique du risque de collision orbitale.

Toutes les constantes proviennent de sources publiees et sont tracees ici :

  MOCAT-4S (Lifson et al. 2022 ; implementation dans OPUS, Rao et al. 2023)
    -> geometrie des coquilles, masses/aires/diametres medians MASTER,
       vitesse d'impact, coefficients d'evitement.
  OPUS, Table 2 (Kaffine & Rao 2025, appendice)
    -> parametres economiques : prix du lift, cout du delta-v, revenus.
  Rao, Burgess & Kaffine (PNAS 2020), depot de replication
    -> coefficients aSS/aSD de l'equation de risque agregee, calibres par
       moindres carres non lineaires sur donnees ESA/Space-Track 1957-2015.

Deux approches du risque sont implementees et confrontees :
  (A) cinetique par coquille  -- theorie cinetique des gaz, forme MOCAT-4S,
      applicable coquille par coquille car la densite spatiale est locale.
  (B) agregee calibree Rao    -- forme exponentielle negative ajustee sur
      donnees observees, valable pour l'orbite basse prise globalement.
La (B) sert a valider l'ordre de grandeur de la (A) ; elle ne peut pas etre
appliquee coquille par coquille, ses coefficients englobant tout le volume LEO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes physiques (MOCAT-4S / MASTER)
# ---------------------------------------------------------------------------
R_EARTH_KM = 6378.1366
MU_M3_S2 = 3.986004418e14          # parametre gravitationnel terrestre [m^3/s^2]
SECONDS_PER_YEAR = 24 * 3600 * 365.25

MASS_SAT_KG = 223.0                # masse mediane satellite (MASTER)
AREA_SAT_M2 = 1.741                # section transversale mediane satellite
DIAM_SAT_M = 1.490                 # diametre median satellite
DIAM_DEBRIS_M = 0.180              # diametre median fragment
CD = 2.2                           # coefficient de trainee

V_IMPACT_KM_S = 10.0               # vitesse d'impact caracteristique en LEO
V_IMPACT_M_YEAR = V_IMPACT_KM_S * 1000.0 * SECONDS_PER_YEAR

# Coefficients d'evitement et de population non catalogue (MOCAT-4S).
#
# ALPHA : part des rencontres avec un objet inerte que le satellite actif ne
#   parvient pas a eviter par manoeuvre.
# DELTA : rapport entre la population de fragments letaux NON catalogues
#   (typiquement 1 a 10 cm, non suivis par les reseaux de surveillance mais
#   suffisamment energetiques pour detruire un satellite a 10 km/s) et la
#   population catalogue. MOCAT-4S retient 10, c'est-a-dire qu'il y aurait
#   dix objets letaux non suivis pour chaque objet au catalogue.
# Le facteur applique aux rencontres satellite-objet inerte est (DELTA + ALPHA)
#   et non ALPHA seul : c'est la formulation de MOCAT4S.m lignes 89-90 et 155-156.
#   L'omettre sous-estime le risque d'un facteur cinquante.
# ALPHA_ACTIVE_ACTIVE : part des rencontres entre deux satellites actifs qui
#   aboutit a une collision. Bien plus faible, les deux parties pouvant
#   manoeuvrer et se coordonner. Ne recoit pas le facteur DELTA, un satellite
#   actif etant par construction catalogue.
ALPHA_AVOIDANCE = 0.2
DELTA_UNTRACKED = 10.0
ALPHA_VS_INERT = DELTA_UNTRACKED + ALPHA_AVOIDANCE
ALPHA_ACTIVE_ACTIVE = 0.01

# ---------------------------------------------------------------------------
# Parametres economiques (OPUS, Table 2)
# ---------------------------------------------------------------------------
LIFT_PRICE_USD_PER_KG = 5000.0     # Corrado, Cropper & Rao (2023)
DELTA_V_COST_USD_PER_MS = 1000.0   # $ par m/s
REVENUE_INTERCEPT_USD_YEAR = 7.5e5  # revenu d'un satellite sans concurrence
DISCOUNT_RATE = 0.05
SAT_LIFETIME_YEARS = 5.0
DISPOSAL_TIME_YEARS = 5.0
DELTA_V_SAFETY_FACTOR = 1.5        # marge pour variabilite du flux solaire
DELTA_V_MARGIN_MS = 100.0          # marge de manoeuvres discretionnaires
NONCOMPLIANCE_RATE = 0.0           # phi : taux de non-conformite au desorbitage

# ---------------------------------------------------------------------------
# Coefficients calibres (Rao, Burgess & Kaffine 2020, depot de replication)
# ---------------------------------------------------------------------------
RAO_A_SS = 1.28962893273151e-06    # coefficient satellite-satellite
RAO_A_SD = 2.56125761328431e-08    # coefficient satellite-debris
RAO_B_SS = 292.676086506845        # fragments par collision satellite-satellite
RAO_B_SD = 5159.58121796774        # fragments par collision satellite-debris
RAO_B_LAUNCH = 4.37995731201141    # fragments de mise en orbite par lancement reussi
RAO_DEBRIS_DECAY = 0.503240876878125
RAO_SAT_SURVIVAL = 0.96678375985002

# Reperes publies pour validation
RAO_OUF_2020_USD = 14900.0         # redevance optimale 2020, $/satellite-annee
RAO_OUF_2040_USD = 235000.0        # redevance optimale 2040
RAO_OUF_GROWTH = 0.14              # croissance annuelle de la redevance


# ---------------------------------------------------------------------------
# Atmosphere : modele exponentiel de Vallado (Table 8-4), US Standard 1976 / CIRA-72
# ---------------------------------------------------------------------------
_ATM = np.array([
    [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682],
    [40, 3.972e-3, 7.554], [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714],
    [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799], [90, 3.396e-6, 5.382],
    [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263], [120, 2.438e-8, 9.473],
    [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149], [150, 2.070e-9, 22.523],
    [180, 5.464e-10, 29.740], [200, 2.789e-10, 37.105], [250, 7.248e-11, 45.546],
    [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298], [400, 3.725e-12, 58.515],
    [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822], [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64], [900, 5.245e-15, 181.05],
    [1000, 3.019e-15, 268.00],
])


def atmospheric_density(alt_km):
    """Densite atmospherique [kg/m^3] a l'altitude donnee, modele exponentiel."""
    alt = np.atleast_1d(np.asarray(alt_km, dtype=float))
    alt = np.maximum(alt, 0.0)
    idx = np.searchsorted(_ATM[:, 0], alt, side="right") - 1
    idx = np.clip(idx, 0, len(_ATM) - 1)
    h0, p0, H = _ATM[idx, 0], _ATM[idx, 1], _ATM[idx, 2]
    return p0 * np.exp(-(alt - h0) / H)


# ---------------------------------------------------------------------------
# Geometrie orbitale
# ---------------------------------------------------------------------------
def shell_volume_m3(alt_lower_km, alt_upper_km):
    """Volume d'une coquille spherique entre deux altitudes [m^3]."""
    r_l = (R_EARTH_KM + np.asarray(alt_lower_km, dtype=float)) * 1000.0
    r_u = (R_EARTH_KM + np.asarray(alt_upper_km, dtype=float)) * 1000.0
    return (4.0 / 3.0) * np.pi * (r_u ** 3 - r_l ** 3)


def cross_section_m2(diam_a_m: float, diam_b_m: float) -> float:
    """Section efficace de collision entre deux objets spheriques [m^2].

    Deux objets entrent en collision si la distance entre leurs centres passe
    sous la somme de leurs rayons : sigma = pi * (r_a + r_b)^2.
    """
    return np.pi * ((diam_a_m + diam_b_m) / 2.0) ** 2


SIGMA_SAT_DEBRIS_M2 = cross_section_m2(DIAM_SAT_M, DIAM_DEBRIS_M)
SIGMA_SAT_SAT_M2 = cross_section_m2(DIAM_SAT_M, DIAM_SAT_M)


# ---------------------------------------------------------------------------
# (A) Risque cinetique par coquille
# ---------------------------------------------------------------------------
def collision_rate_per_satellite(n_inert, n_active, volume_m3):
    """Taux de collision annuel subi par UN satellite actif dans une coquille.

    Theorie cinetique : le taux de rencontre entre une particule test et une
    population de densite spatiale n est n * sigma * v_rel. On somme les deux
    populations rencontrees, ponderees par leur probabilite respective de
    ne pas etre evitee.

    Renvoie un taux annuel (evenements par an), pas une probabilite.
    """
    n_inert = np.asarray(n_inert, dtype=float)
    n_active = np.asarray(n_active, dtype=float)
    volume_m3 = np.asarray(volume_m3, dtype=float)

    rate_inert = ALPHA_VS_INERT * (n_inert / volume_m3) * SIGMA_SAT_DEBRIS_M2 * V_IMPACT_M_YEAR
    # un satellite ne peut pas entrer en collision avec lui-meme : n_active - 1
    others = np.maximum(n_active - 1.0, 0.0)
    rate_active = ALPHA_ACTIVE_ACTIVE * (others / volume_m3) * SIGMA_SAT_SAT_M2 * V_IMPACT_M_YEAR
    return rate_inert + rate_active


def rate_to_probability(rate_per_year):
    """Convertit un taux d'evenements en probabilite annuelle (processus de Poisson)."""
    return 1.0 - np.exp(-np.asarray(rate_per_year, dtype=float))


def marginal_risk_imposed(n_active, volume_m3):
    """Hausse de probabilite de collision imposee a UN satellite deja en place
    par l'ajout d'un satellite actif supplementaire dans la meme coquille.

    d/dS [1 - exp(-lambda(S))] = (dlambda/dS) * exp(-lambda)
    """
    n_active = np.asarray(n_active, dtype=float)
    volume_m3 = np.asarray(volume_m3, dtype=float)
    d_lambda = ALPHA_ACTIVE_ACTIVE * (SIGMA_SAT_SAT_M2 * V_IMPACT_M_YEAR) / volume_m3
    return d_lambda


# ---------------------------------------------------------------------------
# (B) Risque agrege calibre (Rao, Burgess & Kaffine 2020)
# ---------------------------------------------------------------------------
def rao_collision_probability(S, D):
    """Probabilite annuelle de collision d'un satellite, forme calibree Rao et al."""
    return 1.0 - np.exp(-(RAO_A_SS * np.asarray(S, dtype=float))
                        - (RAO_A_SD * np.asarray(D, dtype=float)))


def rao_collision_rate(S, D):
    """Nombre attendu de collisions par an sur l'ensemble du parc."""
    return np.asarray(S, dtype=float) * rao_collision_probability(S, D)


def rao_marginal_risk(S, D):
    """Derivee de la probabilite de collision par rapport au nombre de satellites."""
    return RAO_A_SS * np.exp(-(RAO_A_SS * np.asarray(S, dtype=float))
                             - (RAO_A_SD * np.asarray(D, dtype=float)))


# ---------------------------------------------------------------------------
# Fonction de cout d'un satellite (OPUS, buildCostFunction.m)
# ---------------------------------------------------------------------------
def satellite_cost_usd(alt_km, disposal_altitude_km=None):
    """Cout de deploiement d'un satellite a une altitude donnee [$].

    Reimplementation de la fonction de cout d'OPUS. Trois composantes :
      1. prix du lift        : $/kg x masse du satellite
      2. maintien a poste    : budget delta-v pour compenser la trainee,
                               monetise au cout du delta-v
      3. perte de duree de vie : au-dessus de l'altitude naturellement conforme
                               a la regle de desorbitage, une part du budget
                               delta-v part dans la manoeuvre de desorbitage,
                               ce qui ampute la vie utile ; valorise au revenu
                               maximal annuel (borne superieure assumee).

    DIVERGENCE DOCUMENTEE AVEC LA FIGURE PUBLIEE D'OPUS
    Le code d'OPUS (buildCostFunction.m) calcule le delta-v de trainee en km/s
    puis l'additionne a une marge de 100 exprimee en m/s, avant de monetiser
    l'ensemble a 1000 $/unite. La Table 2 du meme article donne pourtant
    p_delta_v = 1000 $/m/s et f_m = 100 m/s, ce qui impose des m/s partout.
    Cette implementation suit la Table 2, c'est-a-dire des m/s de bout en bout.

    Verification externe : elle donne 59 m/s/an de compensation de trainee a
    400 km, coherent avec les ~50 m/s/an de rehaussement reels de la Station
    spatiale internationale. La version en km/s sous-estimerait ce besoin d'un
    facteur mille.

    Consequence pratique : au-dessus de ~450 km, ou se trouve la quasi-totalite
    du parc, les deux versions coincident a quelques pourcents pres. En dessous
    de 300 km la divergence devient massive, mais cette bande est quasi vide
    (131 satellites actifs sur 15 493 dans nos donnees).
    """
    alt = np.atleast_1d(np.asarray(alt_km, dtype=float))

    # 1. lift
    lift = LIFT_PRICE_USD_PER_KG * MASS_SAT_KG

    # 2. delta-v de compensation de trainee, en m/s par an
    rho = atmospheric_density(alt)                     # kg/m^3
    r_m = (R_EARTH_KM + alt) * 1000.0
    v_orb = np.sqrt(MU_M3_S2 / r_m)                    # m/s
    f_drag = CD * 0.5 * rho * v_orb ** 2 * AREA_SAT_M2  # N
    v_drag = f_drag / MASS_SAT_KG * SECONDS_PER_YEAR    # m/s par an

    delta_v_budget = DELTA_V_SAFETY_FACTOR * SAT_LIFETIME_YEARS * v_drag + DELTA_V_MARGIN_MS
    stationkeeping = delta_v_budget * DELTA_V_COST_USD_PER_MS

    # 3. desorbitage : transfert de Hohmann vers l'altitude conforme
    if disposal_altitude_km is None:
        disposal_altitude_km = naturally_compliant_altitude_km()
    r_target = (R_EARTH_KM + disposal_altitude_km) * 1000.0
    dv1 = np.sqrt(MU_M3_S2 / r_m) * (1.0 - np.sqrt(2.0 * r_target / (r_m + r_target)))
    dv2 = np.sqrt(MU_M3_S2 / r_target) * (np.sqrt(2.0 * r_m / (r_m + r_target)) - 1.0)
    deorbit_dv = np.maximum(dv1, 0.0) + np.maximum(dv2, 0.0)
    needs_deorbit = (alt > disposal_altitude_km).astype(float)

    dv_left = np.maximum(0.0, delta_v_budget - deorbit_dv * needs_deorbit)
    lifetime_left = (dv_left / delta_v_budget) * SAT_LIFETIME_YEARS
    lifetime_loss = (SAT_LIFETIME_YEARS - lifetime_left) / SAT_LIFETIME_YEARS
    lifetime_loss_cost = lifetime_loss * REVENUE_INTERCEPT_USD_YEAR

    return lift + stationkeeping + lifetime_loss_cost * (1.0 - NONCOMPLIANCE_RATE)


def naturally_compliant_altitude_km(disposal_time_years: float = DISPOSAL_TIME_YEARS,
                                    alt_grid=None) -> float:
    """Altitude la plus haute d'ou un objet inerte retombe dans le delai reglementaire.

    Le temps de residence marginal a chaque altitude est l'inverse du taux de
    decroissance par trainee ; on cumule depuis le bas jusqu'a epuiser le delai.
    """
    if alt_grid is None:
        alt_grid = np.arange(200.0, 1601.0, 5.0)
    rho = atmospheric_density(alt_grid)
    r_m = (R_EARTH_KM + alt_grid) * 1000.0
    beta = CD * AREA_SAT_M2 / MASS_SAT_KG            # coefficient balistique
    # vitesse de decroissance radiale [m/an], forme MOCAT-4S
    decay_rate_m_year = rho * beta * np.sqrt(MU_M3_S2 * r_m) * SECONDS_PER_YEAR
    step_m = np.gradient(alt_grid) * 1000.0
    residence = step_m / np.maximum(decay_rate_m_year, 1e-30)
    cumulative = np.cumsum(residence)
    compliant = alt_grid[cumulative <= disposal_time_years]
    return float(compliant.max()) if len(compliant) else float(alt_grid[0])


# ---------------------------------------------------------------------------
# Taxe pigouvienne
# ---------------------------------------------------------------------------
def pigouvian_tax_usd(n_active, volume_m3, satellite_value_usd):
    """Taxe pigouvienne implicite [$/satellite-annee] : canal de CONGESTION seul.

    En ajoutant un satellite, l'operateur augmente la probabilite de collision
    de chacun des satellites actifs deja presents, sans en supporter le cout :

        tau = (satellites affectes) x (hausse de risque subie par chacun)
              x (valeur detruite en cas de collision)

    Ce canal est faible car deux satellites actifs peuvent tous deux manoeuvrer
    et se coordonner (ALPHA_ACTIVE_ACTIVE = 1 %). Il ne represente qu'une
    fraction du cout externe reel : voir pigouvian_tax_full.
    """
    n_active = np.asarray(n_active, dtype=float)
    marginal = marginal_risk_imposed(n_active, volume_m3)
    return n_active * marginal * np.asarray(satellite_value_usd, dtype=float)


def pigouvian_tax_pollution_usd(n_active, volume_m3, satellite_value_usd,
                                fragments_per_launch: float = RAO_B_LAUNCH,
                                becomes_derelict: float = NONCOMPLIANCE_RATE):
    """Taxe pigouvienne implicite [$/satellite-annee] : canal de POLLUTION.

    Un satellite mis en orbite n'impose pas seulement une gene instantanee aux
    autres satellites actifs : il injecte durablement de la matiere inerte dans
    la coquille. Deux apports :
      - les debris de mise en orbite, calibres par Rao et al. a 4,38 fragments
        par lancement reussi ;
      - le satellite lui-meme, s'il n'est pas desorbite en fin de vie
        (parametre de non-conformite).

    Chaque objet inerte ajoute releve le risque de TOUS les satellites actifs
    de la coquille, et avec un poids bien superieur a celui d'un satellite
    actif : un debris ne manoeuvre pas, d'ou le facteur (DELTA + ALPHA) au lieu
    de ALPHA_ACTIVE_ACTIVE, soit environ mille fois plus.

    La pollution persiste ensuite plusieurs annees. On actualise le flux de
    dommages futurs par le taux d'actualisation et le taux de retrait naturel
    des debris, tous deux issus de la calibration de Rao et al. :

        facteur de persistance = 1 / (1 - retention / (1 + r))
    """
    n_active = np.asarray(n_active, dtype=float)
    volume_m3 = np.asarray(volume_m3, dtype=float)

    # hausse de taux de collision, par satellite actif, par objet inerte ajoute
    d_lambda_per_inert = ALPHA_VS_INERT * (SIGMA_SAT_DEBRIS_M2 * V_IMPACT_M_YEAR) / volume_m3
    inert_added = fragments_per_launch + becomes_derelict
    persistence = 1.0 / (1.0 - RAO_DEBRIS_DECAY / (1.0 + DISCOUNT_RATE))

    return (n_active * d_lambda_per_inert * inert_added * persistence
            * np.asarray(satellite_value_usd, dtype=float))


def pigouvian_tax_full_usd(n_active, volume_m3, satellite_value_usd, **kwargs):
    """Taxe pigouvienne implicite totale : congestion + pollution.

    LIMITE ASSUMEE : le canal de "persistance de pollution" de Rao & Rondina
    (2025) -- fragments engendres par les collisions futures entre debris,
    croissance autocatalytique -- n'est pas chiffre ici, faute de calibration
    dynamique par coquille. La valeur reste donc une borne inferieure, mais
    nettement moins lache que le seul canal de congestion.
    """
    return (pigouvian_tax_usd(n_active, volume_m3, satellite_value_usd)
            + pigouvian_tax_pollution_usd(n_active, volume_m3, satellite_value_usd, **kwargs))
