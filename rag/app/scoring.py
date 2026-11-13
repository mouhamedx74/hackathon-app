"""Scoring TrendPulse : durabilité (5 piliers + filtre anti-hype) puis opportunité.
Pondérations issues des slides ; elles sont SIMULÉES et à valider avec Product/Insights."""

WEIGHTS = {"persistence": 0.30, "growth": 0.20, "diversity": 0.20,
           "diffusion": 0.15, "adoption": 0.10}
OPP_WEIGHTS = {"durability": 0.45, "fit": 0.40, "timing": 0.15}
ALERT_THRESHOLD = 75      # seuil d'alerte (Opportunity Feed)
DURABLE_THRESHOLD = 60    # seuil « durable vs buzz » (à calibrer sur les 30 cas)
PILLARS = list(WEIGHTS)


def _check(name: str, value: float) -> float:
    if not 0 <= value <= 100:
        raise ValueError(f"{name} doit être entre 0 et 100 (reçu {value})")
    return float(value)


def hype_malus(viral_spike: float) -> float:
    """viral_spike : 0 = aucun pic ; 1 = pic viral maximal non soutenu.
    Malus de -5 % à -15 % dès qu'il y a un pic, 0 sinon."""
    if not 0 <= viral_spike <= 1:
        raise ValueError("viral_spike doit être entre 0 et 1")
    return 0.0 if viral_spike == 0 else 0.05 + 0.10 * viral_spike


def durability_score(pillars: dict, viral_spike: float = 0.0) -> float:
    base = sum(WEIGHTS[k] * _check(k, pillars[k]) for k in PILLARS)
    return round(base * (1 - hype_malus(viral_spike)), 1)


def opportunity_score(durability: float, fit: float, timing: float) -> float:
    return round(OPP_WEIGHTS["durability"] * _check("durability", durability)
                 + OPP_WEIGHTS["fit"] * _check("fit", fit)
                 + OPP_WEIGHTS["timing"] * _check("timing", timing), 1)


def score_trend(pillars: dict, viral_spike: float, fit: float, timing: float) -> dict:
    d = durability_score(pillars, viral_spike)
    o = opportunity_score(d, fit, timing)
    return {"durability": d, "opportunity": o,
            "is_durable": d >= DURABLE_THRESHOLD,
            "alert": o >= ALERT_THRESHOLD,
            "needs_human_validation": True}   # principe Human-in-the-Loop
