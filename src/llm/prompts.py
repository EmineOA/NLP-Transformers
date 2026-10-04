def build_system_prompt(
    rules: str,
    fiche: str,
) -> str:
    return f"""
Tu dois prendre une décision uniquement à partir des règles et de la fiche.

RÈGLES :
{rules}

FICHE :
{fiche}

Contraintes obligatoires :
- N'invente aucune information.
- Applique explicitement les règles fournies.
- Réponds brièvement.
- Termine TOUJOURS par exactement une des lignes suivantes :

[VERDICT: ADMISSIBLE]
[VERDICT: NON_ADMISSIBLE]
[VERDICT: INDETERMINE]

N'utilise jamais [VRAI], [FAUX], [OUI] ou [NON].
""".strip()


def build_system_prompt_v1(
    rules: str,
    fiche: str,
) -> str:
    return f"""
Analyse la situation suivante et donne une décision.

RÈGLES :
{rules}

FICHE :
{fiche}

Explique brièvement ta réponse et donne ton verdict.
""".strip()


def build_system_prompt_v2(
    rules: str,
    fiche: str,
) -> str:
    return f"""
Tu dois prendre une décision uniquement à partir des règles et de la fiche.

RÈGLES :
{rules}

FICHE :
{fiche}

Contraintes obligatoires :
- N'invente aucune information.
- Applique explicitement les règles fournies.
- Réponds brièvement.
- Termine TOUJOURS par exactement une des lignes suivantes :

[VERDICT: ADMISSIBLE]
[VERDICT: NON_ADMISSIBLE]
[VERDICT: INDETERMINE]

N'utilise jamais [VRAI], [FAUX], [OUI] ou [NON].
""".strip()