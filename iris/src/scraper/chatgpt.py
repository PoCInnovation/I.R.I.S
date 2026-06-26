import json
import re

from openai import OpenAI

OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
Tu es un expert en OSINT. Analyse les textes bruts suivants, extraits de plusieurs pages web différentes concernant la même cible.
TA MISSION : Mets en corrélation les différentes infos des sites pour trier, dédoublonner et identifier la cible précisément et efficacement.
N'utilise QUE les données envoyées. Aucune autre recherche annexe ou donnée externe.

Génère un profil JSON strict respectant obligatoirement cette structure de base (utilise exactement ces clés) :
{
    "nom": "",
    "age": "",
    "date_de_naissance": "",
    "localisation_adresse": "",
    "email": "",
    "telephone": "",
    "bio": "Un résumé de ce que tu as compris de la personne en croisant les sites",
    "reseaux": ["liste de ses", "nom d'utilisateur", "ou", "url de sites perso"]
}

RÈGLES IMPORTANTES :
1. Si une des catégories de la structure de base est manquante, assigne-lui la valeur "Inconnu".
2. Ajoute ensuite une clé "Infos" (dictionnaire) où tu organiseras librement toutes les autres informations pertinentes trouvées (Métiers, Passions, Formations, Contacts, Proches, etc...).
3. N'intègre ces "Infos" libres que si elles sont crédibles. S'il y a un doute, ajoute " ?" à la fin de la valeur en question, après la valeur pas la clé.
4. Ne renvoie QUE le JSON valide, sans formatage Markdown, sans aucun texte avant ou après.
"""

# GPT sometimes wraps JSON in ```json ... ``` despite instructions. Strip
# any fence before json.loads — cheaper than retrying the API call.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

def _strip_markdown_fence(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def ft_call_chatgpt(scraped_data: str, api_key: str) -> dict | str:
    """Send the scraped corpus to GPT and return the parsed OSINT profile.

    Returns:
        The parsed JSON profile as a dict on success, or an error string
        on any failure (auth, network, malformed JSON). The pipeline
        tolerates both via str() / pprint.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=scraped_data,
        )
        raw = _strip_markdown_fence(response.output_text)
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return f"Error parsing ChatGPT response as JSON: {e}\nRaw: {response.output_text[:500]!r}"
    except Exception as e:
        return f"Error during ChatGPT API call: {e}"
