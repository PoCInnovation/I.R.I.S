import os
import json
from google import genai


def ft_call_gemini(scraped_data, api_key):

    system_prompt = """
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

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model = "gemini-3.1-flash-lite",
            contents = f"{system_prompt}\n\nDATA\n\n{scraped_data}",
            config = {"response_mime_type": "application/json"}
        )

        results = json.loads(response.text)
        return results
    except Exception as e:
        return f"Error during Gemini API call: {e}"
