import os
import json
from google import genai


def ft_call_gemini(scraped_data, api_key):

    system_prompt = """
    Tu es un expert en OSINT. Analyse les textes bruts suivants, extraits de plusieurs pages web différentes concernant la même cible.
    TA MISSION : Met en corrélation les différentes infos des sites pour trier, dédoublonner et identifier la cible précisément et efficacement. N'utilise QUE les données envoyées.

    Génère un profil JSON strict respectant obligatoirement cette structure de base :
    'nom'
    'localisation'
    'bio_synthese' (un résumé de ce que tu as compris de la personne en croisant les sites)
    'reseaux_identifies' (liste de ses comptes ou sites perso)

    Ajoute ensuite une clé 'donnees_OSINT' (dictionnaire) où tu organiseras librement toutes les autres informations pertinentes trouvées (métier, passions, contacts etc etc).
    Ne renvoie QUE le JSON valide, sans aucun texte autour.
    """


    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model = "gemini-3.1-flash-lite",
        contents = f"{system_prompt}\n\nDATA\n\n{scraped_data}",
        config = {"response_mime_type": "application/json"}
    )

    results = json.loads(response.text)
    return results