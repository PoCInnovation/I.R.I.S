from openclaw import BrowserAgent
import sys
import argparse

def main() :
    # Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--headless', action='store_true', help='Cacher le navigateur')
    args = parser.parse_args()

    # Setup
    config = AgentConfig(
        provider="groq", # Origine model ('groq', 'openai'...etc)
        model="llama-3.3-70b-versatile", # model
        api_key="gsk...",  # API Key
        headless=False
    )
   
    # Start the agent
    agent = BrowserAgent(config=config)
    
    print("Lancement de OpenClaw ->")
    
    # Set the objectif
    objectif = "Va sur pimeyes.com/en envoie la photo donné en argument dans ce script et retourn moi le resultat"
    
    # Get result
    res = agent.browse(instruction=objectif)

    # Print result summary
    print("\nrésultat de l'action")
    print(res.summary)
    
    # Print data
    if res.structured_data:
        print("Données extraites :", res.structured_data)
    
    # Clean agent
    agent.close()

if __name__ == "__main__":
    main()