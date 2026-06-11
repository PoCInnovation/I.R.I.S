import argparse
import os
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

class PimEyesImageSearch:
    def __init__(self, headless=False):
        
        # init self variable 
        self.headless = headless # argument for open the browser or not
        self.playwright = None # litteraly the library
        self.browser_context = None #
        self.page = None #

    def _setup_driver(self):
        
        # setup all ur need like browser / driver...etc
        self.playwright = sync_playwright().start() # necessary, init the lib

        # create cookies dir
        profile_path = os.path.join(os.getcwd(), "playwright_firefox_profile") # name of the dir
        if not os.path.exists(profile_path):
            os.makedirs(profile_path)

        # open the browser (Firefox)
        self.browser_context = self.playwright.firefox.launch_persistent_context(
            user_data_dir=profile_path, # cookies dir
            headless=self.headless, # if we make the browser visual or not
            viewport={'width': 1920, 'height': 1080}, # size of the windows
            permissions=[], # No permissions
        )
        # open a new page
        self.page = self.browser_context.pages[0] if self.browser_context.pages else self.browser_context.new_page()

    def search_image(self, image_path, wait_time=45):

        # dictionnary, say if its a success, the url and if not, the error
        result = {'success': False, 'url': None, 'error': None}

        # check if we the image exist
        if not os.path.exists(image_path):
            result['error'] = f"Image non trouvée: {image_path}"
            return result

        # if we dont have page, we open one
        if not self.page:
            self._setup_driver()

        try:
            # open PimEyes
            print("Navigation vers PimEyes...")
            self.page.goto("https://pimeyes.com/en") # open the link
            self.page.wait_for_load_state('networkidle') # wait the load

            abs_image_path = os.path.abspath(image_path) # create path to root to image path
            file_input = self.page.locator('input[type="file"]') # search where we put an image
            
            if file_input.count() > 0:
                print("Upload de l'image...")
                file_input.first.set_input_files(abs_image_path) # set the image where we find the balise

                search_btn = self.page.locator('button:has-text("Start Search")').first # search the button start search
                
                try:
                    search_btn.wait_for(state="visible", timeout=10000) # wait is visible
                    self.page.wait_for_timeout(2000) # wait the load

                    search_btn.click(timeout=60000) # click
                
                # if timeout
                except PlaywrightTimeoutError:
                    result['error'] = "Timeout : Le bouton de recherche est resté bloqué ou introuvable."
                    return result

                try:
                    print("Analyse en cours, attente des résultats...")
                    self.page.wait_for_url(
                        lambda url: "/results/" in url.lower() or "captcha" in url.lower() or "challenge" in url.lower(),
                        timeout=wait_time * 1000
                    )

                    current_url = self.page.url
                    if "captcha" in current_url.lower() or "challenge" in current_url.lower() or "i am human" in current_url.lower():
                        result['error'] = "PimEyes a bloqué la requête après le clic."
                        result['url'] = current_url
                    else:
                        result['url'] = current_url
                        result['success'] = True

                except PlaywrightTimeoutError:
                    result['error'] = f"Timeout lors de l'attente des résultats. url actuelle: {self.page.url}"
            else:
                result['error'] = "Impossible de trouver l'input d'upload."

        except Exception as e:
            result['error'] = f"Erreur: {str(e)}"

        return result

    def close(self):
        if self.browser_context:
            self.browser_context.close()
        if self.playwright:
            self.playwright.stop()

    def __enter__(self):
        self._setup_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--headless', action='store_true', help='Cacher le navigateur')
    args = parser.parse_args()

    with PimEyesImageSearch(headless=args.headless) as searcher:
        result = searcher.search_image(args.image_path)
        if not result['success']:
            print(f"ERREUR : {result['error']}")
        else:
            print(f"SUCCÈS : {result['url']}")

if __name__ == "__main__":
    main()