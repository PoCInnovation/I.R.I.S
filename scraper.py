import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from dotenv import load_dotenv
from default import ft_default
import os
from chatgpt import ft_call_chatgpt
import shutil


async def ft_scraper(urls):

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "Please set OPENAI_API_KEY in .env"

    user_dir = os.path.join(os.getcwd(), "context")
    if not os.path.exists(user_dir):
        print("Please follow the instructions in scraper.py for better results")
        
    async with Stealth().use_async(async_playwright()) as p:

        browser = await p.chromium.launch_persistent_context(
            headless=True, ## False if first time running the script
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            user_data_dir=user_dir,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        # if it is your first time running this script you need to uncomment these lines below which will provides you
        # 10min to connect manually on linkedin insta twitter facebook tiktok or any other platform you need or want
        # and ensure that you allow all cookies or close any pop up and then you can close all the tabs
        # to do this do not forget to pass the headless argument to False in the lines above
        # please do not use your personal accounts use fake ones that you have already created before
        ### await asyncio.sleep(600)
        ### return "Context successfully set! You can now run the script again for better results"

        tasks = []
        for url in urls:
            task = asyncio.create_task(ft_default(browser, url))
            tasks.append(task)
        
        results = []
        for tsk in asyncio.as_completed(tasks):
            url, soup = await tsk
            if soup is None:
                continue
            results.append(f"|START              URL:{url}\nCONTENT:{soup}              END|")
        
        await browser.close()

    scraped_data = "\n".join(results)
    shutil.rmtree('./context/Default/Cache')
    shutil.rmtree('./context/Default/Code Cache')
    shutil.rmtree('./context/Default/Service Worker')
    return ft_call_chatgpt(scraped_data, openai_api_key)



if __name__ == "__main__":

    urls = [
        "https://www.linkedin.com/in/thomas-pesquet/",
        "https://www.facebook.com/ESAThomasPesquet/",
        "https://www.instagram.com/thom_astro/",
        "https://www.tiktok.com/@thom_astro",
        "https://x.com/Thom_astro",
        "https://www.esa.int/Space_in_Member_States/France/L_astronaute_de_l_ESA_Thomas_Pesquet"
    ]
    
    results = asyncio.run(ft_scraper(urls))
    print(results)