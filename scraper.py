import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from dotenv import load_dotenv
from models import TargetInfo
from websites.linkedin import ft_linkedin
from websites.facebook import ft_facebook
from websites.instagram import ft_instagram
from websites.tiktok import ft_tiktok
from websites.twitter import ft_twitter
from default import ft_default
import os


async def ft_scraper(urls):

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Please set GEMINI_API_KEY in .env")
        return None

    user_dir = os.path.join(os.getcwd(), "context")

    async with Stealth().use_async(async_playwright()) as p:

        browser = await p.chromium.launch_persistent_context(
            headless=True, # False if first time running the script
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            user_data_dir=user_dir,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        # if it s your first time running this script you need to uncomment this line below which will provides you
        # 10min to connect manually on linkedin insta twitter facebook tiktok, allow cookies and close all the tabs
        # to do this do not forget to pass the headless argument to False in the lines above
        # please do not use your personal accounts use fake ones that you have already created before
        # await asyncio.sleep(600)
        tasks = []

        for url in urls:
            #if "linkedin.com/in/" in url:
            #    task = asyncio.create_task(ft_linkedin(context, url))
            #elif "https://www.facebook.com/" in url:
            #    task = asyncio.create_task(ft_facebook(browser, url))
            #elif "https://www.instagram.com/" in url:
            #    task = asyncio.create_task(ft_instagram(browser, url))
            #elif "https://www.tiktok.com/" in url:  
            #    task = asyncio.create_task(ft_tiktok(browser, url))
            #elif "https://x.com/" in url:
            #    task = asyncio.create_task(ft_twitter(browser, url))
            #else:
            task = asyncio.create_task(ft_default(browser, url))

            tasks.append(task)
        
        results = []
        for tsk in asyncio.as_completed(tasks):
            url, soup = await tsk
            if soup is None:
                continue
            results.append(f"{url, soup}")
        
        await browser.close()

    return results



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
    for result in results:
        print(result)
        print('\n\n')
