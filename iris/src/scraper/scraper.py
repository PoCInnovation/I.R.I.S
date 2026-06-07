import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from dotenv import load_dotenv
from parser import ft_get_and_parse_html
import os
from chatgpt import ft_call_chatgpt
import shutil
from instagram import ft_instaloader


async def ft_scraper(urls):

    # load .env and get api key
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "Please set OPENAI_API_KEY in .env"

    # check if "context" dir exist otherwise advises the user to follow the instructions
    user_dir = os.path.join(os.getcwd(), "context")
    if not os.path.exists(user_dir):
        print("Please follow the instructions in scraper.py for better results")
        
    async with Stealth().use_async(async_playwright()) as p:

        # launch browser with context to simulate a real user
        browser = await p.chromium.launch_persistent_context(
            headless=True, ## False if first time running the script
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            user_data_dir=user_dir,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        ## INSTRUCTIONS ##
        # if it is your first time running this script you need to uncomment these lines below which will provides you
        # 10min to connect manually on linkedin insta twitter facebook tiktok or any other platform you need or want
        # and ensure that you allow all cookies or close any pop up and then you can close all the tabs
        # to do this do not forget to pass the headless argument to False in the lines above
        # please do not use your personal accounts use fake ones that you have already created before
        # note: you can run it multiple times or increase the sleep time if you want to connect on many other platforms
        ###await asyncio.sleep(600)
        ###return "Context successfully set! You can now run the script again for better results"

        # launch all scraping tasks for each url
        tasks = []
        for url in urls:
            if "instagram.com/" in url:
                task = asyncio.create_task(asyncio.to_thread(ft_instaloader, url))
                tasks.append(task)
            else:
                task = asyncio.create_task(ft_get_and_parse_html(browser, url))
                tasks.append(task)
        
        # collect tasks results as soon as each one is complete and then format them all
        results = []
        for tsk in asyncio.as_completed(tasks):
            url, content = await tsk

            if "instagram.com/" in url:
                if content != None:
                    results.append(f"|START              URL:{url}\nCONTENT:{content}              END|")
                # if insta function fails then fallback on default function
                else:
                    fallback_url, fallback_content = await ft_get_and_parse_html(browser, url)
                    if fallback_content is None:
                        continue
                    results.append(f"|START              URL:{fallback_url}\nCONTENT:{fallback_content}              END|")
            else:
                if content is None:
                    continue
                results.append(f"|START              URL:{url}\nCONTENT:{content}              END|")
        
        await browser.close()

    # delete useless and heavy cache directories
    for folder in ['Cache', 'Code Cache', 'Service Worker']:
        path = f'./context/Default/{folder}'
        if os.path.exists(path):
            shutil.rmtree(path)

    # join all results separating them with a new line to send it properly to chatgpt
    scraped_data = "\n".join(results)
    return ft_call_chatgpt(scraped_data, openai_api_key)
