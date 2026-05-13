import asyncio
from bs4 import BeautifulSoup
import os
import random
from models import TargetInfo



async def ft_default(browser, url):

    page = await browser.new_page()

    try:
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        await asyncio.sleep(random.uniform(2, 4))
        await page.mouse.wheel(0, 600)
        await asyncio.sleep(random.uniform(1, 2))
        
        html = await page.evaluate('''() => {
            const main = document.querySelector('main');
            if (main) return main.innerHTML;
            return document.body.innerHTML;
        }''')
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            element.decompose()

        await page.close()
        content = [text for text in soup.stripped_strings]
        content = ' || '.join(content)
        return url, content
    except Exception as e:
        print(f"Failed to scrap {url}: {e}")
        await page.close()
        return url, None
