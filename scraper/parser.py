import asyncio
from bs4 import BeautifulSoup
import os
import random



async def ft_get_and_parse_html(browser, url):

    page = await browser.new_page()

    try:
        # go to the url and simulate human behiavor
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        await asyncio.sleep(random.uniform(2, 4))
        await page.mouse.wheel(0, 600)
        await asyncio.sleep(random.uniform(1, 2))
        
        # get main first then body if main not present
        html = await page.evaluate('''() => {
            const main = document.querySelector('main');
            if (main) return main.innerHTML;
            return document.body.innerHTML;
        }''')
        soup = BeautifulSoup(html, "html.parser")

        # remove useless html elements
        for element in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            element.decompose()

        await page.close()
        
        # extract only text and join it all with a separator
        content = [text for text in soup.stripped_strings]
        content = ' || '.join(content)
        return url, content
    except Exception as e:
        print(f"Failed to scrap {url}: {e}")
        await page.close()
        return url, None
