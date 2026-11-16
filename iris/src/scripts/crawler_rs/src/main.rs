use spider::tokio;
use spider::website::Website;
use std::env;
use scraper::{Html, Selector};
use url::Url;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let url = &args[1];
    let max_pages = args[2].parse::<u32>().unwrap();

    let mut binding = Website::new(url);
    let website= binding.with_limit(max_pages).with_subdomains(true).with_stealth(true);
    let mut rx = website.subscribe(10);

    let task = tokio::spawn(async move {
        while let Ok(page) = rx.recv().await {
            let page_url = match Url::parse(page.get_url()){
                Ok(url) => url,
                Err(_) => continue
            };
            let soup = Html::parse_document(&page.get_html());
            let selector = Selector::parse("img").unwrap();

            // search for img and concatenate its path with the page url
            for element in soup.select(&selector) {
                if let Some(src) = element.value().attr("src"){
                    if src.ends_with(".svg") || src.starts_with("data:"){
                        continue;
                    }

                    if let Ok(full_url) = page_url.join(src) {
                        println!("{}", full_url);
                    }
                }
            }
        }
    });

    website.crawl().await;  
    website.unsubscribe();
    let _ = task.await;
}