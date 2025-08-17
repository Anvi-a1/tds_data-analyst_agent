import logging
from urllib.parse import urlparse

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

MAX_CONTENT_SIZE = 500000
MAX_CONTENT_FOR_CONTEXT = 200000


async def fetch_url(
    url: str,
    method: str = "GET",
    headers: dict = None,
    timeout_seconds: int = 10,
    **kwargs,
) -> dict:
    parsed = urlparse(url)
    logger.info(f"🌐 Fetching URL with Playwright: {url}")
    logger.debug(f"📋 Method: {method}, Timeout: {timeout_seconds}s")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            if headers:
                await page.set_extra_http_headers(headers)
                logger.debug(f"📋 Set custom headers: {list(headers.keys())}")

            if method.upper() == "GET":
                response = await page.goto(
                    url, timeout=timeout_seconds * 1000, wait_until="domcontentloaded"
                )
            else:
                response = await page.goto(
                    url, timeout=timeout_seconds * 1000, wait_until="domcontentloaded"
                )

            if response.status != 200:
                await browser.close()
                raise Exception(f"HTTP {response.status}")

            await page.wait_for_timeout(1000)

            content = await page.content()
            content_type = response.headers.get("content-type", "text/html")

            original_size = len(content)
            if original_size > MAX_CONTENT_SIZE:
                logger.warning(
                    f"⚠️  Large content detected ({original_size} chars). Truncating to {MAX_CONTENT_SIZE} chars"
                )
                content = (
                    content[:MAX_CONTENT_SIZE]
                    + "\n<!-- Content truncated due to size -->"
                )

            logger.info(f"✅ Successfully fetched {len(content)} characters from {url}")
            logger.debug(f"📋 Content type: {content_type}")

            await browser.close()

        return {
            "content": content,
            "content_type": content_type,
            "original_size": original_size,
            "truncated": original_size > MAX_CONTENT_SIZE,
        }

    except Exception as e:
        logger.error(f"💥 Playwright fetch failed for {url}: {e}")
        raise Exception(f"Playwright fetch failed: {e}")


async def scrape_table_data(
    url: str, table_selector: str = "table", timeout_seconds: int = 10, **kwargs
) -> dict:
    logger.info(f"📊 Scraping table data from: {url}")
    logger.debug(f"🎯 Table selector: {table_selector}")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            response = await page.goto(
                url, timeout=timeout_seconds * 1000, wait_until="domcontentloaded"
            )

            if response.status != 200:
                await browser.close()
                raise Exception(f"HTTP {response.status}")

            await page.wait_for_timeout(2000)

            tables = await page.query_selector_all(table_selector)
            table_data = []

            for i, table in enumerate(tables):
                logger.debug(f"📋 Processing table {i+1}/{len(tables)}")

                headers = []
                header_rows = await table.query_selector_all("thead tr, tr:first-child")
                if header_rows:
                    header_cells = await header_rows[0].query_selector_all("th, td")
                    headers = [await cell.inner_text() for cell in header_cells]
                    headers = [h.strip() for h in headers]

                rows = []
                data_rows = await table.query_selector_all("tbody tr, tr")

                max_rows = 1000
                if len(data_rows) > max_rows:
                    logger.warning(
                        f"⚠️  Large table detected ({len(data_rows)} rows). Sampling {max_rows} rows"
                    )
                    sample_size = max_rows // 3
                    sampled_rows = (
                        data_rows[:sample_size]
                        + data_rows[
                            len(data_rows) // 2 : len(data_rows) // 2 + sample_size
                        ]
                        + data_rows[-sample_size:]
                    )
                    data_rows = sampled_rows

                for row in data_rows:
                    cells = await row.query_selector_all("td, th")
                    if cells:
                        row_data = []
                        for cell in cells:
                            text = await cell.inner_text()
                            row_data.append(text.strip())
                        if not headers or row_data != headers:
                            rows.append(row_data)

                if rows:
                    table_info = {
                        "table_index": i,
                        "headers": (
                            headers
                            if headers
                            else [f"Column_{j+1}" for j in range(len(rows[0]))]
                        ),
                        "rows": rows,
                        "row_count": len(rows),
                    }
                    table_data.append(table_info)

            await browser.close()

            logger.info(
                f"✅ Extracted {len(table_data)} tables with total {sum(t['row_count'] for t in table_data)} rows"
            )

        return {"tables": table_data, "table_count": len(table_data)}

    except Exception as e:
        logger.error(f"💥 Table scraping failed for {url}: {e}")
        raise Exception(f"Table scraping failed: {e}")
