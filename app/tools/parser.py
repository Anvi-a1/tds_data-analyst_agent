from bs4 import BeautifulSoup
import re
import json
import logging
from urllib.parse import urlparse
from typing import Optional, Dict, List, TypedDict

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

MAX_ELEMENT_TEXT_SIZE = 50000
MAX_TOTAL_RESULTS_SIZE = 200000


class ElementResult(TypedDict):
    """Single parsed element payload."""

    text: str
    html: str


class ParseHtmlResult(TypedDict):
    """Return type for parse_html."""

    results: List[ElementResult]
    # Number of elements that matched the selector (before applying max_elements)
    total_elements: int
    # Approx cumulative size of returned text + html (after truncation/limits)
    total_size: int


class FetchAndParseHtmlResult(ParseHtmlResult):
    """Return type for fetch_and_parse_html (extends ParseHtmlResult)."""

    url: str
    content_type: str
    status: int


def parse_html(
    html: str, selector: str | None = None, *, max_elements: int | None = None, **kwargs
) -> ParseHtmlResult:
    """
    Parse an HTML string with BeautifulSoup, optionally selecting elements via a CSS selector.

    Args:
        html: Full HTML content to parse.
        selector: Optional CSS selector to target specific elements. If omitted, the whole document is used.
        max_elements: Optional limit for how many matched elements to include in results (applies only when selector is provided).
        **kwargs: Ignored; accepted for forward-compatibility.

    Returns:
        ParseHtmlResult dict with keys:
            - results (List[ElementResult]): Each item has:
                - text (str): Text content of the element (may be truncated to MAX_ELEMENT_TEXT_SIZE).
                - html (str): HTML string of the element (may be truncated to MAX_ELEMENT_TEXT_SIZE).
            - total_elements (int): Total number of elements that matched the selector (before applying max_elements).
            - total_size (int): Approx total size of returned text + html across results (after truncation/limits).
    """
    soup = BeautifulSoup(html, "html.parser")
    if selector:
        matched = soup.select(selector)
        matched_total = len(matched)
        if max_elements is not None and max_elements >= 0:
            elements = matched[:max_elements]
        else:
            elements = matched
    else:
        matched_total = 1
        elements = [soup]

    results = []
    total_size = 0

    for el in elements:
        text = el.get_text(strip=True)
        html_str = str(el)

        if len(text) > MAX_ELEMENT_TEXT_SIZE:
            logger.warning(
                f"⚠️  Large element text truncated from {len(text)} to {MAX_ELEMENT_TEXT_SIZE} chars"
            )
            text = text[:MAX_ELEMENT_TEXT_SIZE] + "\n[...element text truncated...]"

        if len(html_str) > MAX_ELEMENT_TEXT_SIZE:
            logger.warning(
                f"⚠️  Large element HTML truncated from {len(html_str)} to {MAX_ELEMENT_TEXT_SIZE} chars"
            )
            html_str = (
                html_str[:MAX_ELEMENT_TEXT_SIZE] + "\n[...element HTML truncated...]"
            )

        result = {"text": text, "html": html_str}
        results.append(result)

        total_size += len(text) + len(html_str)

        if total_size > MAX_TOTAL_RESULTS_SIZE:
            logger.warning(
                f"⚠️  Results truncated at {len(results)} elements due to size limit"
            )
            results.append(
                {"text": "[...additional results truncated due to size...]", "html": ""}
            )
            break

    return {
        "results": results,
        "total_elements": matched_total,
        "total_size": total_size,
    }


async def fetch_and_parse_html(
    url: str,
    selector: str | None = None,
    *,
    max_elements: int | None = None,
    method: str = "GET",
    headers: dict | None = None,
    timeout_seconds: int = 10,
    **kwargs,
) -> FetchAndParseHtmlResult:
    """
    Fetch HTML with Playwright, then parse with BeautifulSoup.

    Args:
        url: The URL to fetch.
        selector: Optional CSS selector to extract specific elements from the fetched HTML.
        max_elements: Optional limit for how many matched elements to include in results (applies only when selector is provided).
        method: HTTP method (default: "GET").
        headers: Custom HTTP headers (optional).
        timeout_seconds: Request timeout in seconds (default: 10).
        **kwargs: Ignored; accepted for forward-compatibility.

    Returns:
        FetchAndParseHtmlResult dict with keys:
            - results (List[ElementResult]): Each item has:
                - text (str): Text content of the element (may be truncated to MAX_ELEMENT_TEXT_SIZE).
                - html (str): HTML string of the element (may be truncated to MAX_ELEMENT_TEXT_SIZE).
            - total_elements (int): Total number of elements that matched the selector (before applying max_elements).
            - total_size (int): Approx total size of returned text + html across results (after truncation/limits).
            - url (str): The fetched URL.
            - content_type (str): Response Content-Type header value (best-effort).
            - status (int): HTTP status code.
    """
    parsed = urlparse(url)
    logger.info(f"🌐 Fetching and parsing URL: {url}")
    logger.debug(
        f"📋 Method: {method}, Timeout: {timeout_seconds}s, Selector: {selector}"
    )

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

            logger.info(f"✅ Successfully fetched {len(content)} characters from {url}")
            logger.debug(f"📋 Content type: {content_type}")

            await browser.close()

        # Reuse parse_html for consistent truncation and sizing behavior
        parsed_result = parse_html(
            content, selector=selector, max_elements=max_elements
        )

        logger.info(f"🔍 Parsed {len(parsed_result['results'])} elements from HTML")

        return {
            **parsed_result,
            "url": url,
            "content_type": content_type,
            "status": response.status,
        }

    except Exception as e:
        logger.error(f"💥 Fetch and parse failed for {url}: {e}")
        raise Exception(f"Fetch and parse failed: {e}")
