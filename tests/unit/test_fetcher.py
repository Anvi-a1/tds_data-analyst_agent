import pytest
from unittest.mock import AsyncMock, MagicMock

from app.tools.fetcher import fetch_url, scrape_table_data


@pytest.fixture
def mock_playwright_page(mocker):
    """A fixture to mock the playwright page and its methods."""
    mock_page = AsyncMock()

    # Default successful response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.headers = {"content-type": "text/html"}

    mock_page.goto.return_value = mock_response
    mock_page.content.return_value = "<html><body><h1>Test</h1></body></html>"

    mock_context = AsyncMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = AsyncMock()
    mock_browser.new_context.return_value = mock_context

    mock_p = AsyncMock()
    mock_p.chromium.launch.return_value = mock_browser

    mock_async_playwright = mocker.patch("app.tools.fetcher.async_playwright")
    mock_async_playwright.return_value.__aenter__.return_value = mock_p

    return mock_page


@pytest.mark.asyncio
async def test_fetch_url_success(mock_playwright_page):
    """
    Test that fetch_url successfully fetches content from a URL.
    """
    result = await fetch_url("http://example.com")
    assert result["content"] == "<html><body><h1>Test</h1></body></html>"
    assert result["content_type"] == "text/html"
    assert not result["truncated"]


@pytest.mark.asyncio
async def test_fetch_url_http_error(mock_playwright_page):
    """
    Test that fetch_url raises an exception for non-200 HTTP status codes.
    """
    mock_response = MagicMock()
    mock_response.status = 404
    mock_playwright_page.goto.return_value = mock_response

    with pytest.raises(Exception, match="Playwright fetch failed: HTTP 404"):
        await fetch_url("http://example.com")


@pytest.mark.asyncio
async def test_scrape_table_data_success(mock_playwright_page):
    """
    Test that scrape_table_data successfully scrapes table data from a URL.
    """
    mock_table = AsyncMock()
    mock_header_row = AsyncMock()
    mock_header_cell = AsyncMock()
    mock_header_cell.inner_text.return_value = "Header"
    mock_header_row.query_selector_all.return_value = [mock_header_cell]

    mock_data_row = AsyncMock()
    mock_data_cell = AsyncMock()
    mock_data_cell.inner_text.return_value = "Cell"
    mock_data_row.query_selector_all.return_value = [mock_data_cell]

    mock_table.query_selector_all.side_effect = [[mock_header_row], [mock_data_row]]

    mock_playwright_page.query_selector_all.return_value = [mock_table]

    result = await scrape_table_data("http://example.com")

    assert result["table_count"] == 1
    assert len(result["tables"]) == 1
    assert result["tables"][0]["headers"] == ["Header"]
    assert result["tables"][0]["rows"] == [["Cell"]]


@pytest.mark.asyncio
async def test_fetch_url_content_truncation(mock_playwright_page):
    """
    Test that fetch_url truncates content that is too large.
    """
    long_content = "a" * 600000
    mock_playwright_page.content.return_value = long_content

    result = await fetch_url("http://example.com")

    assert len(result["content"]) < len(long_content)
    assert result["content"].endswith("\n<!-- Content truncated due to size -->")
    assert result["truncated"]
