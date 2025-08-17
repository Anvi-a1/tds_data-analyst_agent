import pytest
from unittest.mock import AsyncMock, MagicMock
from app.tools.parser import parse_html, fetch_and_parse_html, FetchAndParseHtmlResult


@pytest.fixture
def sample_html():
    return """
    <html>
        <body>
            <h1>Title</h1>
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
            <div class="container">
                <span>Span 1</span>
                <span>Span 2</span>
            </div>
        </body>
    </html>
    """


def test_parse_html_no_selector(sample_html):
    """
    Test parsing the entire HTML document when no selector is provided.
    """
    result = parse_html(sample_html)
    assert len(result["results"]) == 1
    assert "Title" in result["results"][0]["text"]
    assert "First paragraph." in result["results"][0]["text"]


def test_parse_html_with_selector(sample_html):
    """
    Test parsing with a CSS selector to extract specific elements.
    """
    result = parse_html(sample_html, selector="p")
    assert len(result["results"]) == 2
    assert result["results"][0]["text"] == "First paragraph."
    assert result["results"][1]["text"] == "Second paragraph."


def test_parse_html_with_selector_and_max_elements(sample_html):
    result = parse_html(sample_html, selector=".container span", max_elements=1)
    assert len(result["results"]) == 1
    assert result["total_elements"] == 2  # before slicing
    assert result["results"][0]["text"] == "Span 1"


def test_parse_html_with_class_selector(sample_html):
    """
    Test parsing with a class selector.
    """
    result = parse_html(sample_html, selector=".container span")
    assert len(result["results"]) == 2
    assert result["results"][0]["text"] == "Span 1"
    assert result["results"][1]["text"] == "Span 2"


def test_parse_html_empty_html():
    """
    Test parsing an empty HTML string.
    """
    result = parse_html("")
    assert len(result["results"]) == 1
    assert result["results"][0]["text"] == ""


@pytest.fixture
def mock_playwright_page(mocker):
    """A fixture to mock the playwright page and its methods."""
    mock_page = AsyncMock()

    # Default successful response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.headers = {"content-type": "text/html"}

    mock_page.goto.return_value = mock_response
    mock_page.content.return_value = "<html><body><p>Test</p></body></html>"

    mock_context = AsyncMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = AsyncMock()
    mock_browser.new_context.return_value = mock_context

    mock_p = AsyncMock()
    mock_p.chromium.launch.return_value = mock_browser

    mock_async_playwright = mocker.patch("app.tools.parser.async_playwright")
    mock_async_playwright.return_value.__aenter__.return_value = mock_p

    return mock_page


@pytest.mark.asyncio
async def test_fetch_and_parse_html_success(mock_playwright_page):
    """
    Test that fetch_and_parse_html successfully fetches and parses HTML.
    """
    result: FetchAndParseHtmlResult = await fetch_and_parse_html(
        "http://example.com", selector="p", max_elements=5
    )

    assert result["url"] == "http://example.com"
    assert result["status"] == 200
    assert len(result["results"]) == 1
    assert result["results"][0]["text"] == "Test"
    # Type/shape checks per FetchAndParseHtmlResult
    assert isinstance(result["results"], list)
    assert isinstance(result["results"][0], dict)
    assert set(["text", "html"]).issubset(result["results"][0].keys())
    assert isinstance(result["url"], str)
    assert isinstance(result["content_type"], str)
    assert isinstance(result["status"], int)
    assert isinstance(result["total_elements"], int)
    assert isinstance(result["total_size"], int)


def test_parse_html_truncation(monkeypatch):
    from app.tools import parser as parser_mod

    # Build an HTML with very long text to trigger truncation
    long_text = "A" * (parser_mod.MAX_ELEMENT_TEXT_SIZE + 100)
    html = f"<html><body><div>{long_text}</div></body></html>"

    res = parse_html(html, selector="div")
    assert len(res["results"]) == 1
    el = res["results"][0]
    assert (
        len(el["text"]) <= parser_mod.MAX_ELEMENT_TEXT_SIZE + 50
    )  # includes truncation marker
    assert "truncated" in el["text"].lower()
