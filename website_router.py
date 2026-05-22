"""URL classifier and fetch handlers for all site types (Types 1–10)."""

import asyncio
import hashlib
import logging
import os
import random
import re
import tempfile
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from identity_pool import identity_pool, can_fetch, DELAY_PROFILE

logger = logging.getLogger(__name__)

HARD_PAYWALL_DOMAINS = {
    "bloomberg.com", "spglobal.com", "gartner.com", "idc.com",
    "forrester.com", "woodmac.com", "ihsmarkit.com", "wsj.com",
}

SOFT_PAYWALL_DOMAINS = {
    "ft.com", "reuters.com", "theregister.com", "computerweekly.com",
    "hbr.org", "economist.com",
}


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")


def _is_js_rendered(text: str) -> bool:
    return len(text.split()) < 500


def classify_url(url: str, head_status: int | None = None) -> str:
    if "linkedin.com" in url:
        return "TYPE_7_LINKEDIN"
    if url.lower().endswith(".pdf"):
        return "TYPE_8_PDF"
    d = _domain(url)
    if any(d == hw or d.endswith("." + hw) for hw in HARD_PAYWALL_DOMAINS):
        return "TYPE_4_HARD_PAYWALL"
    if any(d == sw or d.endswith("." + sw) for sw in SOFT_PAYWALL_DOMAINS):
        return "TYPE_3_SOFT_PAYWALL"
    if head_status in (403, 429, 503):
        return "TYPE_6_BOT_PROTECTED"
    return "TYPE_1_STATIC_HTML"


async def _get_playwright_page(url: str, wait_for: str = "networkidle", timeout: int = 30000):
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import stealth_async
    except ImportError:
        return None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        await stealth_async(page)
        try:
            await page.goto(url, wait_until=wait_for, timeout=timeout)
            content = await page.inner_text("body")
            html = await page.content()
            await browser.close()
            return content, html
        except Exception as e:
            logger.warning(f"Playwright error on {url}: {e}")
            await browser.close()
            return None


JINA_PREFERRED_DOMAINS = {
    "businesswire.com", "prnewswire.com", "globenewswire.com",
    "sec.gov", "bseindia.com", "nseindia.com",
    "moneycontrol.com", "economictimes.com", "financialexpress.com",
    "livemint.com", "thehindu.com", "hindustantimes.com",
    "finextra.com", "bankingtech.com", "paymentssource.com",
    "cio.com", "cioinsight.com", "computerweekly.com",
    "zdnet.com", "techrepublic.com", "theregister.com",
}


async def fetch_type1_static(url: str) -> tuple[str, str] | None:
    """Static HTML fetch. For known bot-blocking domains, use Jina Reader directly."""
    d = _domain(url)
    path = urlparse(url).path or "/"

    # Skip direct fetch for domains that always block bots — go straight to Jina
    if any(d == jd or d.endswith("." + jd) for jd in JINA_PREFERRED_DOMAINS):
        result = await fetch_via_jina_reader(url)
        if result:
            return result
        # Fall through to direct fetch as last resort

    if not can_fetch(d, path):
        logger.info(f"ROBOTS_BLOCKED: {url}")
        return None

    await identity_pool.wait_for_domain(url, "static")
    headers = identity_pool.next_headers()

    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            body = (
                soup.select_one("article")
                or soup.select_one(".body-content")
                or soup.select_one(".release-body")
                or soup.select_one(".bw-release-story")
                or soup.select_one(".article-body")
                or soup.find("main")
                or soup.find("body")
            )
            text = body.get_text(separator=" ", strip=True) if body else r.text
            # Check if JS-rendered (too little content) — try Jina Reader
            if _is_js_rendered(text):
                return await fetch_type2_js(url)
            return text, r.text
    except Exception as e:
        logger.warning(f"TYPE1 fetch failed {url}: {e}")
        # Fallback to Jina Reader on fetch error
        return await fetch_via_jina_reader(url)


async def fetch_type2_js(url: str) -> tuple[str, str] | None:
    """JS-rendered SPA — Jina Reader first, scrape.do fallback."""
    result = await fetch_via_jina_reader(url)
    if result:
        return result
    return await fetch_via_scrapedo(url)


async def fetch_type3_soft_paywall(url: str) -> tuple[str, str] | None:
    """Soft paywall bypass — try multiple strategies."""
    await identity_pool.wait_for_domain(url, "js")

    # 1. Reader mode race — grab at DOMContentLoaded
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import stealth_async
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await context.new_page()
            await stealth_async(page)
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            content = await page.evaluate("document.body.innerText")
            html = await page.content()
            await browser.close()
            if content and len(content.split()) > 200:
                logger.info(f"TYPE3 reader-mode success: {url}")
                return content, html
    except Exception as e:
        logger.debug(f"TYPE3 reader mode failed: {e}")

    # 2. Wayback Machine
    try:
        wb = await _fetch_wayback(url)
        if wb:
            logger.info(f"TYPE3 wayback success: {url}")
            return wb
    except Exception:
        pass

    # 3. Google Cache
    cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            r = await client.get(cache_url, headers=identity_pool.next_headers())
            soup = BeautifulSoup(r.text, "lxml")
            # Strip Google header
            for tag in soup.select(".google-cache-hdr"):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            if len(text.split()) > 200:
                logger.info(f"TYPE3 google cache success: {url}")
                return text, r.text
    except Exception:
        pass

    # 4. scrape.do — JS render + paywall bypass last resort
    result = await fetch_via_scrapedo(url)
    if result:
        logger.info(f"TYPE3 scrape.do success: {url}")
        return result

    logger.warning(f"soft_paywall_blocked: {url}")
    return None


async def fetch_type4_hard_paywall(url: str) -> tuple[str, str] | None:
    """Hard paywall — extract meta only."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url, headers=identity_pool.next_headers())
            soup = BeautifulSoup(r.text, "lxml")
            title = (
                (soup.find("meta", property="og:title") or {}).get("content", "")
                or soup.title.string if soup.title else ""
            )
            desc = (
                (soup.find("meta", property="og:description") or {}).get("content", "")
                or (soup.find("meta", attrs={"name": "description"}) or {}).get("content", "")
            )
            first_p = soup.find("p")
            first_text = first_p.get_text(strip=True) if first_p else ""
            combined = f"{title}\n{desc}\n{first_text}\n[Full article behind paywall; headline and abstract only.]"
            return combined, r.text
    except Exception as e:
        logger.warning(f"TYPE4 meta extract failed {url}: {e}")
        return None


async def fetch_type6_bot_protected(url: str, proxy_pool: list[str] | None = None) -> tuple[str, str] | None:
    """Bot-protected sites — stealth + proxy fallbacks."""
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import stealth_async
        async with async_playwright() as p:
            proxy = None
            if proxy_pool:
                proxy = {"server": random.choice(proxy_pool)}
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                proxy=proxy
            )
            page = await context.new_page()
            await stealth_async(page)
            await page.goto(url, wait_until="networkidle", timeout=45000)
            await page.mouse.move(300 + random.random() * 200, 200 + random.random() * 150)
            await asyncio.sleep(1 + random.random() * 2)
            content = await page.inner_text("body")
            html = await page.content()
            await browser.close()
            if content and len(content.split()) > 100:
                return content, html
    except Exception as e:
        logger.warning(f"TYPE6 stealth failed {url}: {e}")

    # Browserless fallback
    bl_key = os.getenv("BROWSERLESS_KEY")
    if bl_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"https://chrome.browserless.io/content?token={bl_key}",
                    json={"url": url, "waitFor": "networkidle"},
                )
                soup = BeautifulSoup(r.text, "lxml")
                return soup.get_text(separator=" ", strip=True), r.text
        except Exception as e:
            logger.warning(f"TYPE6 browserless failed: {e}")

    # scrape.do fallback — handles JS rendering + Cloudflare bypass
    sd_key = os.getenv("SCRAPEDO_KEY")
    if sd_key:
        result = await fetch_via_scrapedo(url, sd_key)
        if result:
            logger.info(f"TYPE6 scrape.do success: {url}")
            return result

    logger.warning(f"bot_protected_blocked: {url}")
    return None


async def fetch_type7_linkedin(linkedin_url: str, config=None) -> list[str]:
    """LinkedIn posts — returns list of post texts."""
    posts = []

    # Option A: Proxycurl
    pc_key = os.getenv("PROXYCURL_KEY")
    if pc_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                params = {"linkedin_company_profile_url": linkedin_url}
                r = await client.get(
                    "https://nubela.co/proxycurl/api/v2/linkedin/company/posts",
                    params=params,
                    headers={"Authorization": f"Bearer {pc_key}"},
                )
                data = r.json()
                for post in data.get("posts", []):
                    text = post.get("text", "")
                    if text:
                        posts.append(text)
            if posts:
                return posts
        except Exception as e:
            logger.warning(f"Proxycurl failed: {e}")

    # Option B: Playwright login
    li_user = os.getenv("LINKEDIN_USER")
    li_pass = os.getenv("LINKEDIN_PASS")
    if li_user and li_pass and config and config.run_linkedin:
        try:
            from playwright.async_api import async_playwright
            from playwright_stealth import stealth_async
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(viewport={"width": 1280, "height": 800})
                page = await context.new_page()
                await stealth_async(page)
                await page.goto("https://www.linkedin.com/login")
                await page.fill("#username", li_user)
                await page.fill("#password", li_pass)
                await page.click("button[type=submit]")
                await page.wait_for_url("**/feed**", timeout=20000)
                await page.goto(linkedin_url)

                # Infinite scroll (Type 5 strategy)
                prev_count = 0
                max_posts = 50
                for _ in range(20):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(random.uniform(4.0, 8.0))
                    items = await page.query_selector_all(".feed-shared-update-v2")
                    if len(items) == prev_count or len(items) >= max_posts:
                        break
                    prev_count = len(items)

                for item in items[:max_posts]:
                    text = await item.inner_text()
                    posts.append(text)
                await browser.close()
        except Exception as e:
            logger.warning(f"LinkedIn playwright failed: {e}")

    # Option C: Google search for public posts
    if not posts:
        company_name = config.company_name if config else ""
        query = f'site:linkedin.com/posts "{company_name}" deal OR contract OR selected'
        posts.append(f"[LinkedIn search fallback — query: {query}]")

    return posts


async def fetch_type8_pdf(url: str) -> str | None:
    """Download and extract text from PDF."""
    try:
        sha = hashlib.sha256(url.encode()).hexdigest()[:16]
        tmp_path = os.path.join(tempfile.gettempdir(), f"{sha}.pdf")

        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(url, headers=identity_pool.next_headers())
            with open(tmp_path, "wb") as f:
                f.write(r.content)

        text = _extract_pdf_text(tmp_path)
        os.remove(tmp_path)
        return text
    except Exception as e:
        logger.warning(f"TYPE8 PDF failed {url}: {e}")
        return None


def _extract_pdf_text(path: str) -> str:
    # A: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
        if text.strip():
            return text
    except Exception:
        pass

    # B: PyMuPDF
    try:
        import fitz
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        if text.strip():
            return text
    except Exception:
        pass

    # C: OCR fallback
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(path)
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        logger.warning(f"PDF OCR failed: {e}")
        return ""


async def fetch_via_jina_reader(url: str) -> tuple[str, str] | None:
    """Fetch any URL via Jina Reader (r.jina.ai) — renders JS, bypasses many blocks.
    Returns clean markdown text. Uses the same JINA_KEY as the search API.
    Free tier available; much more reliable than scrape.do for news sites.
    """
    key = os.getenv("JINA_KEY", "")
    if not key:
        return None
    try:
        reader_url = f"https://r.jina.ai/{url}"
        # Domain-specific content selectors to skip nav/footer boilerplate
        d = _domain(url)
        content_selector = None
        if "businesswire.com" in d:
            content_selector = ".bw-release-story"
        elif "prnewswire.com" in d:
            content_selector = ".release-body"
        elif "globenewswire.com" in d:
            content_selector = ".article-body"

        headers: dict = {
            "Authorization": f"Bearer {key}",
            "Accept": "text/plain",
            "X-Return-Format": "text",
            "X-Remove-Selector": "nav, header, footer, .sidebar, .related-news, script, style",
        }
        if content_selector:
            headers["X-Target-Selector"] = content_selector

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(reader_url, headers=headers)
            if r.status_code != 200:
                logger.warning(f"Jina Reader {r.status_code} for {url}")
                # On 4xx/5xx try without target selector (broader extraction)
                if content_selector:
                    headers.pop("X-Target-Selector", None)
                    r = await client.get(reader_url, headers=headers)
                if r.status_code != 200:
                    return None
            text = r.text.strip()
            if len(text.split()) < 50:
                return None
            logger.info(f"Jina Reader success: {url[:80]} ({len(text)} chars)")
            return text, text   # no raw HTML in reader mode
    except Exception as e:
        logger.warning(f"Jina Reader failed for {url}: {e}")
        return None


async def fetch_via_scrapedo(url: str, token: str | None = None) -> tuple[str, str] | None:
    """Fetch any URL via scrape.do — JS rendering + Cloudflare bypass."""
    from urllib.parse import quote
    key = token or os.getenv("SCRAPEDO_KEY", "")
    if not key:
        return None
    scrape_url = f"https://api.scrape.do?token={key}&url={quote(url, safe='')}&render=true"
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.get(scrape_url)
            if r.status_code != 200:
                logger.warning(f"scrape.do fetch {r.status_code} for {url}")
                return None
            soup = BeautifulSoup(r.text, "lxml")
            body = (
                soup.select_one("article")
                or soup.select_one(".body-content")
                or soup.select_one(".release-body")
                or soup.select_one(".bw-release-story")
                or soup.select_one(".article-body")
                or soup.find("main")
                or soup.find("body")
            )
            text = body.get_text(separator=" ", strip=True) if body else r.text
            if len(text.split()) < 50:
                return None
            return text, r.text
    except Exception as e:
        logger.warning(f"scrape.do fetch failed for {url}: {e}")
        return None


async def _fetch_wayback(url: str) -> tuple[str, str] | None:
    async with httpx.AsyncClient(timeout=8) as client:
        avail = await client.get(
            "https://archive.org/wayback/available",
            params={"url": url}
        )
        data = avail.json()
        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if snapshot.get("available") and snapshot.get("url"):
            r = await client.get(snapshot["url"])
            soup = BeautifulSoup(r.text, "lxml")
            return soup.get_text(separator=" ", strip=True), r.text
    return None


async def fetch_url(url: str, proxy_pool: list[str] | None = None, config=None) -> tuple[str | None, str | None, str]:
    """
    Main entry: classify URL, dispatch to correct handler.
    Returns (text, html, url_type).
    """
    url_type = classify_url(url)

    if url_type == "TYPE_1_STATIC_HTML":
        result = await fetch_type1_static(url)
    elif url_type == "TYPE_2_JS_RENDERED":
        result = await fetch_type2_js(url)
    elif url_type == "TYPE_3_SOFT_PAYWALL":
        result = await fetch_type3_soft_paywall(url)
    elif url_type == "TYPE_4_HARD_PAYWALL":
        result = await fetch_type4_hard_paywall(url)
    elif url_type == "TYPE_6_BOT_PROTECTED":
        result = await fetch_type6_bot_protected(url, proxy_pool)
    elif url_type == "TYPE_7_LINKEDIN":
        posts = await fetch_type7_linkedin(url, config)
        return "\n\n".join(posts), None, url_type
    elif url_type == "TYPE_8_PDF":
        text = await fetch_type8_pdf(url)
        return text, None, url_type
    else:
        result = await fetch_type1_static(url)

    # No wayback fallback in API mode — too slow, per-URL timeout handles failures
    if result:
        return result[0], result[1], url_type
    return None, None, url_type
