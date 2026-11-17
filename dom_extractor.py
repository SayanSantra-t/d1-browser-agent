"""dom_extractor.py: Model-agnostic semantic DOM & accessibility candidate extractor.
Resolves rich labels from card containers, headings, aria-labels, and titles so SPA
components (YouTube video cards, GitHub repos, search results) never appear unlabeled.
"""

from __future__ import annotations
from typing import Any, Dict, List

EXTRACTOR_JS = """
(opts) => {
    const maxCandidates = (opts && opts.maxCandidates) || 40;
    document.querySelectorAll('[data-d1-ref]').forEach(el => el.removeAttribute('data-d1-ref'));

    const candidates = [];
    const seen = new Set();
    let counter = 1;

    const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
    const isTimestampOnly = (s) => /^(\\d+:)+\\d+$/.test(s) || /^(LIVE|SHORTS|NEW|\\d+:\\d+\\s*Now playing)$/i.test(s);
    const isGenericCta = (s) => /^(watch|watch now|read more|learn more|subscribe|visit site|open|get started|install|download|shop now|buy now)$/i.test(s);

    const getRegion = (el) => {
        if (el.closest('[role="dialog"], [aria-modal="true"], dialog, .modal, #consent-bump, tp-yt-paper-dialog')) return 'modal';
        if (el.closest('header, #masthead, ytd-masthead, [role="banner"]')) return 'header';
        if (el.closest('nav, [role="navigation"], #guide, ytd-guide-renderer')) return 'nav';
        if (el.closest('footer, [role="contentinfo"]')) return 'footer';
        return 'main';
    };

    const resolveRichLabel = (el) => {
        let label = clean(el.getAttribute('aria-label')) || clean(el.getAttribute('title'));
        let inner = clean(el.innerText || el.textContent);
        if (inner && !isTimestampOnly(inner) && !isGenericCta(inner)) {
            if (!label || isGenericCta(label) || inner.length > label.length) label = inner;
        }

        // Walk up to card container for SPA links (YouTube, GitHub, e-commerce)
        if (!label || isTimestampOnly(label) || isGenericCta(label) || label.length < 3) {
            const card = el.closest(
                'ytd-rich-item-renderer, ytd-video-renderer, ytd-grid-video-renderer, ytd-compact-video-renderer, ytd-channel-renderer, article, [role="listitem"], .Box-row'
            );
            if (card) {
                const titleEl = card.querySelector('#video-title, #channel-title, h1, h2, h3, h4, [id*="title"], a[title]');
                if (titleEl) {
                    const cardTitle = clean(titleEl.getAttribute('title') || titleEl.getAttribute('aria-label') || titleEl.innerText);
                    if (cardTitle && !isTimestampOnly(cardTitle) && !isGenericCta(cardTitle)) {
                        label = cardTitle;
                    }
                }
            }
        }

        if (!label || isGenericCta(label)) {
            const img = el.querySelector('img[alt]');
            if (img) {
                const alt = clean(img.getAttribute('alt'));
                if (alt && !isGenericCta(alt)) label = alt;
            }
        }
        return label;
    };

    // 1. Prioritize visible text/search inputs
    const inputs = document.querySelectorAll(
        'input[type="text"], input[type="search"], input:not([type]), textarea, [role="searchbox"], [role="combobox"]'
    );
    for (const el of inputs) {
        const rect = el.getBoundingClientRect();
        if (rect.width < 12 || rect.height < 12) continue;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;

        const ref = `e${counter++}`;
        el.setAttribute('data-d1-ref', ref);
        const region = getRegion(el);
        const rawLabel = clean(
            el.getAttribute('placeholder') ||
            el.getAttribute('aria-label') ||
            el.getAttribute('title') ||
            el.getAttribute('name') ||
            el.value ||
            'Search query input box'
        );
        const desc = `[${region}] search_input: ${rawLabel.slice(0, 70)}`;
        candidates.push({
            ref,
            tag: el.tagName.toLowerCase(),
            role: 'search_input',
            region,
            is_input: true,
            label: rawLabel,
            y: Math.round(rect.top),
            description: desc
        });
        if (candidates.length >= 3) break;
    }

    // 2. Collect visible interactive elements (main content + header/modal, skipping sidebar clutter)
    let selector = [
        '[role="dialog"] button, [role="dialog"] a[href], [aria-modal="true"] button, [aria-modal="true"] a[href]',
        '#contents a[href], #contents button, ytd-search a[href], ytd-search button',
        'yt-tab-shape, [role="tab"]',
        '[role="main"] a[href], [role="main"] button, main a[href], main button',
        'header a[href], header button, #masthead button'
    ].join(', ');

    let elements = Array.from(document.querySelectorAll(selector));
    if (elements.length < 5) {
        elements = Array.from(document.querySelectorAll('button, a[href], [role="button"], [role="link"], [role="tab"]'));
    }

    for (const el of elements) {
        if (el.closest('#guide, ytd-guide-renderer, #guide-content, ytd-ad-slot-renderer, ytd-promoted-sparkles-web-renderer, ytd-display-ad-renderer, [data-ad-slot]')) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width < 8 || rect.height < 8) continue;
        if (rect.bottom < -50 || rect.top > window.innerHeight * 2.5) continue;

        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
        if (el.getAttribute('aria-hidden') === 'true') continue;
        if (el.hasAttribute('data-d1-ref')) continue;
        if (el.parentElement && el.parentElement.closest('[data-d1-ref]')) continue;

        const tag = el.tagName.toLowerCase();
        const href = el.getAttribute('href') || '';
        const roleAttr = (el.getAttribute('role') || '').toLowerCase();

        // Skip bare thumbnail or generic CTA links if they don't resolve to a real title
        let label = resolveRichLabel(el);
        if (!label || isTimestampOnly(label) || isGenericCta(label)) {
            if (href.includes('watch?v=') || isGenericCta(label)) continue;
            label = 'unlabeled';
        }
        if (label.length > 85) label = label.slice(0, 82) + '...';

        let role = tag;
        if (roleAttr === 'tab' || tag === 'yt-tab-shape') {
            role = 'tab';
        } else if (tag === 'a' || roleAttr === 'link') {
            role = 'link';
            if (href.startsWith('/@') || href.includes('/channel/') || href.includes('/c/')) {
                role = 'channel_link';
            } else if (href.includes('watch?v=') || href.includes('/shorts/')) {
                role = 'video_link';
            }
        } else if (tag === 'button' || roleAttr === 'button') {
            role = 'button';
        }

        const region = getRegion(el);
        const dedupKey = `${role}:${label.toLowerCase()}`;
        if (seen.has(dedupKey)) continue;
        seen.add(dedupKey);

        const ref = `e${counter++}`;
        el.setAttribute('data-d1-ref', ref);

        candidates.push({
            ref,
            tag,
            role,
            region,
            href,
            is_input: false,
            label,
            y: Math.round(rect.top),
            description: `[${region}] ${role}: ${label}`
        });

        if (candidates.length >= maxCandidates) break;
    }

    return candidates;
}
"""


def build_criteria_menu(
    raw_items: List[Dict[str, Any]],
    include_scroll: bool = True,
    include_done: bool = False,
    include_human: bool = False,
    include_idk: bool = False,
) -> Dict[str, str]:
    """Converts extracted DOM items + control primitives into a d1 Choice criteria dict."""
    criteria: Dict[str, str] = {item["ref"]: item["description"] for item in raw_items}
    if include_scroll:
        criteria["opt_scroll"] = "action: Scroll down the page to reveal more content"
    if include_done:
        criteria["opt_done"] = "action: Target is already open and visible on screen, stop"
    if include_human:
        criteria["opt_human"] = "action: Request human help for login, password, or CAPTCHA"
    if include_idk:
        criteria["idk"] = "I don't know"
    return criteria
