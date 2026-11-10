import os
import sys
import time
import re
from playwright.sync_api import sync_playwright

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from d1_engine import D1DecisionEngine

def extract_search_text(goal: str) -> str:
    quotes = re.findall(r'["\']([^"\']+)["\']', goal)
    if quotes:
        return quotes[0].strip()
    text = goal.strip()
    text = re.sub(r'^(?:please\s+)?(?:search(?:\s+for)?|find|open|look\s+up|go\s+to|navigate\s+to|play|show(?:\s+me)?)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+(?:on|in|at)\s+(?:youtube|google|wikipedia|amazon|reddit|twitter|github|bing)(?:\.com)?(?:\s+channel|\s+website|\s+page)?$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+(?:youtube\s+channel|channel|official\s+channel|website|page|video)$', '', text, flags=re.IGNORECASE)
    return text.strip()

EXTRACTOR_JS = """
() => {
    document.querySelectorAll('[data-d1-ref]').forEach(el => el.removeAttribute('data-d1-ref'));
    const candidates = [];
    let counter = 1;
    
    // 1. First prioritize search/text input boxes
    const inputs = document.querySelectorAll(
        'input[type="text"], input[type="search"], input:not([type]), textarea, [role="searchbox"], [role="combobox"]'
    );
    for (const el of inputs) {
        const rect = el.getBoundingClientRect();
        if (rect.width < 10 || rect.height < 10) continue;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;
        
        const ref = `e${counter++}`;
        el.setAttribute('data-d1-ref', ref);
        let label = el.getAttribute('placeholder') || el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('name') || el.value || 'search input box';
        candidates.push({
            ref: ref,
            tag: el.tagName.toLowerCase(),
            is_input: true,
            description: `input_box: ${label.trim()}`
        });
        if (candidates.length >= 3) break;
    }

    // 2. Next collect visible clickable links and buttons
    const elements = document.querySelectorAll(
        'button, a[href], [role="button"], [role="link"]'
    );
    for (const el of elements) {
        const rect = el.getBoundingClientRect();
        if (rect.width < 8 || rect.height < 8) continue;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
        if (el.getAttribute('aria-hidden') === 'true') continue;
        if (el.hasAttribute('data-d1-ref')) continue;
        if (el.parentElement && el.parentElement.closest('[data-d1-ref]')) continue;
        
        const ref = `e${counter++}`;
        el.setAttribute('data-d1-ref', ref);
        let label = el.innerText || el.textContent || '';
        label = label.trim().replace(/\\s+/g, ' ');
        if (!label) {
            label = el.getAttribute('aria-label') || el.getAttribute('title') || '';
        }
        if (label.length > 60) label = label.slice(0, 57) + '...';
        
        const tag = el.tagName.toLowerCase();
        let role = tag;
        if (tag === 'a') {
            role = 'link';
            const href = el.getAttribute('href') || '';
            if (href.startsWith('/@') || href.includes('/channel/') || href.includes('/c/')) {
                role = 'channel_link';
            } else if (href.includes('watch?v=')) {
                role = 'video_link';
            }
        }
        
        candidates.push({
            ref: ref,
            tag: tag,
            is_input: false,
            description: `${role}: ${label || 'unlabeled'}`
        });
        if (candidates.length >= 45) break;
    }
    return candidates;
}
"""

def run_test():
    engine = D1DecisionEngine(device="cuda")
    engine.load()

    goal = "open techchip youtube channel"
    target = extract_search_text(goal)
    print(f"Goal: '{goal}'")
    print(f"Target Query: '{target}'")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://127.0.0.1:9110")
        ctx = browser.contexts[0]
        page = ctx.new_page()

        print("\nNavigating to https://www.youtube.com ...")
        page.goto("https://www.youtube.com", wait_until="domcontentloaded")
        time.sleep(2.0)

        history = []
        step = 1
        max_safety_steps = 20

        while step <= max_safety_steps:
            title = page.title()
            url = page.url
            print(f"\n================ STEP {step} ================")
            print(f"URL: {url}")
            print(f"Title: {title}")

            # Check if destination channel is reached
            if step > 1 and "/results" not in url:
                target_clean = re.sub(r'[^a-zA-Z0-9]', '', target.lower())
                url_clean = re.sub(r'[^a-zA-Z0-9]', '', url.lower())
                title_clean = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
                if (target_clean in url_clean or target_clean in title_clean) and ("youtube" in url_clean):
                    print(f"\n[DONE] Target channel '{target}' successfully opened!")
                    print(f"Current Title: {title} | URL: {url}")
                    print(f"Dynamic stopping triggered: Completed in {step - 1} actions!")
                    break

            raw_items = page.evaluate(EXTRACTOR_JS)
            elements_map = {item["ref"]: item for item in raw_items}
            criteria = {item["ref"]: item["description"] for item in raw_items}
            criteria["opt_done"] = f"action: Target '{target}' is already opened and visible on screen, stop"

            print(f"Extracted {len(criteria)} candidate elements.")

            # Decision
            t0 = time.time()
            decision = engine.select_element_action(
                goal=goal,
                current_url=url,
                page_title=title,
                elements=criteria,
                history=history
            )
            lat = (time.time() - t0) * 1000.0

            chosen_ref = decision["chosen_element"]
            chosen_desc = criteria.get(chosen_ref, "unknown")
            conf = decision["element_confidence"]

            print(f"D1 Selected: {chosen_ref} ({chosen_desc}) | Conf: {conf:.1%} | Latency: {lat:.1f}ms")

            if chosen_ref == "opt_done" or decision["action_type"] == "stop":
                print(f"\n[DONE] D1 selected stop action! Task completed at step {step}.")
                break

            locator = page.locator(f'[data-d1-ref="{chosen_ref}"]')
            if locator.count() == 0:
                print(f"Element {chosen_ref} not found on page, refreshing...")
                step += 1
                time.sleep(1.0)
                continue

            elem_info = elements_map.get(chosen_ref, {})
            is_input = elem_info.get("is_input", False) or "input" in chosen_desc.lower()

            if is_input:
                print(f"-> Typing '{target}' into search box and pressing Enter...")
                try:
                    locator.click()
                    locator.fill(target)
                    locator.press("Enter")
                    history.append(f"Typed '{target}' into {chosen_desc}")
                except Exception as ex:
                    print(f"Input error: {ex}")
                time.sleep(3.0)
            else:
                print(f"-> Clicking {chosen_ref} ({chosen_desc})...")
                try:
                    locator.click()
                    history.append(f"Clicked {chosen_desc}")
                except Exception as ex:
                    print(f"Click error: {ex}")
                time.sleep(3.0)

            step += 1

        print("\nTest completed successfully! Closing test tab in 3s...")
        time.sleep(3.0)
        page.close()

if __name__ == "__main__":
    run_test()
