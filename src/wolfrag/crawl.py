"""Crawl the Werewolf: the Apocalypse subset of the White Wolf wiki.

Access is through the MediaWiki API rather than HTML scraping: it is the
sanctioned interface, it returns the original wikitext, and it lets us stay
polite with a fixed delay and an identifying User-Agent.

The crawler writes raw wikitext plus metadata to a JSONL snapshot. Cleaning and
chunking happen in preprocess.py, so the expensive network step runs once and
every later stage is reproducible offline from the committed snapshot.

Run:
    python -m wolfrag.crawl
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

import requests

from wolfrag import config


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT})
    return session


def _api(session: requests.Session, params: dict[str, Any]) -> dict[str, Any]:
    """Call the MediaWiki API with retries and a fixed politeness delay."""
    payload = dict(params)
    payload["format"] = "json"
    payload["formatversion"] = "2"

    last_error: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        try:
            response = session.get(
                config.WIKI_API, params=payload, timeout=45
            )
            response.raise_for_status()
            time.sleep(config.REQUEST_DELAY_SECONDS)
            return response.json()
        except Exception as exc:  # noqa: BLE001 - network layer, retry anything
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"API call failed after retries: {last_error}")


def _is_excluded_category(name: str) -> bool:
    lowered = name.lower()
    return any(bad in lowered for bad in config.EXCLUDED_CATEGORY_SUBSTRINGS)


def _is_excluded_title(title: str) -> bool:
    return any(bad in title for bad in config.EXCLUDED_TITLE_SUBSTRINGS)


def _category_members(
    session: requests.Session, category: str, member_type: str
) -> Iterator[dict[str, Any]]:
    """Yield members of a category, following continuation tokens."""
    cont: dict[str, str] = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": "500",
            "cmtype": member_type,
        }
        params.update(cont)
        data = _api(session, params)
        for member in data.get("query", {}).get("categorymembers", []):
            yield member
        if "continue" in data:
            cont = data["continue"]
        else:
            return


def collect_titles(session: requests.Session, verbose: bool = True) -> list[str]:
    """Walk the seed categories breadth first and collect article titles."""
    seen_categories: set[str] = set()
    titles: set[str] = set()

    frontier: list[tuple[str, int]] = [(c, 0) for c in config.SEED_CATEGORIES]

    while frontier:
        category, depth = frontier.pop(0)
        key = category.lower()
        if key in seen_categories:
            continue
        seen_categories.add(key)

        if _is_excluded_category(category):
            continue

        for page in _category_members(session, category, "page"):
            title = page.get("title", "")
            # Namespace 0 only: skip File:, Template:, Forum: and friends.
            if page.get("ns") != 0:
                continue
            if _is_excluded_title(title):
                continue
            titles.add(title)

        if depth < config.MAX_CATEGORY_DEPTH:
            for sub in _category_members(session, category, "subcat"):
                sub_name = sub.get("title", "").replace("Category:", "")
                if sub_name and not _is_excluded_category(sub_name):
                    frontier.append((sub_name, depth + 1))

        if verbose:
            print(
                f"  visited category {category!r} "
                f"(depth {depth}); running total {len(titles)} titles",
                flush=True,
            )

    return sorted(titles)


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_pages(
    session: requests.Session, titles: list[str], verbose: bool = True
) -> list[dict[str, Any]]:
    """Fetch wikitext and metadata for every title, in deterministic batches."""
    records: list[dict[str, Any]] = []
    total_batches = (len(titles) + config.BATCH_SIZE - 1) // config.BATCH_SIZE

    for index, batch in enumerate(_batches(titles, config.BATCH_SIZE), start=1):
        data = _api(
            session,
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "revisions|categories|info",
                "rvprop": "content|timestamp|ids",
                "rvslots": "main",
                "cllimit": "max",
                "inprop": "url",
            },
        )
        for page in data.get("query", {}).get("pages", []):
            if page.get("missing") or "revisions" not in page:
                continue
            revision = page["revisions"][0]
            wikitext = revision.get("slots", {}).get("main", {}).get("content", "")
            if not wikitext:
                continue
            records.append(
                {
                    "pageid": page.get("pageid"),
                    "title": page.get("title", ""),
                    "url": page.get("fullurl")
                    or config.WIKI_BASE + page.get("title", "").replace(" ", "_"),
                    "categories": [
                        c.get("title", "").replace("Category:", "")
                        for c in page.get("categories", [])
                    ],
                    "revision_id": revision.get("revid"),
                    "revision_timestamp": revision.get("timestamp"),
                    "wikitext": wikitext,
                }
            )
        if verbose:
            print(
                f"  fetched batch {index}/{total_batches} "
                f"({len(records)} pages so far)",
                flush=True,
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit-titles",
        type=int,
        default=None,
        help="Fetch only the first N titles. Smoke testing only.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    config.set_seeds()
    session = _session()

    print("Collecting article titles from seed categories ...", flush=True)
    titles = collect_titles(session, verbose=verbose)
    print(f"Collected {len(titles)} unique candidate titles.", flush=True)

    if args.limit_titles:
        titles = titles[: args.limit_titles]
        print(f"Smoke test: limiting to {len(titles)} titles.", flush=True)

    print("Fetching page content ...", flush=True)
    records = fetch_pages(session, titles, verbose=verbose)
    print(f"Fetched {len(records)} pages with content.", flush=True)

    crawled_at = datetime.now(timezone.utc).isoformat()
    for record in records:
        record["crawled_at"] = crawled_at

    records.sort(key=lambda r: r["title"])

    config.RAW_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.RAW_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source": "White Wolf Wiki (whitewolf.fandom.com)",
        "license": "CC BY-SA 3.0",
        "game_line": "Werewolf: the Apocalypse",
        "crawled_at": crawled_at,
        "api": config.WIKI_API,
        "seed_categories": config.SEED_CATEGORIES,
        "max_category_depth": config.MAX_CATEGORY_DEPTH,
        "excluded_category_substrings": config.EXCLUDED_CATEGORY_SUBSTRINGS,
        "excluded_title_substrings": config.EXCLUDED_TITLE_SUBSTRINGS,
        "candidate_titles": len(titles),
        "pages_fetched": len(records),
    }
    config.CRAWL_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {config.RAW_CORPUS_PATH}", flush=True)
    print(f"Wrote {config.CRAWL_MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
