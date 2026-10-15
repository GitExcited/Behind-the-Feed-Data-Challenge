# How the data was collected

How each dataset in `data/` was built, as a starting point for replicating or extending it. The collection scripts are not in this repository yet (see the end).

## National: Google News front page, 2022–2026 → `data/google_news_front_page/`

1. **List the snapshots.** The Internet Archive's CDX API lists every saved copy of a page:
   `https://web.archive.org/cdx/search/cdx?url=news.google.com/home?hl=en-CA%26gl=CA%26ceid=CA:en&output=json&filter=statuscode:200`
   (and the same with `fr-CA` / `CA:fr`). May 2022 to Oct 2026: about 155,000 English and 64,000 French snapshots.
2. **Sample.** One random snapshot per edition per 2-day period (random seed 6450), plus a few same-day pairs to check stability: 1,480 drawn, 1,474 downloaded.
3. **Download** each snapshot unmodified: `https://web.archive.org/web/<timestamp>id_/<original url>`. One request at a time, 5–8 s apart. The Archive often answers 503 under load: wait and retry.
4. **Check the date served.** The Archive sometimes redirects to a nearby snapshot. Compare the timestamp in the final URL with the one requested; 5 snapshots were more than 3 h off and are marked `exclude`.
5. **Parse.** Each page embeds its stories as JSON in a `AF_initDataCallback(...)` block with the key `gbres`. It contains sections ("Top stories", ...), then story cards, then a lead article and related articles. Each article has a headline, link, publisher, publication time and story ID.
6. **Tag outlets.**
   - Match each article's website to Google's 2024 open-call list (PDF on blog.google: legal name and website).
   - Then match it to the Canadian Journalism Collective's "payments to date" lists (cjc-ccj.ca, Years 1 and 2) by legal name, using fuzzy matching.
   - Add up payees of the same parent company (Québecor, Corus, Glacier).
   - `major` = the parent got $2M+ in Year 2.
   - About 250 outlets were checked by hand.

## National: search check, Oct 12, 2026 → `data/search_check/`

1. **Pick contested stories (31):** the same event covered by both a smaller paid outlet and a major one, confirmed by hand.
   - Pairs from the Local-page cards (Sept 28–Oct 12);
   - Archive front-page clusters from Sept–Oct 2026 that contain a smaller outlet;
   - One Full Coverage page.
2. **Query** = the words both headlines share (no quotes), limited to the story's dates:
   - Google News: `news.google.com/search?q=<query> after:<date> before:<date>&hl=en-CA&gl=CA&ceid=CA:en`
   - Google Search, News tab: `google.com/search?q=<query>&tbm=nws&tbs=cdr:1,cd_min:<M/D/YYYY>,cd_max:<M/D/YYYY>`
3. **Record** the top 10 results (outlet, headline, time) from a signed-in Chrome in Montreal. Results about a different story were marked off-story by hand.

## Local: Google News Local pages → `data/google_news_local_pages/`

1. **Towns.** The 414 Ontario municipalities (Ministry of Municipal Affairs and Housing list) with Census 2021 population and French mother-tongue share (Statistics Canada). The 390 with a contested 2026 race come from AMO's candidate lists.
2. **Page address.** A town's Local page is `news.google.com/topics/<id>?hl=en-CA&gl=CA&ceid=CA:en`, where the ID encodes the town's Google Knowledge Graph ID. That ID comes from Wikidata and was checked against the town's StatCan code: 391 of 395 matched.
3. **Read** each page logged out, in English and French, about 30 s apart. A full pass of 780 pages takes about 7 hours.
   - Full pass: Oct 10–11.
   - Repeat readings of a 40-town sample: Oct 12 (twice), Oct 13 and Oct 15.
   - A page can't be read for a past date, so each day not read is lost.
4. **Parse** each story card: rank on the page, headline, outlet, time, link.
5. **Tag** each card as about the town (names it or one of its communities), about its race (election words), or about another town's race. All race cards were checked by hand.

## Local: what news exists → `data/ontario_towns/`

- **Stories naming each town:**
  - GDELT GKG 2.1 bulk files for Oct 1–9 (about 10.5 GB, streamed and filtered to Ontario);
  - about 160 local outlets' own RSS feeds and sitemaps (Aug 21–Oct 9);
  - the GDELT DOC API for the 40-town sample.

  Each story is matched to towns by name. GDELT misses some outlets (e.g. Village Media sites), hence the outlet feeds.
- **Local outlets:** rosters from OCNA, Village Media, Postmedia, Réseau.Presse, the CJC list, the Local News Map and Wikipedia, with homepages checked.
- **Civic outcomes:** 2022 municipal results (turnout, acclamations) from the Ministry's official file; 2026 acclamations from AMO.

## Notebooks

- `collection/01_front_page.ipynb`: front page, steps 1–6 above.
- `collection/02_local_pages.ipynb`: Local pages, steps 3–4.
- `collection/03_search_links.ipynb`: the search links.

The story counts per town (GDELT and outlet feeds) and the hand-checked tags are provided as finished tables in `data/`.
