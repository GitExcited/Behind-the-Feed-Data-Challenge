# Local News Visibility on Google

Goal: measure how visible small-town and independent local news is on Google's news surfaces in Canada:
- whether a town's own stories reach its Google News Local page;
- whether local outlets get the top spot when bigger outlets cover the same story.

How each dataset was collected: [`DATA_COLLECTION.md`](DATA_COLLECTION.md).

## Notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

**`main_findings.ipynb`** reproduces the main results from `data/`, one chart each (saved to `outputs/`):
- own election stories shown on Local pages, by town size;
- lead slot by outlet group on the front page;
- leads when an outlet published first;
- the lead ratio over time;
- the same measure in search;
- acclaimed mayors in towns with and without their own outlet.

**`collection/`** rebuilds the data from raw sources. Raw pages go to `cache/` (not committed); tables go to `outputs/`.

**`feature_engineering.ipynb`** builds the feature table for the `is_lead` ranking task from `data/google_news_front_page/`, following the role-separated plan (merit features fed to the score; fairness-constraint features audited but never fed to it; evaluation-only trend features) — saves `outputs/engineered_features_is_lead.csv` and `outputs/headline_embeddings_is_lead.npz`. Headline embeddings use a real multilingual sentence-transformer when installed, falling back automatically to a dependency-free vectorizer otherwise (see the notebook's own intro for a Windows install note).

| Notebook | What it does |
|---|---|
| `01_front_page.ipynb` | Lists and downloads the archived front pages, reads the stories, tags each outlet's payment group, and checks the result against `data/`. Set `LIMIT` to try a few pages first. |
| `02_local_pages.ipynb` | Reads each town's Local page today (40-town sample or all towns) and saves the story cards |
| `03_search_links.ipynb` | Builds the Google News and Google Search links for the 31 search-check stories |

`data/raw/` holds the inputs: CJC payment lists, Google's open-call list, hand-checked outlet tags and each town's Local page address.

## Data

### `data/google_news_front_page/`: Google News Canada front page, 2022–2026
Internet Archive snapshots of `news.google.com/home` (English `en-CA`, French `fr-CA`), one random snapshot per edition per 2-day period.

| File | Contents |
|---|---|
| `front_page_articles.csv` | One row per article shown on a snapshot, in every section |
| `snapshots.csv` | Every snapshot drawn. `purpose = main` is the analysis sample; `repeat` marks same-day pairs. `exclude = True` marks 5 snapshots the Archive served from a different date. |
| `outlets.csv` | Each outlet (website) with its payee, group, country and language |
| `cjc_payees.csv` | Canadian Journalism Collective payees, Year 1 and Year 2 amounts (CAD), parent business |

Key columns in `front_page_articles.csv`:
- `timestamp`: snapshot time, UTC.
- `section`: e.g. "Top stories" / "À la une".
- `card`: the story's position in the section, 0 = first.
- `slot`: position inside the story, 0 = the lead article.
- `cluster_size`: number of articles in the story.
- `outlet`: the article's website.
- `group`: `major` (paid; parent business got $2M+ in Year 2), `local_en` / `franco` / `diaspora` (paid, smaller), `cdn_unpaid`, `foreign` or `unknown`.

### `data/google_news_local_pages/`: Google News Local pages, Ontario municipalities, Oct 2026
The Google News Local page of each Ontario municipality with a contested 2026 council race, read logged out in English and French.

| File | Contents |
|---|---|
| `local_page_cards.csv` | Every story card on each town's page (Oct 10–11). `rank` = position on the page. Each card is tagged: about the town (`town_card`), about its race (`race_card`), about another town's race (`other_town_race`). |
| `local_page_cards_repeat_readings.csv` | Later readings (Oct 12–13) of a 40-town sample of the same pages, same columns without tags |
| `local_page_by_town.csv` | One row per town: cards shown, race stories shown, race stories that exist, and the outcome |

### `data/search_check/`: Google News search and Google Search, Oct 12, 2026
31 recent stories covered by both a smaller paid outlet and a major outlet, each searched in Google News and in Google Search's News tab and limited to the story's dates.

| File | Contents |
|---|---|
| `search_stories.csv` | The 31 stories: the two outlets' headlines and times, and the query used |
| `search_results.csv` | Results in order (`engine` = `gnews` or `gsearch`), with `size_group` and `on_story` (false = a result about a different story) |

### `data/ontario_towns/`: reference data for the 414 Ontario municipalities

| File | Contents |
|---|---|
| `towns.csv` | Population (Census 2021), French mother-tongue share, region, coordinates |
| `races_2026.csv` | 2026 municipal races: candidates, acclamations |
| `local_outlets.csv` | Local news outlets, their home town, owner and website |
| `local_stories.csv` | News stories naming each town (Oct 2026), with their source and whether they are about the town's race |
| `civic_outcomes.csv` | 2022 turnout and acclamations, 2026 contested races, whether the town has its own outlet |

## Sources
- **Front-page snapshots:** Internet Archive Wayback Machine.
- **Live pages:** Google News Local pages and search results, read through public pages.
- **Payments:** Canadian Journalism Collective, "payments to date" (Years 1–2); Google's 2024 open-call list.
- **Ontario:** Ontario Ministry of Municipal Affairs and Housing (municipalities, 2022 results); AMO 2026 candidate lists; Statistics Canada Census 2021.
- **Stories:** GDELT and local outlets' public feeds.

Headlines and links belong to their publishers. The search results reflect one signed-in account in Montreal.
