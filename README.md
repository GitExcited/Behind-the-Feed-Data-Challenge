# Local News Visibility on Google

Goal: measure how visible small-town and independent local news is on Google's news surfaces in Canada:
- whether a town's own stories reach its Google News Local page;
- whether local outlets get the top spot when bigger outlets cover the same story.

How each dataset was collected: [`DATA_COLLECTION.md`](DATA_COLLECTION.md).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Python >= 3.10 (we use 3.12)
pip install -r requirements.txt
jupyter notebook
```

`sentence-transformers` must be **6.2 or newer**: our saved models were written with 6.x and an older version cannot
load them (`No module named sentence_transformers.base`). Training uses a CUDA GPU or Apple silicon (MPS) if there is
one; latency is always measured on CPU.

**Model weights are not in git.** Put them in `models/` (a copy or a symlink, e.g. `ln -s ~/Downloads/minilm_robust models/minilm_robust`):

| Folder | Model | Made in |
|---|---|---|
| `models/minilm_finetuned` | **v1**: fine-tuned MiniLM bi-encoder | Milestone 2 (`milestone2_training.ipynb`) |
| `models/minilm_robust` | **v2**: v1 recipe + noise augmentation; deployed with the input sanitizer, start model of Milestone 4 | Milestone 3 (`milestone3_robustness.ipynb`) |
| `models/exp/cross_encoder` | fine-tuned cross-encoder (comparison only) | Milestone 2 (`milestone2_experiments.ipynb`) |

The pretrained `paraphrase-multilingual-MiniLM-L12-v2` downloads into `models/` on first use. Other locations can be
set with `M3_V1`, `M4_START` (or `--start-model`) and `CL_MODELS` (where continual-learning versions are saved).

## Notebooks

| Milestone | Notebook | What it does | Outputs |
|---|---|---|---|
| 1 | `main_findings.ipynb`, `data_report.ipynb` | Main visibility findings and the data report | `outputs/milestone1/` |
| 2 | `milestone2_training.ipynb` | Fine-tunes the same-story model (v1) | `outputs/milestone2/` |
| 2 | `milestone2_experiments.ipynb`, `milestone2_label_noise.ipynb` | Experiments after v1 (augmentations, cross-encoder) and the label-noise study | `outputs/milestone2/experiments/`, `label_noise/` |
| 2 | `milestone2_efficiency.ipynb` | Training and inference efficiency (`src/m2_efficiency.py`) | `outputs/milestone2/efficiency/` |
| 3 | `milestone3_risk.ipynb` | Part 1.3 stress tests (`src/m3_stress.py`) | `outputs/milestone3/stress/` |
| 3 | `milestone3_robustness.ipynb` | Section 2: noise and attacks, v2, sanitizer, calibration (`src/robustness.py`) | `outputs/milestone3/` |
| 3 | `milestone3_monitoring.ipynb` | Section 3: monitoring dashboard and alerts (`src/m3_monitoring.py`) | `outputs/milestone3/` |
| 3 | `milestone3_adaptation.ipynb` | Section 4.3: drift, prior shift and domain shift adaptation (`src/adaptation.py`) | `outputs/milestone3/adaptation/` |
| 4 | `milestone4_continual.ipynb` | Part 1.4: simulated drift + continual update experiment (`src/m4_experiment.py`) | `outputs/milestone4/experiment/` |

**`main_findings.ipynb`** reproduces the main results from `data/`, one chart each (saved to `outputs/`):
- own election stories shown on Local pages, by town size;
- lead slot by outlet group on the front page;
- leads when an outlet published first;
- the lead ratio over time;
- the same measure in search;
- acclaimed mayors in towns with and without their own outlet.

**`milestone3_monitoring.ipynb`** (Milestone 3, Section 3) simulates a production monitoring
setup for the deployed bi-encoder using `data/pairs/val.csv` as the reference distribution and
`data/pairs/test.csv`, replayed in weekly windows, as the simulated post-deployment stream.
Computes data-quality drift (PSI/Jensen-Shannon/Wasserstein on headline length, PSI on outlet-group
mix), rolling model-quality metrics (F1/ROC-AUC/PR-AUC against real labels), and alerting rules,
saving a dashboard and the underlying tables to `outputs/milestone3/`. Shared drift/monitoring
functions are in `src/m3_monitoring.py`.

### Continual learning (Milestone 4): `src/continual.py`

Updates the deployed bi-encoder as new months of labelled pairs arrive, without forgetting older
data and without deploying a model that got worse. One update cycle:

1. **Trigger check** (Milestone 3 rules): T1 data drift or T2 F1 drop → fine-tune; T3 change in the
   share of same-story pairs → only recalibrate the threshold (no training); nothing fired → no update.
2. **Fine-tune with replay**: the month's pairs (minus a 20% holdout of front pages) + the same number
   of old pairs from a 4,000-pair replay buffer (balanced by edition and label). Same training recipe as Milestone 2.
3. **Accept/reject gate**: the new model is deployed only if it is at least as good on the holdout and loses
   at most 0.01 F1 on the old validation data. Rejected models are logged and their weights deleted.
4. **Versioning**: every model (v2 → v2.1 → v2.2 …) is recorded in `registry.json` with its parent,
   threshold, calibration, data hashes, gate result and status; `Registry.rollback()` restores the previous one.

```bash
python -m src.continual --policy triggered      # real May–Oct 2026 stream, update only when a trigger fires
python -m src.continual --policy always         # update every month   (--policy never = no-update baseline)
python -m src.continual --mode finetune         # no replay            (--mode full = full retrain from the base model)
python -m src.continual --drift typos           # simulated drift: typo rate grows month by month
python -m src.continual --smoke                 # quick check that everything runs (CPU, a few minutes)
```

The starting model is the deployed Milestone 3 model, **v2 + input sanitizer**, read from `models/minilm_robust`
(weights are not in git; use `--start-model <folder>` or `M4_START=<folder>` if yours is elsewhere). Updates are named
v2.1, v2.2, … To start from Milestone 2's v1 instead: `--start-model models/minilm_finetuned --start-name v1 --no-sanitizer`. Model versions are saved in `models/cl/<run>/` (not committed). Results go to `outputs/milestone4/continual/<run>/`:
`stream_log.csv` (before/after F1 per month, triggers, gate decision, update time, peak RAM/VRAM, model size),
`stream.png` (metric trajectories), `registry.json` and the replay buffer. Settings are in `CL_CFG` at the top of the file.
For human-in-the-loop / active learning, pass the newly labelled pairs to `update()` as the new batch.

### Continual learning experiment (Milestone 4, Part 1.4): `src/m4_experiment.py`

Replays May–Oct 2026 one month at a time from v2 + sanitizer (each month is scored before any update sees it) and compares
update strategies: no update, fine-tune without replay, replay + gate (every month or only when a trigger fires) and full
retrain. Runs that train use 3 seeds (6450–6452), reported as mean ± std.

| Part | What | Question |
|---|---|---|
| A | Real months, no added drift | Do we need updates on real data? |
| B | Simulated drift: typo rate grows 0 → 30% of letters over the six months | Does replay + gate recover accuracy, and at what cost? |
| C | One update on May–Jul labels with 10–40% wrong labels (random, or missed same-story links) | Does the gate stop updates trained on bad human labels? |
| D | CPU page latency and model size, v2 vs updated models | Do updates change serving cost? |

```bash
python -m src.m4_experiment                 # everything, about 30 min on an Apple M4 Max laptop
python -m src.m4_experiment --parts B C     # only some parts
python -m src.m4_experiment --smoke         # quick check that everything runs (about 2 min)
python -m src.m4_experiment --report        # rebuild tables and figures from saved runs, no training
```

Finished runs are cached, so an interrupted job continues where it stopped. Results go to `outputs/milestone4/experiment/`:
`summary.csv` (one row per strategy), `per_month.csv`, `stream_logs_all.csv` (every month of every run: metrics, triggers,
gate result, update time, peak RAM / GPU memory), `bad_labels.csv` and `bad_labels_summary.csv`, `latency_size.csv`,
`runs/<run>/` (log, `registry.json` and settings of each run) and `figures/` (metric trajectories and per-update F1 before/after,
update time and memory for A and B; bad labels for C). Model versions are written to `models/cl/experiment/` (not committed); only the two final models used for the
latency check are kept. `milestone4_continual.ipynb` shows the results (set `RUN = True` in it to run everything again).

**`collection/`** rebuilds the data from raw sources. Raw pages go to `cache/` (not committed); tables go to `outputs/`.

| Notebook | What it does |
|---|---|
| `01_front_page.ipynb` | Lists and downloads the archived front pages, reads the stories, tags each outlet's payment group, and checks the result against `data/`. Set `LIMIT` to try a few pages first. |
| `02_local_pages.ipynb` | Reads each town's Local page today (40-town sample or all towns) and saves the story cards |
| `03_search_links.ipynb` | Builds the Google News and Google Search links for the 31 search-check stories |

`data/raw/` holds the inputs: CJC payment lists, Google's open-call list, hand-checked outlet tags and each town's Local page address.

## Code Structure
```
Behind-the-Feed-Data-Challenge/
├── collection/                  # Milestone 1 collection notebooks
├── data/
│   ├── google_news_front_page/
│   │   ├── front_page_articles.csv
│   │   ├── snapshots.csv
│   │   └── ...                  # other Milestone 1 files
│   ├── ...                      # other existing data folders
│   └── pairs/                   # Pairs created by train before 2026, val Jan–Apr, test May onward
│       ├── train.csv            # 21,963 pairs
│       ├── val.csv              # 2,850 pairs
│       └── test.csv             # 2,457 pairs
├── models/                      # model weights (not in git, see Setup); continual-learning versions in models/cl/
├── outputs/
│   ├── milestone1/              # charts + local_missingness.csv
│   ├── milestone2/              # baselines, training, experiments, label noise, efficiency
│   ├── milestone3/              # robustness, stress/, monitoring dashboard, adaptation/
│   └── milestone4/experiment/   # continual learning experiment (tables, runs/, figures/)
├── report/
│   ├── milestone1/              # LaTeX source and figures
│   └── milestone2/              # LaTeX source, figure and compiled PDF
├── src/
│   ├── build_pairs.py           # M2: filter cards, build pairs, split, save
│   ├── translate_headlines.py   # M2: EN <-> FR translations for augmentation
│   ├── baselines.py             # M2: TF-IDF and the pretrained MiniLM
│   ├── embeddings.py            # M2: embed_headlines(), similarity()
│   ├── m2_common.py             # M2: shared data, metrics, scoring and training
│   ├── m2_efficiency.py         # M2: latency, throughput, memory, FLOPs, ONNX and int8 runs
│   ├── m3_stress.py             # M3 Part 1.3: stress-test corruptions and scoring
│   ├── robustness.py            # M3 Section 2: corruptions, attacks, v2 training, sanitizer, calibration
│   ├── m3_monitoring.py         # M3 Section 3: drift and monitoring functions
│   ├── adaptation.py            # M3 Section 4.3: drift statistics, prior-shift correction, incremental fine-tuning
│   ├── continual.py             # M4 Part 1: continual learning (replay buffer, triggers, gate, model registry)
│   └── m4_experiment.py         # M4 Part 1.4: drift + continual update experiment
├── DATA_COLLECTION.md
├── README.md
├── data_report.ipynb
├── main_findings.ipynb
├── milestone2_*.ipynb           # training, experiments, label noise, efficiency
├── milestone3_*.ipynb           # risk, robustness, monitoring, adaptation
├── milestone4_continual.ipynb   # continual learning experiment
└── requirements.txt
```

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
