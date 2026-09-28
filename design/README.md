# Site redesign

The design for the next version of alinaderiparizi.com. This folder is the source of truth for
how the site should look and behave; the Jekyll layouts and CSS are built to match it.

- **[index.html](index.html)**: a clickable mockup of every page type (Home, Blog, Post, Projects,
  About), in light and dark. Open it straight from disk; it needs no build.
- **This file**: why the current site gets in the way, the principles, and the spec per page.

`design/` is excluded from the Jekyll build (`_config.yml`), so none of it is published.

## 1. What's wrong today

The current site looks like a sci-fi control panel. It's distinctive, but it works against the
two things visitors come for: **reading a post** and **finding out who Ali is**.

| Problem | Where | Why it hurts |
| --- | --- | --- |
| Jargon labels | "SYSTEM_STATUS NOMINAL // 100%", "INITIATE SEARCH_QUERIES…", "Blog deployments", "Article dispatch", "Feed status live", "Library modules", "Systems architecture" | A visitor has to decode the UI before using it. None of it carries information. |
| ALL-CAPS titles | page and post titles, card titles | Caps are ~10–15% slower to read, and long post titles become walls of shouting. |
| Home is a dashboard | GitHub metrics, "Publication impact monitor", typed-text hero | The first screen shows loading dashes and "Loading publications…" instead of who you are and what you wrote. |
| Live widgets that fail in the browser | GitHub metrics, ORCID fetch on Home, Projects and Research | On a slow or filtered network they stay empty (`--`, "Syncing…"). Research is almost blank until ORCID answers. |
| Hard-to-read posts | post layout | Lines run ~150 characters, body text is small grey on near-black, and the animated background sits behind the text. |
| Animated WebGL background | every page | Costs battery and CPU on every page for decoration, and lowers text contrast. |
| Dark only | `<html class="dark">` | Ignores the visitor's system setting; long reads in daylight are tiring. |
| Weight and dependencies | Bootstrap 4 (CDN), Font Awesome kit, Material Symbols, Bootstrap Icons, Boxicons, 10 vendor libs (12 MB), ~30 unused font zips (6 MB), MathJax on every page | Slow first load, four icon sets for one site, and third-party requests on every page. |
| Search leaves the site | top bar | Opens Google in a new tab. |
| Titles | `<title>Home</title>` | Browser tabs and search results just say "Home", "Blog". |
| Hidden content | Notes collection | Built but not reachable from the navigation. |

## 2. Principles

1. **Content first.** Every page leads with the thing it's named after. No dashboards above the fold.
2. **Plain words.** Labels say what's there: "Blog", "Recent posts", "Search". No mock-terminal copy.
3. **Readable by default.** A 68-character reading column, 18px body text, 1.7 line height,
   sentence-case headings, real contrast (WCAG AA at minimum).
4. **Light and dark, following the system**, with a toggle that remembers the choice.
5. **Numbers are always there.** GitHub and academic stats are fetched at *build* time and baked
   into the page, and the site rebuilds daily, so a visitor never sees `--` or "Loading…" and no
   page depends on a third-party API answering in their browser (see *Stats* below).
6. **Small.** One stylesheet; Inter for text, Anton for display, JetBrains Mono for code (all self-hosted);
   one icon set (inline SVG), no CSS framework. MathJax and Prism load only on posts that need them.
7. **Standard patterns.** Top navigation, breadcrumbs back from a post, prev/next
   at the end. Nothing a visitor has to learn.

## 3. Information architecture

```text
Home ─┬─ Blog ──────── post
      ├─ Projects ──── project post
      ├─ Research          (publications, from _data/publications.yml)
      ├─ Library ───── book review
      └─ About             (bio, experience, education, CV download)
```

- **Top nav**: name on the left; Home · Blog · Projects · Research · Library · About on the right; then
  search and the theme toggle. The current page's item is highlighted (Home included). On phones
  it collapses into a menu button.
- **No footer.** Every page ends on the **closing quote**: Orwell's *"Freedom is the freedom to say
  that two plus two make four."* in Anton at 64px (38px on phones), in the gradient, with a faint
  oversized quotation mark behind it and the attribution in mono small caps. Links (GitHub, email,
  CV) live in the nav and on About; RSS is advertised in `<head>` for feed readers.
- Notes join the nav under Blog once there is more than one.

## 4. Design tokens

The look is **DeFi, kept professional, with an edge**: a near-black canvas with violet and
crimson glows, glass cards, sharp corners, and one violet → magenta → crimson gradient as the
signature. Dark is the primary theme; light is a clean, equal alternative. The gradient is for
*accents only* (brand, primary button, stat numbers, a 3px top line on panels, active filter):
body text is always solid, and the reading column has no glass or glow behind it.

### Voice

Ali is serious and direct, a fighter in how he works, and the site says so.

- **Type is heavy.** Headings at weight 800 with tight tracking; the name in the hero at 56px.
  Section labels in uppercase mono with wide tracking; buttons uppercase and bold.
- **Copy is short and declarative, and never self-praise.** No "Hi, I'm…", no "I build / I ship /
  I do" sentences. The work speaks; the hero carries an **epigraph** instead: a sharp line from a
  great engineer or thinker, set as a code comment (`// Talk is cheap. Show me the code.` —
  Linus Torvalds) in JetBrains Mono, with the attribution under it in small caps. The quote can be
  swapped in `_config.yml` without touching templates. Buttons say *Read the work*, *Contact*,
  *Download CV*.
- **Crimson is the edge**, used for kickers, deltas and the end of the gradient, never for body
  text or large fills.
- **Still readable.** Long posts keep a calm, solid 18px column; the intensity lives in the frame,
  not the prose.

### Portrait

The current photo (suit, white studio background) clashes with a dark poster design, so it
becomes a **booking photo**. `design/tools/cutout.py` cuts Ali out of the plain studio wall
(flood fill from the frame edges, no segmentation model; pure-white collar excluded so the fill
can't leak through it; mask feathered) into `assets/images/me-cutout.png`. The portrait then stacks:
violet → crimson wall, height chart, a vignette that fades the wall into the corners, and Ali in
black and white **in front of** the chart with a flash shadow on the wall; grain and a 3px gradient
bar on top. The chart never crosses the face. Square crop; hovering shows the photo in colour.

**Mugshot.** Behind the photo sits a police-lineup **height chart**: drawn to Ali's real height,
**191 cm**. The chart is scaled to the photo: the hair top sits 4.4% from the top of the frame and
the chin at 56%, and an average head (≈23.5 cm) gives ~0.455 cm per percent, so the frame spans
≈193 cm at the top to ≈148 cm at the bottom. Full lines and labels every 5 cm, edge ticks every
1 cm, and a crimson line labelled **191** at the top of his head. One inline SVG over the gradient;
if the photo changes, re-measure hair and chin and regenerate it. On About, a **booking placard** hangs under it:
`NADERI, A.` in Anton, `TEHRAN · 2026`, and the booking number `#GPL-0003` in crimson (the joke:
booked under the GPL, version 3).

Better still, a new photo shot for this design: black-and-white or low-key, dark background, side
or hard light, casual (a dark tee or jacket), looking at the camera or three-quarter, against a
plain wall so `cutout.py` still works.

### Freedom, humanity, truth

Ali follows GNU and the free software movement, and cares about freedom, humanity and truth. The
site *shows* those values instead of claiming them:

- **The site is free software.** Code under **GPL-3.0**, writing under **CC BY-SA 4.0**, with a
  `LICENSE` file in the repo (there is none today). One line at the end of About says so.
- **No trackers, no ads, no cookies**, and no third-party requests at page load (fonts
  self-hosted, search client-side, stats baked at build time), stated in the same About line.
- **The closing quote** (see *Information architecture*) carries freedom and truth at the end of
  every page.
- **Epigraphs lead with freedom.** The hero quote list in `_config.yml` starts with Stallman
  (*"Free software is a matter of liberty, not price."*), followed by Torvalds, Dijkstra, Kay and
  Knuth. The first is rendered at build time; clicking it cycles through the rest.

### Humor

Ali is funny, and the humor lives in the corners, never in the way of the work:

- **404: "This page went solo."** *It left the band and never came back. The rest of the tracklist
  is still here.* Button: *Back to the setlist*.
- **The quote cycles on click**, a small reward for the curious.
- **A note in the browser console** for anyone reading the source: `// Reading the source? Good.
  It's free software. Take it, change it, share it.` with the repo link.

### Athletic

About gets an **Off the keyboard** section: the sports Ali trains, as short lines with a stat
each where there is one (e.g. distance, years, grade). Content to come from Ali.

### Rock and metal

Ali loves rock and metal, and the frame of the site borrows from album art and gig posters,
tastefully: no skulls, flames or novelty fonts.

- **Poster type.** Page titles, post titles, the hero name and stat numbers are set in **Anton**
  (SIL OFL, self-hosted in `assets/fonts/`), uppercase, tight leading. The hero name runs 104px
  (64px on phones) in the gradient. Body, UI and code stay in Inter and JetBrains Mono.
- **Tracklist numbering.** Section titles read `01 — AT A GLANCE`, `02 — RECENT POSTS`; post rows
  lead with `01 / 28 SEPT 2026`. Numbers in crimson, via CSS counters (no markup).
- **Brushed steel.** In dark mode stat numbers are filled with a chrome gradient (`--steel`); in
  light mode they keep the brand gradient.
- **Grain.** A fixed SVG noise layer at 6% opacity over the page gives print-poster grit. It's one
  static data URI, `pointer-events: none`, no animation.
- **The slash.** A 3px gradient bar with an angled end closes the hero, like a band logo's
  underline.

All colors are CSS custom properties on `:root`, redefined for dark mode. Nothing outside the token
block uses a raw color.

| Token | Light | Dark | Use |
| --- | --- | --- | --- |
| `--bg` | `#f7f8fc` | `#040409` | page |
| `--surface` | `#ffffff` | `#0b0c16` | code blocks, tiles |
| `--glass` | `rgba(255,255,255,.72)` | `rgba(22,26,52,.58)` | cards, panels, buttons (with `backdrop-filter: blur`) |
| `--border` | `#e3e6f3` | `rgba(148,163,255,.14)` | dividers, card borders |
| `--text` | `#0d1030` | `#f5f6ff` | body |
| `--muted` | `#555d80` | `#a7abc7` | meta, captions |
| `--accent` | `#6d28d9` | `#a78bfa` | links, focus, active nav |
| `--accent-2` | `#dc2626` | `#f87171` | the edge: kickers, deltas, TOC marker |
| `--accent-soft` | `#f1ebff` | `rgba(139,92,246,.16)` | tag chips, callouts |
| `--on-accent` | `#ffffff` | `#05060d` | text on the gradient button |
| `--grad` | `#6d28d9 → #c026d3 → #dc2626` | `#8b5cf6 → #e879f9 → #ef4444` | the signature gradient, 115° |
| `--glow-a`, `--glow-b` | 10% / 8% | 28% / 16% | two static radial glows behind the page |

The glows are fixed CSS gradients, not the current WebGL animation: same atmosphere, no CPU cost.

**DeFi details, all small:** numbers (stats, dates, meta labels) in JetBrains Mono, uppercase with
slight letter-spacing, like a dashboard; stat values in gradient text; square chips and sharp corners
throughout; a gradient dot before each section title; cards lift with a
soft violet glow on hover.

| Type | Size / line height | Weight |
| --- | --- | --- |
| Page title | 40 / 1.15 (32 on phones) | 700 |
| Post title | 36 / 1.2 (28 on phones) | 700 |
| H2 | 26 / 1.3 | 650 |
| H3 | 20 / 1.4 | 600 |
| Body | 18 / 1.7 | 400 |
| Meta, captions | 14 / 1.5 | 400–500 |
| Code | 15 / 1.6, JetBrains Mono | 400 |

Spacing is a 4px scale (`4 8 12 16 24 32 48 64 96`). **Edges are sharp everywhere**: `border-radius: 0` on cards, panels, buttons, tiles, chips,
thumbnails, the profile photo and code blocks. Square corners read as precise and technical, and
let the gradient and glass carry the DeFi feel on their own. Content widths: `68ch` for reading, `1120px` for grids.

## 5. Components

- **Post row**: thumbnail (the post's `img`, 16:9, 200px wide; 96px square-cropped on phones) ·
  date · title · one-line brief · tags. Used on Home and Blog. The thumbnail sits on the left so
  titles still line up and a long list stays scannable. A post without `img` gets a tinted
  placeholder with its first tag, so rows never jump.
- **Stat tile**: big number, label under it, optional small delta ("+12 this year"). Tiles sit in a
  row of 4 (2 on phones) inside a bordered panel with a title and a "source" link.
- **Contribution graph**: GitHub's last-year contribution grid as 53×7 small squares in 5 accent
  steps, with month labels; scrolls horizontally on phones instead of shrinking.
- **Project card**: image (16:9, `object-fit: cover`), title, brief, tech chips, links to post and repo.
- **Book card**: cover (2:3), title, author, one-line takeaway.
- **Publication row**: authors (Ali in bold), title, venue and year, links (PDF · DOI · code).
- **Tag chip**: square, `--accent-soft` background, sentence case, links to the tag's filtered list.
- **Table** in posts: **centred** in the reading column at its natural width (not stretched), a
  2px gradient line on top, headers in mono small caps, 1px row dividers, a soft highlight on the
  hovered row. Column alignment from Markdown (`---:`) is kept. A table wider than the column
  scrolls inside itself.
- **Callout** in posts: note / warning, `--accent-soft` background, gradient left border.
- **Code block, with syntax highlighting.** Jekyll's built-in **Rouge** already tokenises every
  fenced block at build time (` ```python ` → `<span class="k">` …), so highlighting needs **no
  JavaScript highlighter**: `Prism.js` is removed and the site only ships a Rouge colour theme,
  built from the palette via `--hl-*` tokens (light / dark):

  | Token | Rouge classes | Light | Dark |
  | --- | --- | --- | --- |
  | keyword | `k kd kn kr kc ow` (bold) | `#6d28d9` | `#a78bfa` |
  | string | `s s1 s2 sa sd …` | `#0f766e` | `#5eead4` |
  | number | `m mi mf mh …` | `#be123c` | `#fda4af` |
  | function | `nf fm` | `#a21caf` | `#e879f9` |
  | class, type | `nc nn kt ne` | `#86198f` | `#f0abfc` |
  | builtin, constant | `nb bp no nv` | `#1d4ed8` | `#93c5fd` |
  | decorator, interpolation | `nd na si se cp` | `#db2777` | `#f472b6` |
  | comment | `c c1 cm …` (italic) | `#6b7280` | `#6b7394` |
  | operator, punctuation | `o p` | `#475069` | `#b8bdd8` |

  The block: `--surface` background, 1px border, a 2px gradient line on top, the language in mono
  small caps top-left (from the `language-*` class, CSS only), and a **Copy** button top-right (a
  few lines of JS; the only script a post needs besides MathJax on math posts). Long lines scroll
  inside the block, never the page.

## 6. Pages

**Home.** The intro (photo, name, role kicker, and the epigraph), three buttons (Read the work · Contact · Download CV), then *At a glance* (the stats panels, below)
right under the intro, then *Recent posts* (5 rows), *Featured projects* (3 cards) and *Selected
publications* (2–3 rows).

**At a glance** is two panels, stacked full width (the year-long contribution graph is ~690px, too
wide for half the 1120px column):

- **Open source**: public repos · stars · contributions in the last year · followers, the
  contribution graph, and "Most active in: Python, Rust, TypeScript" from repo languages. Links to
  the GitHub profile.
- **Academic**: publications · citations · h-index · peer reviews, then the latest publication as
  one line. Links to Google Scholar and ORCID.

**Blog.** Title, a tag filter row (All · AI · Embedded · Network · Math …), and posts grouped by year
as post rows. Filtering is client-side on data attributes; with JavaScript off, all posts show.

**Post.** Breadcrumb (Blog / Projects), title, meta line (date · reading time · tags), optional hero
image, then the 68ch body.

**On this page** (posts with 3+ headings), always visible and always showing where you are:

- **Wide screens (> 1100px):** a 240px rail in the right margin, `position: sticky` 88px from the
  top, so it stays in view for the whole post; it scrolls on its own if a post has many headings.
  A gradient **progress line** under its title fills as you read.
- **Narrow screens:** a bar pinned under the nav: `ON THIS PAGE` + the current section's name, with
  the progress line along its bottom edge. Tapping it drops down the full list; picking a section
  jumps there and closes it.
- **Current section highlighted** in both: the last heading scrolled past the top quarter of the
  viewport is the current one; its link turns bold and bright with a crimson left border and a
  soft gradient wash. One passive scroll listener, throttled with `requestAnimationFrame`.
- Links jump with `scrollIntoView` and `scroll-margin-top` on headings (96px, 124px on phones), so
  a heading never lands under the nav or the bar; smooth unless `prefers-reduced-motion`. Post
  images load eagerly so the page doesn't grow mid-jump.
- H3s appear indented under their H2. At the end: GitHub link if the post has one, prev/next, and back to the list.

**Projects.** The Open source panel on top, then the grid of project cards (3 / 2 / 1 columns),
newest first.

**Research.** The Academic panel on top, then publications from `_data/publications.yml`,
grouped by year, then Service (reviewing, judging).

**Library.** Grid of book cards.

**About.** Portrait and a one-line bio: *"Graduated from IUST, one of Iran's toughest universities."*
No current role or employer there (the CV timeline below carries the history), then Experience and Education as simple timelines, skills as
chips, and a prominent "Download CV" button. Contact links in one row.

## 7. Stats: where the numbers come from

Today the browser fetches GitHub and ORCID on every visit, which is why the panels show `--` on a
slow or filtered network. Instead, the numbers are fetched once per build and written to
`_data/stats.yml`, which the templates read like any other data:

| Number | Source | Fetched by |
| --- | --- | --- |
| repos, stars, followers, languages | GitHub REST API (`/users/mralinp`, `/users/mralinp/repos`) | build script |
| contributions + graph | GitHub GraphQL `contributionsCollection` (the workflow's `GITHUB_TOKEN`) | build script |
| publications | ORCID public API (`/v3.0/<orcid>/works`) | build script |
| citations, h-index | Semantic Scholar API (author id); Google Scholar has no API | build script |
| peer reviews | ORCID `peer-reviews` | build script |

- A small script (`scripts/fetch-stats.mjs`, Node, no dependencies) runs in the Pages workflow
  before `jekyll build`. The workflow gains a daily `schedule`, so numbers are at most a day old.
- **If a source fails, the build keeps the last committed values** in `_data/stats.yml` rather
  than failing or showing zeros. Each panel shows "Updated 28 Sep 2026" from the file.
- Citations are shown as Semantic Scholar reports them, labelled as such; the Google Scholar link
  sits beside them for anyone who wants that count.

## 8. Behavior and accessibility

- **No horizontal page scroll at any width.** Wide things (tables, code, the contribution graph)
  scroll inside their own box; grid and flex children that hold them get `min-width: 0` so they
  can shrink. Checked on every page at 375, 768, 1280 and 1440px.

- Visible focus ring (`2px` accent outline) on every interactive element; a skip-to-content link.
- Respect `prefers-reduced-motion`: no animations then (and there are very few anyway).
- Every image has `alt`; post images get `loading="lazy"`.
- Page `<title>` is "Post title · Ali Naderi"; each page gets a `<meta name="description">` from
  its `brief`, plus Open Graph tags so shared links get a preview card.
- **Search, on the page, no external service** (replaces the Google box that opened a new tab):
  - Opens from the Search button, **`/`**, or **Ctrl/⌘+K** as a native `<dialog>` (focus trap,
    Esc and backdrop close for free) over a blurred page, 2px gradient on top.
  - Results update as you type: thumbnail, kind · tags · date in mono, title, brief. Up to 8.
    Empty query shows the latest posts. No match: **"No hits. Not even a B-side."**
  - **Every term must match** (title, tags or brief); title hits outrank tag hits outrank brief
    hits, then newer first. Matches are highlighted in the accent colour.
  - Keyboard: ↑ ↓ move, ↵ opens, Esc closes. The **Esc** badge in the box and **Esc close** in the
    footer are real buttons, so clicking them closes it too, as does clicking the backdrop. The input is an ARIA combobox with
    `aria-activedescendant`, so screen readers announce the selected result.
  - **The index** is `/search.json`, generated by Jekyll at build time, fetched once on first open
    (a few KB for the whole site):

    ```liquid
    ---
    layout: null
    permalink: /search.json
    ---
    [{% for p in site.posts %}{"t":{{ p.title | jsonify }},"d":{{ p.brief | jsonify }},"u":{{ p.url | relative_url | jsonify }},"img":{{ p.img | jsonify }},"date":{{ p.date | date: "%F" | jsonify }},"tags":{{ p.categories | join: " " | jsonify }}}{% unless forloop.last %},{% endunless %}{% endfor %}]
    ```

    Drafts aren't in `site.posts`, so unpublished posts never appear in search.

## 9. Rollout

1. Tokens, base styles and the new `main`/`post` layouts. Posts keep their front matter unchanged.
2. Home, Blog and Post (the pages most people see).
3. Stats: `scripts/fetch-stats.mjs`, `_data/stats.yml`, the daily schedule, and the two panels.
4. Projects, Library, About, Research (with `_data/publications.yml`).
5. Delete what nothing uses any more: Bootstrap, the extra icon sets, `assets/vendor/*`, the font
   zips, the WebGL background, `github-metrics.js`, `pager.js`, `prism.js` and `prism.css`.
