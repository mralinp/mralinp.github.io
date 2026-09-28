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
6. **Small.** One stylesheet, one font family (Inter, already bundled) plus JetBrains Mono for code,
   one icon set (inline SVG), no CSS framework. MathJax and Prism load only on posts that need them.
7. **Standard patterns.** Top navigation, a footer with links, breadcrumbs back from a post, prev/next
   at the end. Nothing a visitor has to learn.

## 3. Information architecture

```text
Home ─┬─ Blog ──────── post
      ├─ Projects ──── project post
      ├─ Research          (publications, from _data/publications.yml)
      ├─ Library ───── book review
      └─ About             (bio, experience, education, CV download)
```

- **Top nav**: name on the left; Blog · Projects · Research · Library · About on the right; then
  search and the theme toggle. On phones it collapses into a menu button.
- **Footer**: GitHub · LinkedIn · Twitter/X · Email · RSS · CV, and © year.
- Notes join the nav under Blog once there is more than one.

## 4. Design tokens

All colors are CSS custom properties on `:root`, redefined for dark mode. Nothing outside the token
block uses a raw color.

| Token | Light | Dark | Use |
| --- | --- | --- | --- |
| `--bg` | `#ffffff` | `#0f1115` | page |
| `--surface` | `#f6f7f9` | `#171a21` | cards, code blocks |
| `--border` | `#e4e7ec` | `#2a2f3a` | dividers, card borders |
| `--text` | `#1a1d23` | `#e6e8ec` | body |
| `--muted` | `#5b6472` | `#9aa3b2` | meta, captions |
| `--accent` | `#d9480f` | `#ff8a4c` | links, focus, active nav |
| `--accent-soft` | `#fff1e8` | `#2a1a12` | tag chips, highlights |
| `--on-accent` | `#ffffff` | `#1a0d05` | text on accent buttons |

The accent keeps the current site's orange, so the redesign still feels like the same site.

| Type | Size / line height | Weight |
| --- | --- | --- |
| Page title | 40 / 1.15 (32 on phones) | 700 |
| Post title | 36 / 1.2 (28 on phones) | 700 |
| H2 | 26 / 1.3 | 650 |
| H3 | 20 / 1.4 | 600 |
| Body | 18 / 1.7 | 400 |
| Meta, captions | 14 / 1.5 | 400–500 |
| Code | 15 / 1.6, JetBrains Mono | 400 |

Spacing is a 4px scale (`4 8 12 16 24 32 48 64 96`). Radius is `8px` for cards and `6px` for chips
and buttons. Content widths: `68ch` for reading, `1120px` for grids.

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
- **Tag chip**: `--accent-soft` background, sentence case, links to the tag's filtered list.
- **Callout** in posts: note / warning, left border in the accent.
- **Code block**: `--surface` background, copy button, language label.

## 6. Pages

**Home.** A short intro (photo, name, one sentence: "Machine learning researcher and software
engineer in Tehran. I write about deep learning, medical imaging, networks and the software I
build."), two buttons (Read the blog · About me), then *At a glance* (the stats panels, below)
right under the intro, then *Recent posts* (5 rows), *Featured projects* (3 cards) and *Selected
publications* (2–3 rows).

**At a glance** is two panels side by side (stacked on phones):

- **Open source**: public repos · stars · contributions in the last year · followers, the
  contribution graph, and "Most active in: Python, Rust, TypeScript" from repo languages. Links to
  the GitHub profile.
- **Academic**: publications · citations · h-index · peer reviews, then the latest publication as
  one line. Links to Google Scholar and ORCID.

**Blog.** Title, a tag filter row (All · AI · Embedded · Network · Math …), and posts grouped by year
as post rows. Filtering is client-side on data attributes; with JavaScript off, all posts show.

**Post.** Breadcrumb (Blog / Projects), title, meta line (date · reading time · tags), optional hero
image, then the 68ch body. A sticky table of contents sits in the right margin on wide screens for
posts with 4+ headings. At the end: GitHub link if the post has one, prev/next, and back to the list.

**Projects.** The Open source panel on top, then the grid of project cards (3 / 2 / 1 columns),
newest first.

**Research.** The Academic panel on top, then publications from `_data/publications.yml`,
grouped by year, then Service (reviewing, judging).

**Library.** Grid of book cards.

**About.** Photo and short bio, then Experience and Education as simple timelines, skills as
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

- Visible focus ring (`2px` accent outline) on every interactive element; a skip-to-content link.
- Respect `prefers-reduced-motion`: no animations then (and there are very few anyway).
- Every image has `alt`; post images get `loading="lazy"`.
- Page `<title>` is "Post title · Ali Naderi"; each page gets a `<meta name="description">` from
  its `brief`, plus Open Graph tags so shared links get a preview card.
- Search: client-side over a build-time JSON index (title, brief, tags). No external service.

## 9. Rollout

1. Tokens, base styles and the new `main`/`post` layouts. Posts keep their front matter unchanged.
2. Home, Blog and Post (the pages most people see).
3. Stats: `scripts/fetch-stats.mjs`, `_data/stats.yml`, the daily schedule, and the two panels.
4. Projects, Library, About, Research (with `_data/publications.yml`).
5. Delete what nothing uses any more: Bootstrap, the extra icon sets, `assets/vendor/*`, the font
   zips, the WebGL background, `github-metrics.js`, `pager.js`.
