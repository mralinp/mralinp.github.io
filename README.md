[![Deploy](https://github.com/mralinp/mralinp.github.io/actions/workflows/jekyll.yml/badge.svg)](https://github.com/mralinp/mralinp.github.io/actions/workflows/jekyll.yml)

# alinaderiparizi.com

Source of my personal site: blog, projects, research, notes, library and CV. [Jekyll](https://jekyllrb.com/) with hand-written HTML/CSS/JS, hosted on GitHub Pages. No trackers, no third-party scripts, no framework.

## Run it

Ruby `3.2.6` (see `.ruby-version`) and Bundler.

```bash
bundle install
bundle exec jekyll serve --drafts
```

Open <http://127.0.0.1:4000>. Drafts in `_drafts/` show up locally and never in production.

<details>
<summary>Installing Ruby</summary>

**macOS**

```bash
brew install chruby ruby-install xz
ruby-install ruby 3.2.6
echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
```

**Debian / Ubuntu**

```bash
sudo apt-get install ruby-full build-essential zlib1g-dev
echo 'export GEM_HOME="$HOME/gems"' >> ~/.zshrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.zshrc
gem install bundler
```

**Arch / Manjaro**

```bash
sudo pacman -S --needed base-devel git zlib openssl libffi libyaml gmp readline rbenv ruby-build
rbenv install 3.2.6 && rbenv global 3.2.6
gem install bundler
```

If `Liquid Exception ... tainted?` shows up, Liquid is too old for Ruby 3.2: `bundle update liquid`.

</details>

## Write

| What | Where | Template |
|---|---|---|
| Blog post (incl. case studies) | `_posts/blog/YYYY-MM-DD-slug.markdown` | `_templates/blog-post.markdown` |
| Project write-up | `_posts/project/` | same, first category `project` |
| Book review | `_posts/books/` | same, first category `book` |
| Note | `_notes/YYYY-MM-DD-slug.markdown` | `_templates/note.markdown` |
| Unpublished | `_drafts/` | |

The **first category** decides where a post appears (Blog, Projects or Library). Put `featured: true` on a project to list it under "Selected work". Post images go in `assets/images/posts/`.

CV, nav, profiles and publications are data files in `_data/`. The About page is `about.md`.

## What's generated

- `search.json` feeds the on-page search.
- `/llms.txt` is hand-written; `/llms-full.txt` is built by `_plugins/llms_full.rb` from every published post.
- `scripts/fetch-stats.mjs` (Node 18+) writes `_data/stats.json` with the GitHub and academic numbers. It runs in CI and keeps the last value if a source fails. Run it locally with `GITHUB_TOKEN=... node scripts/fetch-stats.mjs`.
- `design/` is the design spec and mockup. It is excluded from the build.

## Deploy

Push to `main`. The [workflow](.github/workflows/jekyll.yml) refreshes the stats, builds, and publishes to GitHub Pages. It also runs daily at 03:17 UTC to keep the numbers fresh.

## License

Code: [GPL-3.0](LICENSE). Writing and original images: [CC BY-SA 4.0](LICENSE-CONTENT.md).
