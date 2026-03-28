![example workflow](https://github.com/mralinp/mralinp.github.io/actions/workflows/pages/pages-build-deployment/badge.svg)
# My personal website repo
My personal homepage which is also my blog and home page. It's created using [Jekyll](https://jekyllrb.com/) and [Bootstrap](https://getbootstrap.com/) and hosted on [Github-Pages](https://pages.github.com/).
## Development Setup

### Mac OSX
- Step 1: Install Homebrew

```console
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- Step 2: Install chruby and the latest Ruby with ruby-install

```console
$ brew install chruby ruby-install xz
```

Install Ruby `3.2.6`:

```console
$ ruby-install ruby 3.2.6
```

This will take a few minutes, and once it’s done, configure your shell to automatically use chruby:

```console
$ echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
$ echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
$ echo "chruby ruby-3.2.6" >> ~/.zshrc # run 'chruby' to see actual version
```

Quit and relaunch Terminal, then check that everything is working:

```console
$ ruby -v
```

### Linux (Debian/Ubuntu)

This project has the following requirements to run:
Ruby, Gem, Jekyll

```console
$ sudo apt-get install ruby-full build-essential zlib1g-dev
```

to enable bundle by setting environment variables:
```console
$ echo '# Install Ruby Gems to ~/gems' >> ~/.zshrc &&
echo 'export GEM_HOME="$HOME/gems"' >> ~/.zshrc &&
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.zshrc &&
```

reopen the terminal, then install gem package manager:
```console
$ gem install jekyll bundler
```

Finally, install gem packages using bundle:
```console
$ bundle install
```

### Linux (Arch/Manjaro)

This project works well on Arch-based systems using `rbenv`.

Install dependencies and Ruby tools:
```console
$ sudo pacman -Syu
$ sudo pacman -S --needed base-devel git zlib openssl libffi libyaml gmp readline rbenv ruby-build
```

Enable `rbenv` in your shell:
```console
$ echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.zshrc
$ echo 'eval "$(rbenv init - zsh)"' >> ~/.zshrc
$ exec zsh
```

Install and select Ruby `3.2.6`:
```console
$ rbenv install 3.2.6
$ rbenv global 3.2.6
$ ruby -v
```

Install Bundler version used by this repository and install gems:
```console
$ gem install bundler -v 2.3.18
$ cd /home/ozma/Source/alinaderiparizi.com/mralinp.github.io
$ bundle _2.3.18_ install
```

## Run the project
First, install ruby and gem.

```console
$ bundle exec jekyll serve --watch --drafts --host 127.0.0.1 --port 4000
```

Then open <http://127.0.0.1:4000>.

## Troubleshooting

### `ruby: command not found`

If you use `rbenv`, make sure your shell is initialized:

```console
$ echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.zshrc
$ echo 'eval "$(rbenv init - zsh)"' >> ~/.zshrc
$ exec zsh
```

### `Liquid Exception ... tainted?`

On Ruby `3.2.x`, `liquid 4.0.3` can fail. This repo pins Liquid to a compatible version (`>= 4.0.4`), so run:

```console
$ bundle _2.3.18_ update liquid
$ bundle _2.3.18_ install
```

## Content workflow

This repository is organized for three markdown-first content types:

- Blog posts: `_posts/`
- Notes: `_notes/`
- About/Resume page: `about.md`

### Create a new blog post

1. Copy `_templates/blog-post.markdown` to `_posts/` with a date prefix.
2. Example filename: `_posts/blog/2026-03-27-my-post.markdown`
3. Commit and push to your branch.

### Create a new note

1. Copy `_templates/note.markdown` to `_notes/` with a date prefix.
2. Example filename: `_notes/2026-03-27-learning-note.markdown`
3. Commit and push to your branch.

### Update about/resume

Edit `about.md` directly in markdown.

## Deploy flow

Push changes to your branch and merge into `main`. GitHub Actions builds and deploys the site to GitHub Pages automatically.
