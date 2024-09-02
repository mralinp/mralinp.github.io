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

Install the latest stable version of Ruby (supported by Jekyll):

```console
$ ruby-install ruby 3.1.3
```

This will take a few minutes, and once it’s done, configure your shell to automatically use chruby:

```console
$ echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
$ echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
$ echo "chruby ruby-3.1.3" >> ~/.zshrc # run 'chruby' to see actual version
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

## Run the project
First, install ruby and gem.

```console
$ bundle exec jekyll serve --watch --drafts
```
