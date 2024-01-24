# My personal website repo
My personal homepage which is also my blog and home page. It's created using [Jekyll](https://jekyllrb.com/) and [Bootstrap](https://getbootstrap.com/) and hosted on [Github-Pages](https://pages.github.com/).
## Requirements

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
