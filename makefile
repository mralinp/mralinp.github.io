all:
	bundle exec jekyll serve --watch --drafts

install:
	sudo apt-get install ruby-full build-essential zlib1g-dev --yes
	echo '# Install Ruby Gems to ~/gems' >> ~/.zshrc
	echo 'export GEM_HOME="$HOME/gems"' >> ~/.zshrc
	echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.zshrc
	source ~/.zshrc
	gem install jekyll bundler