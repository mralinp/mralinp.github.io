# Writes /llms-full.txt: the whole site as one Markdown file for language models (llmstxt.org).
# /llms.txt is the index; this is the text behind it. A generator, because generators run before
# rendering, so `post.content` is still the author's Markdown, not HTML. Drafts are never in
# site.posts on a normal build, so unpublished posts never leak into it.
module LlmsFull
  class Generator < Jekyll::Generator
    safe true
    priority :lowest

    def generate(site)
      base = site.config["url"].to_s.chomp("/")
      data = site.data
      out = []

      out << "# Ali Naderi: full text of alinaderiparizi.com\n\n" \
             "> Every page and post on the site, as Markdown. Index: #{base}/llms.txt. " \
             "Writing is CC BY-SA 4.0 (credit Ali Naderi, link back); code is GPL-3.0."

      about = site.pages.find { |p| p.url == "/about/" }
      out << "## About\n\n#{strip_tags(about.content).strip}" if about

      cv = data["cv"] || {}
      { "experience" => "Experience", "projects" => "Research projects",
        "teaching" => "Teaching", "education" => "Education" }.each do |key, title|
        next unless cv[key]
        out << "## #{title}\n\n" + cv[key].map { |e| cv_entry(e) }.join("\n\n")
      end

      if data["publications"]
        out << "## Publications\n\n" + data["publications"].map { |p|
          venue = p["status"] == "ongoing" ? "ongoing work" : "#{p['venue']}, #{p['year']}"
          doi = (p["links"] || []).map { |l| " #{l['url']}" }.join
          "- #{p['title']}. #{p['authors'].join(', ')}. #{venue}.#{doi}"
        }.join("\n")
      end

      if cv["skills"]
        out << "## Skills\n\n" + cv["skills"].map { |s| "- #{s['group']}: #{s['items']}" }.join("\n")
      end

      site.posts.docs.reverse_each do |post|
        d = post.data
        kind = { "project" => "Project", "book" => "Book review" }.fetch(d["categories"].first, "Blog post")
        head = ["# #{d['title']}", "",
                "- URL: #{base}#{post.url}",
                "- Date: #{d['date'].strftime('%Y-%m-%d')}",
                "- Kind: #{kind}",
                "- Topics: #{d['categories'].drop(1).join(', ')}"]
        head << "- Code: #{d['github']}" if d["github"]
        head << "- Summary: #{d['brief']}" if d["brief"]
        out << head.join("\n") + "\n\n" + absolute(post.content.strip, base)
      end

      page = Jekyll::PageWithoutAFile.new(site, site.source, "", "llms-full.txt")
      page.content = out.join("\n\n---\n\n") + "\n"
      page.data["layout"] = nil
      page.data["sitemap"] = false
      page.data["render_with_liquid"] = false # posts contain {{ }} in code samples
      site.pages << page
    end

    private

    def cv_entry(e)
      s = "### #{e['what']} (#{e['when']})\n#{e['where']}"
      s += " · #{e['site']['url']}" if e["site"]
      s += "\n" + e["bullets"].map { |b| "- #{strip_tags(b)}" }.join("\n") if e["bullets"]
      s += "\nTags: #{e['tags'].join(', ')}" if e["tags"]
      s += "\n#{e['link']['label']}: #{e['link']['url']}" if e["link"]
      s
    end

    def strip_tags(html)
      html.to_s.gsub(/<[^>]+>/, "")
    end

    # root-relative links and images become absolute, so they still work outside the site
    def absolute(md, base)
      md.gsub(/\]\(\//, "](#{base}/").gsub(/(src|href)="\//, "\\1=\"#{base}/")
    end
  end
end
