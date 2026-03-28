---
title: Projects
layout: main
---

<style>
    #content {
        height: 100%;
    }

    .ops-metrics {
        position: relative;
        margin-bottom: 1rem;
        border: 1px solid #1a1a1a;
        background: rgba(5, 5, 5, 0.78);
        padding: 1.05rem;
        overflow: hidden;
    }

    .ops-bg {
        position: absolute;
        top: 0.2rem;
        right: 0.45rem;
        font-size: 6.2rem;
        color: rgba(255, 255, 255, 0.06);
        user-select: none;
        pointer-events: none;
    }

    .ops-shell {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: 1.1fr 1fr;
        gap: 1rem;
        align-items: start;
    }

    .ops-kicker {
        margin: 0;
        color: #ff5a00;
        font: 500 0.66rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.22em;
        text-transform: uppercase;
    }

    .ops-title {
        margin: 0.2rem 0 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: clamp(1.2rem, 2.6vw, 1.75rem);
    }

    .ops-subtitle {
        margin: 0 0 0.85rem;
        color: #94a3b8;
        font-size: 0.88rem;
    }

    .ops-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.55rem;
    }

    .ops-tile {
        border: 1px solid #2b2b2b;
        background: rgba(0, 0, 0, 0.35);
        padding: 0.6rem;
    }

    .ops-tile .value {
        margin: 0;
        font: 700 1.15rem/1.1 "JetBrains Mono", monospace;
    }

    .ops-tile .label {
        margin: 0.28rem 0 0;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font: 500 0.56rem/1 "JetBrains Mono", monospace;
    }

    .ops-grid-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.45rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font: 500 0.56rem/1 "JetBrains Mono", monospace;
    }

    .ops-grid-head strong {
        color: #22c55e;
    }

    .ops-contrib {
        border: 1px solid #2b2b2b;
        background: rgba(0, 0, 0, 0.38);
        padding: 0.7rem;
    }

    .ops-contrib-grid {
        display: grid;
        grid-template-rows: repeat(7, 8px);
        grid-auto-flow: column;
        grid-auto-columns: 8px;
        gap: 3px;
        overflow-x: auto;
    }

    .ops-cell {
        width: 8px;
        height: 8px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .ops-l0 { background: #0b0b0b; }
    .ops-l1 { background: #063015; }
    .ops-l2 { background: #0d6e35; }
    .ops-l3 { background: #1ea05d; }
    .ops-l4 { background: #4ade80; }

    .ops-axis {
        margin-top: 0.45rem;
        display: flex;
        justify-content: space-between;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font: 500 0.53rem/1 "JetBrains Mono", monospace;
    }

    .ops-ref {
        margin-top: 0.5rem;
        text-align: right;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font: 500 0.56rem/1 "JetBrains Mono", monospace;
    }

    .ops-ref a {
        color: #ff8a44;
        text-decoration: none;
        border-bottom: 1px solid rgba(255, 138, 68, 0.35);
    }

    .ops-ref a:hover {
        color: #ff5a00;
    }

    .blog-matrix-head {
        display: flex;
        flex-wrap: wrap;
        align-items: end;
        justify-content: space-between;
        gap: 0.8rem;
        border-bottom: 1px solid #1a1a1a;
        padding-bottom: 0.6rem;
        margin-bottom: 1rem;
    }

    .blog-matrix-head p {
        margin: 0;
        color: #ff5a00;
        font: 500 0.66rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    .blog-matrix-head h2 {
        margin: 0.2rem 0 0;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: clamp(1.2rem, 2.2vw, 1.8rem);
    }

    .blog-matrix-status {
        color: #9ca3af;
        font: 500 0.62rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .blog-card {
        height: 100%;
        border: 1px solid #1a1a1a;
        background: rgba(5, 5, 5, 0.72);
        display: flex;
        flex-direction: column;
        transition: border-color 0.35s ease;
    }

    .blog-card:hover {
        border-color: rgba(255, 90, 0, 0.62);
    }

    .blog-card-media {
        position: relative;
        height: 168px;
        border-bottom: 1px solid #1a1a1a;
        overflow: hidden;
        background: #000;
    }

    .blog-card-fallback {
        width: 100%;
        height: 100%;
        background:
            radial-gradient(circle at 25% 22%, rgba(106, 76, 255, 0.32), transparent 38%),
            radial-gradient(circle at 74% 68%, rgba(255, 90, 0, 0.24), transparent 40%),
            linear-gradient(160deg, #070910 0%, #06070f 42%, #03040a 100%);
    }

    .blog-card-media img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: grayscale(1);
        opacity: 0.48;
        transition: transform 0.55s ease, filter 0.55s ease, opacity 0.55s ease;
    }

    .blog-card:hover .blog-card-media img {
        transform: scale(1.05);
        filter: grayscale(0);
        opacity: 0.65;
    }

    .blog-card-badge {
        position: absolute;
        top: 0.6rem;
        left: 0.6rem;
        background: #ff5a00;
        color: #000;
        padding: 0.15rem 0.45rem;
        font: 700 0.56rem/1.2 "JetBrains Mono", monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .blog-card-open {
        position: absolute;
        right: 0.55rem;
        bottom: 0.45rem;
        color: #ff5a00;
        opacity: 0;
        transition: opacity 0.35s ease;
    }

    .blog-card:hover .blog-card-open {
        opacity: 1;
    }

    .blog-card-content {
        padding: 0.85rem;
        display: flex;
        flex-direction: column;
        flex: 1;
    }

    .blog-card-top {
        display: flex;
        gap: 0.5rem;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 0.45rem;
    }

    .blog-card-title {
        margin: 0;
        font-size: 1rem;
        line-height: 1.35;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }

    .blog-card-title a {
        color: #f8fafc;
        text-decoration: none;
    }

    .blog-card-title a:hover {
        color: #ff5a00;
    }

    .blog-card-date {
        white-space: nowrap;
        font: 500 0.58rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.12em;
        color: #9ca3af;
        text-transform: uppercase;
        margin-top: 0.15rem;
    }

    .blog-card-stat {
        white-space: nowrap;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font: 500 0.54rem/1 "JetBrains Mono", monospace;
        margin-top: 0.15rem;
    }

    .blog-card p {
        color: #94a3b8;
        margin: 0 0 0.65rem;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .blog-card-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0 0 0.85rem;
    }

    .blog-card-tags span {
        border: 1px solid #2b2b2b;
        color: #9ca3af;
        padding: 0.1rem 0.35rem;
        font: 500 0.56rem/1.2 "JetBrains Mono", monospace;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .blog-card-actions {
        margin-top: auto;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.45rem;
    }

    .blog-card-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
        padding: 0.46rem 0.4rem;
        font: 700 0.56rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        text-decoration: none;
        border: 1px solid #333;
        background: #ff5a00;
        color: #000;
        border-color: #ff5a00;
    }

    .blog-card-btn:hover {
        background: #d84b00;
        color: #000;
    }

    .blog-card-btn.ghost {
        background: transparent;
        color: #f8fafc;
        border-color: #333;
    }

    .blog-card-btn.ghost:hover {
        background: rgba(255, 255, 255, 0.04);
        color: #ff5a00;
        border-color: #ff5a00;
    }

    @media (max-width: 980px) {
        .ops-shell {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 720px) {
        .ops-grid {
            grid-template-columns: 1fr;
        }
    }
</style>

<section>
    <div class="ops-metrics">
        <span class="material-symbols-outlined ops-bg">analytics</span>
        <div class="ops-shell">
            <div>
                <p class="ops-kicker">Operational Metrics</p>
                <h2 class="ops-title">Open Source Impact</h2>
                <p class="ops-subtitle">Live telemetry from your GitHub profile and yearly contribution channel.</p>
                <div class="ops-grid">
                    <article class="ops-tile">
                        <p class="value" id="projectMetricRepos">--</p>
                        <p class="label">Repos</p>
                    </article>
                    <article class="ops-tile">
                        <p class="value" id="projectMetricStars">--</p>
                        <p class="label">Stars</p>
                    </article>
                    <article class="ops-tile">
                        <p class="value" id="projectMetricCommits">--</p>
                        <p class="label">Commits (1Y)</p>
                    </article>
                </div>
            </div>
            <div>
                <div class="ops-grid-head">
                    <span>Global Contribution Grid</span>
                    <strong id="projectMetricsStatus">Syncing...</strong>
                </div>
                <div class="ops-contrib">
                    <div class="ops-contrib-grid" id="projectsContribGrid"></div>
                    <div class="ops-axis">
                        <span>Jan</span><span>Apr</span><span>Jul</span><span>Oct</span>
                    </div>
                    <div class="ops-ref">
                        Source: <a href="https://github.com/mralinp" target="_blank" rel="noopener noreferrer">github.com/mralinp</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="blog-matrix-head">
        <div>
            <p>Portfolio Matrix</p>
            <h2>Project Deployments</h2>
        </div>
        <span class="blog-matrix-status">Feed status: live</span>
    </div>

    {% assign projects = site.posts | where_exp: "post", "post.categories.first == 'project'" %}
    <div class="row" id="projectsGrid">
        {% for post in projects %}
        <div class="col-lg-4 col-md-6 mb-4 js-page-item">
            <article class="blog-card">
                <div class="blog-card-media">
                    <a href="{{ post.url }}">
                        {% if post.img %}
                        <img src="{{ post.img }}" alt="{{ post.title | escape }}">
                        {% else %}
                        <div class="blog-card-fallback"></div>
                        {% endif %}
                    </a>
                    <span class="blog-card-badge">{{ post.categories[0] | default: "Project" | upcase }}</span>
                    <span class="blog-card-open material-symbols-outlined">north_east</span>
                </div>
                <div class="blog-card-content">
                    <div class="blog-card-top">
                        <h3 class="blog-card-title"><a href="{{ post.url }}" title="{{ post.title }}">{{ post.title | strip_html | truncate: 62 }}</a></h3>
                        <span class="blog-card-stat">Updated // {{ post.date | date: "%b %Y" }}</span>
                    </div>
                    <p>{{ post.brief | default: post.excerpt | strip_html | truncate: 130 }}</p>
                    <div class="blog-card-tags">
                        {% for category in post.categories limit: 3 %}
                        <span>{{ category }}</span>
                        {% endfor %}
                    </div>
                    <div class="blog-card-actions">
                        <a href="{{ post.url }}" class="blog-card-btn">Read more</a>
                        <a href="{{ post.github | default: 'https://github.com/mralinp' }}" target="_blank" rel="noopener noreferrer" class="blog-card-btn ghost"><i class="fab fa-github"></i> View in GitHub</a>
                    </div>
                </div>
            </article>
        </div>
        {% endfor %}
    </div>

    <nav aria-label="Page navigation" id="PageNav">
        <ul class="pagination justify-content-center" id="projectsPager"></ul>
    </nav>
</section>

<script>
    (function () {
        var username = "mralinp";
        var grid = document.getElementById("projectsContribGrid");
        var reposEl = document.getElementById("projectMetricRepos");
        var starsEl = document.getElementById("projectMetricStars");
        var commitsEl = document.getElementById("projectMetricCommits");
        var statusEl = document.getElementById("projectMetricsStatus");
        if (!grid) return;

        function formatCount(value) {
            if (value >= 1000000) return (value / 1000000).toFixed(1).replace(".0", "") + "M";
            if (value >= 1000) return (value / 1000).toFixed(1).replace(".0", "") + "K";
            return String(value);
        }

        function paintGrid(contributions) {
            var cells = contributions || [];
            var totalCells = 7 * 53;
            var start = Math.max(0, cells.length - totalCells);
            var trimmed = cells.slice(start);
            while (trimmed.length < totalCells) trimmed.unshift({ level: 0 });
            for (var i = 0; i < trimmed.length; i += 1) {
                var item = trimmed[i] || { level: 0 };
                var level = typeof item.level === "number" ? item.level : 0;
                if (level < 0) level = 0;
                if (level > 4) level = 4;
                var cell = document.createElement("span");
                cell.className = "ops-cell ops-l" + level;
                grid.appendChild(cell);
            }
        }

        function fetchAllRepos() {
            var page = 1;
            var allRepos = [];
            function next() {
                return fetch("https://api.github.com/users/" + username + "/repos?per_page=100&page=" + page + "&sort=updated")
                    .then(function (res) { return res.ok ? res.json() : []; })
                    .then(function (repos) {
                        if (!Array.isArray(repos) || repos.length === 0) return allRepos;
                        allRepos = allRepos.concat(repos);
                        if (repos.length < 100) return allRepos;
                        page += 1;
                        return next();
                    });
            }
            return next();
        }

        Promise.all([
            fetch("https://api.github.com/users/" + username).then(function (res) { return res.ok ? res.json() : null; }),
            fetchAllRepos(),
            fetch("https://github-contributions-api.jogruber.de/v4/" + username + "?y=last").then(function (res) { return res.ok ? res.json() : null; })
        ]).then(function (results) {
            var user = results[0];
            var repos = results[1];
            var contributions = results[2];
            var publicRepos = user && typeof user.public_repos === "number" ? user.public_repos : 0;
            if (reposEl) reposEl.textContent = formatCount(publicRepos);
            var stars = 0;
            for (var i = 0; i < repos.length; i += 1) stars += repos[i].stargazers_count || 0;
            if (starsEl) starsEl.textContent = formatCount(stars);
            var yearlyCommits = contributions && contributions.total && typeof contributions.total.lastYear === "number"
                ? contributions.total.lastYear
                : 0;
            if (commitsEl) commitsEl.textContent = formatCount(yearlyCommits);
            paintGrid(contributions && Array.isArray(contributions.contributions) ? contributions.contributions : []);
            if (statusEl) statusEl.textContent = "Live // GitHub";
        }).catch(function () {
            if (reposEl) reposEl.textContent = "55";
            if (starsEl) starsEl.textContent = "254";
            if (commitsEl) commitsEl.textContent = "226";
            paintGrid([]);
            if (statusEl) statusEl.textContent = "Fallback";
        });
    })();
</script>

<script>
    (function () {
        var perPage = 6;
        var items = Array.prototype.slice.call(document.querySelectorAll("#projectsGrid .js-page-item"));
        var pager = document.getElementById("projectsPager");
        if (!items.length || !pager) return;
        var totalPages = Math.ceil(items.length / perPage);
        var current = 1;

        function renderPage(page) {
            current = Math.max(1, Math.min(totalPages, page));
            var start = (current - 1) * perPage;
            var end = start + perPage;
            for (var i = 0; i < items.length; i += 1) {
                items[i].style.display = i >= start && i < end ? "" : "none";
            }
            renderPager();
            window.scrollTo({ top: 0, behavior: "smooth" });
        }

        function pageLink(label, page, disabled, active) {
            var li = document.createElement("li");
            li.className = "page-item" + (disabled ? " disabled" : "") + (active ? " active" : "");
            var a = document.createElement(active ? "span" : "a");
            a.className = "page-link";
            a.textContent = label;
            if (!disabled && !active) {
                a.href = "#";
                a.addEventListener("click", function (e) {
                    e.preventDefault();
                    renderPage(page);
                });
            }
            li.appendChild(a);
            return li;
        }

        function renderPager() {
            pager.innerHTML = "";
            pager.appendChild(pageLink("Previous", current - 1, current === 1, false));
            for (var p = 1; p <= totalPages; p += 1) {
                pager.appendChild(pageLink(String(p), p, false, p === current));
            }
            pager.appendChild(pageLink("Next", current + 1, current === totalPages, false));
        }

        renderPage(1);
    })();
</script>