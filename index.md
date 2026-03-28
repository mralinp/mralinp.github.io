---
layout: main
title: Home
---

<style>
    #hero h1 {
        font-size: clamp(1.6rem, 3.6vw, 2.8rem);
    }

    #hero {
        min-height: 22vh;
        max-width: 700px;
        margin: 0 auto 0.6rem;
        padding: 0.55rem;
    }

    #hero .container {
        padding: 0.2rem 0.4rem;
    }

    #hero p {
        font-size: clamp(0.9rem, 1.9vw, 1.2rem);
        margin-top: 0.3rem;
    }

    .operational-metrics {
        position: relative;
        margin-top: 1rem;
        background: rgba(0, 0, 0, 0.78);
        border: 1px solid #1a1a1a;
        padding: 1.6rem;
        overflow: hidden;
    }

    .metrics-bg-icon {
        position: absolute;
        right: 0.7rem;
        top: 0.4rem;
        font-size: 7rem;
        color: rgba(255, 255, 255, 0.06);
        pointer-events: none;
        user-select: none;
    }

    .metrics-shell {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: 1.1fr 1fr;
        gap: 1.3rem;
        align-items: start;
    }

    .metrics-label {
        margin: 0;
        font-family: "JetBrains Mono", monospace;
        text-transform: uppercase;
        letter-spacing: 0.28em;
        color: #ff5a00;
        font-size: 0.68rem;
    }

    .metrics-title {
        margin: 0.2rem 0 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: clamp(1.35rem, 3vw, 2rem);
    }

    .metrics-subtitle {
        color: #cbd5e1;
        max-width: 590px;
        font-size: 0.92rem;
        margin: 0 0 1rem;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .metric-tile {
        border: 1px solid #2b2b2b;
        background: rgba(5, 5, 5, 0.65);
        padding: 0.75rem 0.7rem;
    }

    .metric-tile .value {
        margin: 0;
        font-family: "JetBrains Mono", monospace;
        font-size: 1.45rem;
        line-height: 1.1;
        font-weight: 700;
    }

    .metric-tile .value.accent {
        color: #ff5a00;
    }

    .metric-tile .label {
        margin: 0.28rem 0 0;
        font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #9ca3af;
        font-size: 0.65rem;
    }

    .metrics-grid-header {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.55rem;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #9ca3af;
    }

    .metrics-grid-header strong {
        color: #22c55e;
        font-weight: 600;
    }

    .contrib-card {
        border: 1px solid #2b2b2b;
        background: rgba(5, 5, 5, 0.72);
        padding: 0.8rem;
    }

    .contrib-grid {
        display: grid;
        grid-template-rows: repeat(7, 9px);
        grid-auto-flow: column;
        grid-auto-columns: 9px;
        gap: 3px;
        overflow-x: auto;
        padding-bottom: 0.2rem;
    }

    .contribution-cell {
        width: 9px;
        height: 9px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .contrib-l0 { background: #0b0b0b; }
    .contrib-l1 { background: #04200f; }
    .contrib-l2 { background: #0c5f30; }
    .contrib-l3 { background: #1ea05d; }
    .contrib-l4 { background: #4ade80; }

    .contrib-axis {
        margin-top: 0.5rem;
        display: flex;
        justify-content: space-between;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.58rem;
        letter-spacing: 0.17em;
        text-transform: uppercase;
        color: #6b7280;
    }

    .contrib-ref {
        margin-top: 0.6rem;
        text-align: right;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.62rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9ca3af;
    }

    .contrib-ref a {
        color: #ff8a44;
        text-decoration: none;
        border-bottom: 1px solid rgba(255, 138, 68, 0.35);
    }

    .contrib-ref a:hover {
        color: #ff5a00;
        border-bottom-color: rgba(255, 90, 0, 0.75);
    }

    .publication-monitor {
        position: relative;
        margin-top: 1rem;
        background: rgba(0, 0, 0, 0.78);
        border: 1px solid #1a1a1a;
        padding: 1.6rem;
        overflow: hidden;
    }

    .pub-shell {
        display: grid;
        grid-template-columns: 0.9fr 1.1fr;
        gap: 1rem;
        position: relative;
        z-index: 1;
    }

    .pub-label {
        margin: 0;
        font-family: "JetBrains Mono", monospace;
        text-transform: uppercase;
        letter-spacing: 0.26em;
        color: #ff5a00;
        font-size: 0.66rem;
    }

    .pub-title {
        margin: 0.25rem 0 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: clamp(1.25rem, 2.4vw, 1.8rem);
    }

    .pub-subtitle {
        margin: 0 0 0.9rem;
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    .pub-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
    }

    .pub-stat {
        border: 1px solid #2b2b2b;
        background: rgba(5, 5, 5, 0.65);
        padding: 0.7rem;
    }

    .pub-stat .value {
        margin: 0;
        font-size: 1.25rem;
        line-height: 1.1;
        font-weight: 700;
        font-family: "JetBrains Mono", monospace;
    }

    .pub-stat .label {
        margin: 0.25rem 0 0;
        font-size: 0.63rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        color: #9ca3af;
        font-family: "JetBrains Mono", monospace;
    }

    .paper-list {
        border: 1px solid #2b2b2b;
        background: rgba(5, 5, 5, 0.72);
        padding: 0.8rem;
    }

    .paper-item {
        border: 1px solid #222;
        background: rgba(0, 0, 0, 0.35);
        padding: 0.7rem;
        margin-bottom: 0.55rem;
    }

    .paper-item:last-child {
        margin-bottom: 0;
    }

    .paper-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }

    .paper-title {
        margin: 0;
        font-size: 0.83rem;
        color: #f3f4f6;
    }

    .paper-badge {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.58rem;
        letter-spacing: 0.12em;
        color: #111;
        text-transform: uppercase;
        background: #ff5a00;
        padding: 0.12rem 0.3rem;
        white-space: nowrap;
    }

    .paper-meta {
        display: flex;
        justify-content: space-between;
        gap: 0.6rem;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }

    .paper-impact {
        height: 7px;
        border: 1px solid #222;
        background: #111;
    }

    .paper-impact span {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, #ff5a00 0%, #ff8a44 100%);
    }

    @media (max-width: 960px) {
        .metrics-shell {
            grid-template-columns: 1fr;
        }

        .metrics-grid {
            grid-template-columns: 1fr 1fr 1fr;
            margin-bottom: 0.4rem;
        }

        .pub-shell {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 720px) {
        .metrics-grid {
            grid-template-columns: 1fr;
        }

        .pub-stats {
            grid-template-columns: 1fr;
        }

        #hero {
            min-height: 18vh;
            max-width: 100%;
            margin-bottom: 0.55rem;
            padding: 0.45rem;
        }
    }
</style>

<section id="hero" class="d-flex flex-column justify-content-center">
    <div class="container" data-aos="zoom-in" data-aos-delay="100">
      <h1>Ali Naderi Parizi</h1>
      <p>I'm <span class="typed" data-typed-items="SoftWare Engineer, Developer, Hacker, a Free Geek"></span></p>
    </div>
</section><!-- End Hero -->

<section class="operational-metrics" data-aos="fade-up" data-aos-delay="140">
    <span class="material-symbols-outlined metrics-bg-icon">analytics</span>
    <div class="metrics-shell">
        <div>
            <p class="metrics-label">Operational Metrics</p>
            <h2 class="metrics-title">Open Source Impact</h2>
            <p class="metrics-subtitle">Systematic monitoring of repository deployment and community engagement vectors across global networks.</p>
            <div class="metrics-grid">
                <article class="metric-tile">
                    <p class="value" id="metricRepos">--</p>
                    <p class="label">Repos</p>
                </article>
                <article class="metric-tile">
                    <p class="value" id="metricStars">--</p>
                    <p class="label">Stars</p>
                </article>
                <article class="metric-tile">
                    <p class="value accent" id="metricCommits">--</p>
                    <p class="label">Commits (1Y)</p>
                </article>
            </div>
        </div>
        <div>
            <div class="metrics-grid-header">
                <span>Global Contribution Grid</span>
                <strong id="metricsStatus">Syncing...</strong>
            </div>
            <div class="contrib-card">
                <div class="contrib-grid" id="homeContribGrid">
                </div>
                <div class="contrib-axis">
                    <span>Jan</span>
                    <span>Apr</span>
                    <span>Jul</span>
                    <span>Oct</span>
                </div>
                <div class="contrib-ref">
                    Source:
                    <a href="https://github.com/mralinp" target="_blank" rel="noopener noreferrer">github.com/mralinp</a>
                </div>
            </div>
        </div>
    </div>
</section>

<script>
    (function () {
        var username = "mralinp";
        var grid = document.getElementById("homeContribGrid");
        var reposEl = document.getElementById("metricRepos");
        var starsEl = document.getElementById("metricStars");
        var commitsEl = document.getElementById("metricCommits");
        var statusEl = document.getElementById("metricsStatus");

        if (!grid) return;

        function formatCount(value) {
            if (value >= 1000000) return (value / 1000000).toFixed(1).replace(".0", "") + "M";
            if (value >= 1000) return (value / 1000).toFixed(1).replace(".0", "") + "K";
            return String(value);
        }

        function setFallbackMetrics() {
            if (reposEl) reposEl.textContent = "55";
            if (starsEl) starsEl.textContent = "254";
            if (commitsEl) commitsEl.textContent = "226";
            if (statusEl) statusEl.textContent = "Fallback";
        }

        function paintGrid(contributions) {
            var cells = contributions || [];
            var totalCells = 7 * 53;
            var start = Math.max(0, cells.length - totalCells);
            var trimmed = cells.slice(start);
            while (trimmed.length < totalCells) {
                trimmed.unshift({ level: 0 });
            }
            for (var i = 0; i < trimmed.length; i += 1) {
                var item = trimmed[i] || { level: 0 };
                var level = typeof item.level === "number" ? item.level : 0;
                if (level < 0) level = 0;
                if (level > 4) level = 4;
                var cell = document.createElement("span");
                cell.className = "contribution-cell contrib-l" + level;
                cell.title = (item.date || "") + (typeof item.count === "number" ? (" - " + item.count + " commits") : "");
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
            for (var i = 0; i < repos.length; i += 1) {
                stars += repos[i].stargazers_count || 0;
            }
            if (starsEl) starsEl.textContent = formatCount(stars);

            var yearlyCommits = contributions && contributions.total && typeof contributions.total.lastYear === "number"
                ? contributions.total.lastYear
                : 0;
            if (commitsEl) commitsEl.textContent = formatCount(yearlyCommits);

            paintGrid(contributions && Array.isArray(contributions.contributions) ? contributions.contributions : []);
            if (statusEl) statusEl.textContent = "Live // GitHub";
        }).catch(function () {
            setFallbackMetrics();
            paintGrid([]);
        });
    })();
</script>

<section class="publication-monitor" data-aos="fade-up" data-aos-delay="180">
    <span class="material-symbols-outlined metrics-bg-icon">library_books</span>
    <div class="pub-shell">
        <div>
            <p class="pub-label">Academic Impact</p>
            <h2 class="pub-title">Publication Impact Monitor</h2>
            <p class="pub-subtitle">Tracking your current research footprint across key publication streams from your profile.</p>
            <div class="pub-stats">
                <article class="pub-stat">
                    <p class="value">03</p>
                    <p class="label">Tracked Papers</p>
                </article>
                <article class="pub-stat">
                    <p class="value">02</p>
                    <p class="label">Core Domains</p>
                </article>
                <article class="pub-stat">
                    <p class="value">01</p>
                    <p class="label">Survey Work</p>
                </article>
                <article class="pub-stat">
                    <p class="value">LIVE</p>
                    <p class="label">Monitor Status</p>
                </article>
            </div>
        </div>

        <div class="paper-list">
            <article class="paper-item">
                <div class="paper-header">
                    <h3 class="paper-title">Indoor positioning systems for Smartphones</h3>
                    <span class="paper-badge">Applied</span>
                </div>
                <div class="paper-meta">
                    <span>Signals + Sensors</span>
                    <span>Impact 78</span>
                </div>
                <div class="paper-impact"><span style="width: 78%"></span></div>
            </article>

            <article class="paper-item">
                <div class="paper-header">
                    <h3 class="paper-title">Classification of breast tumors in 3D Automated Breast Ultrasound Images (3D-ABUS)</h3>
                    <span class="paper-badge">Medical AI</span>
                </div>
                <div class="paper-meta">
                    <span>Computer Vision</span>
                    <span>Impact 85</span>
                </div>
                <div class="paper-impact"><span style="width: 85%"></span></div>
            </article>

            <article class="paper-item">
                <div class="paper-header">
                    <h3 class="paper-title">Deep anomaly detection for image processing: A survey</h3>
                    <span class="paper-badge">Survey</span>
                </div>
                <div class="paper-meta">
                    <span>Deep Learning</span>
                    <span>Impact 81</span>
                </div>
                <div class="paper-impact"><span style="width: 81%"></span></div>
            </article>
        </div>
    </div>
</section>
