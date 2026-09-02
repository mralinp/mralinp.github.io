---
title: Research
layout: main
---
<style>
    #content {
        height: 100%;
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

    .research-subtitle {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin: 0 0 1.1rem;
    }

    .research-subtitle a {
        color: #ff8a44;
    }

    .profile-links {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.1rem;
    }

    .profile-link {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border: 1px solid #2b2b2b;
        background: rgba(0, 0, 0, 0.35);
        color: #cbd5e1;
        padding: 0.32rem 0.65rem;
        font: 600 0.64rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        text-decoration: none;
    }

    .profile-link:hover {
        border-color: #ff5a00;
        color: #ff5a00;
    }

    .pub-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem;
        margin-bottom: 1.1rem;
    }

    .pub-stat {
        border: 1px solid #2b2b2b;
        background: rgba(5, 5, 5, 0.65);
        padding: 0.8rem;
    }

    .pub-stat .value {
        margin: 0;
        font-size: 1.45rem;
        line-height: 1.1;
        font-weight: 700;
        font-family: "JetBrains Mono", monospace;
    }

    .pub-stat .value a {
        color: inherit;
        text-decoration: none;
    }

    .pub-stat .value a:hover {
        color: #ff5a00;
    }

    .pub-stat .label {
        margin: 0.3rem 0 0;
        font-size: 0.63rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        color: #9ca3af;
        font-family: "JetBrains Mono", monospace;
    }

    .stat-source {
        margin: 0.3rem 0 0;
        font-size: 0.56rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
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
        padding: 0.8rem;
        margin-bottom: 0.55rem;
    }

    .paper-item:last-child {
        margin-bottom: 0;
    }

    .paper-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }

    .paper-title {
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.4;
        color: #f3f4f6;
    }

    .paper-title a {
        color: inherit;
        text-decoration: none;
        border-bottom: 1px solid rgba(255, 138, 68, 0.35);
    }

    .paper-title a:hover {
        color: #ff8a44;
        border-bottom-color: rgba(255, 90, 0, 0.75);
    }

    .paper-badge {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.58rem;
        letter-spacing: 0.12em;
        color: #111;
        text-transform: uppercase;
        background: #ff5a00;
        padding: 0.15rem 0.35rem;
        white-space: nowrap;
    }

    .paper-meta {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 0.6rem;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
    }

    .journal-rank {
        display: inline-block;
        margin-top: 0.5rem;
        font: 600 0.6rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.18rem 0.4rem;
        border: 1px solid;
    }

    .journal-rank.q1 {
        color: #4ade80;
        border-color: rgba(74, 222, 128, 0.4);
        background: rgba(74, 222, 128, 0.1);
    }

    .journal-rank.q2 {
        color: #60a5fa;
        border-color: rgba(96, 165, 250, 0.4);
        background: rgba(96, 165, 250, 0.1);
    }

    .journal-rank.q3 {
        color: #f59e0b;
        border-color: rgba(245, 158, 11, 0.4);
        background: rgba(245, 158, 11, 0.1);
    }

    .journal-rank.q4 {
        color: #9ca3af;
        border-color: rgba(156, 163, 175, 0.4);
        background: rgba(156, 163, 175, 0.1);
    }

    .research-loading,
    .research-empty {
        color: #9ca3af;
        font-size: 0.85rem;
        margin: 0;
        padding: 0.4rem;
    }

    @media (max-width: 780px) {
        .pub-stats {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 480px) {
        .pub-stats {
            grid-template-columns: 1fr;
        }

        .paper-header {
            flex-direction: column;
        }
    }
</style>

<section>
    <div class="blog-matrix-head">
        <div>
            <p>Academic Impact</p>
            <h2>Research &amp; Publications</h2>
        </div>
        <span class="blog-matrix-status" id="researchStatus">Feed status: syncing...</span>
    </div>
    <p class="research-subtitle">H-Index and citations per Google Scholar; publication record pulled live from ORCID.</p>

    <div class="profile-links">
        <a class="profile-link" href="https://scholar.google.com/citations?user=2zycrawAAAAJ&hl=en" target="_blank" rel="noopener noreferrer"><i class="fa fa-graduation-cap"></i> Google Scholar</a>
        <a class="profile-link" href="https://orcid.org/0009-0004-6491-8950" target="_blank" rel="noopener noreferrer"><i class="fa fa-id-card"></i> ORCID</a>
        <a class="profile-link" href="https://www.semanticscholar.org/author/2385183618" target="_blank" rel="noopener noreferrer"><i class="fa fa-book"></i> Semantic Scholar</a>
    </div>

    <div class="pub-stats">
        <article class="pub-stat">
            <p class="value">1</p>
            <p class="label">H-Index</p>
            <p class="stat-source" id="statHIndex">Semantic Scholar: --</p>
        </article>
        <article class="pub-stat">
            <p class="value">1</p>
            <p class="label">Citations</p>
            <p class="stat-source" id="statCitations">Semantic Scholar: --</p>
        </article>
        <article class="pub-stat">
            <p class="value" id="statPapers">--</p>
            <p class="label">Published Works</p>
        </article>
        <article class="pub-stat">
            <p class="value" id="statStatus">--</p>
            <p class="label">Monitor Status</p>
        </article>
    </div>

    <div class="paper-list" id="paperList">
        <p class="research-loading">Loading publications from ORCID...</p>
    </div>
</section>

<script>
    (function () {
        var ORCID_ID = "0009-0004-6491-8950";
        var S2_AUTHOR_ID = "2385183618";

        var statusEl = document.getElementById("researchStatus");
        var statStatusEl = document.getElementById("statStatus");
        var hIndexEl = document.getElementById("statHIndex");
        var citationsEl = document.getElementById("statCitations");
        var papersEl = document.getElementById("statPapers");
        var listEl = document.getElementById("paperList");

        var TYPE_LABELS = {
            "journal-article": "Journal Article",
            "conference-paper": "Conference Paper",
            "conference-abstract": "Conference Abstract",
            "book": "Book",
            "book-chapter": "Book Chapter",
            "preprint": "Preprint",
            "dissertation-thesis": "Thesis",
            "working-paper": "Working Paper",
            "report": "Report"
        };

        // Journal-level facts (quartile, impact factor) are not available from any
        // free live API - Clarivate JCR is paywalled. Keyed by lowercased journal
        // name so any future paper in the same venue picks it up automatically.
        // Source: Scimago Journal Rank / Clarivate JCR, checked manually.
        var JOURNAL_METRICS = {
            "expert systems with applications": { quartile: "Q1", impactFactor: 10.48 }
        };

        function journalRank(venue) {
            if (!venue) return null;
            return JOURNAL_METRICS[venue.trim().toLowerCase()] || null;
        }

        function typeLabel(type) {
            if (TYPE_LABELS[type]) return TYPE_LABELS[type];
            if (!type) return "Publication";
            return type.replace(/-/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
        }

        function escapeHtml(str) {
            var div = document.createElement("div");
            div.textContent = str || "";
            return div.innerHTML;
        }

        function renderPapers(works, citationsByDoi) {
            if (!works.length) {
                listEl.innerHTML = '<p class="research-empty">No publications on record yet.</p>';
                return;
            }
            listEl.innerHTML = "";
            works.forEach(function (w) {
                var cites = w.doi && citationsByDoi[w.doi] !== undefined ? citationsByDoi[w.doi] : null;
                var titleHtml = w.url
                    ? '<a href="' + w.url + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(w.title) + '</a>'
                    : escapeHtml(w.title);
                var metaLeft = [w.venue, w.year].filter(Boolean).join(" &middot; ");
                var metaRight = cites !== null ? "Cited " + cites + "x" : "";
                var rank = journalRank(w.venue);
                var rankHtml = rank
                    ? '<span class="journal-rank ' + rank.quartile.toLowerCase() + '">' + rank.quartile + " &middot; Impact Factor " + rank.impactFactor + "</span>"
                    : "";

                var item = document.createElement("article");
                item.className = "paper-item";
                item.innerHTML =
                    '<div class="paper-header">' +
                        '<h3 class="paper-title">' + titleHtml + "</h3>" +
                        '<span class="paper-badge">' + escapeHtml(typeLabel(w.type)) + "</span>" +
                    "</div>" +
                    '<div class="paper-meta"><span>' + metaLeft + "</span><span>" + metaRight + "</span></div>" +
                    rankHtml;
                listEl.appendChild(item);
            });
        }

        function setFallback() {
            statusEl.textContent = "Offline // showing cached record";
            statStatusEl.textContent = "Offline";
            hIndexEl.textContent = "Semantic Scholar: 0";
            citationsEl.textContent = "Semantic Scholar: 0";
            papersEl.textContent = "1";
            renderPapers([{
                title: "Breast mass classification in 3D ABUS based on Laplace-Beltrami spectra and dual path CNN",
                venue: "Expert Systems with Applications",
                year: 2025,
                type: "journal-article",
                doi: "10.1016/j.eswa.2025.129973",
                url: "https://doi.org/10.1016/j.eswa.2025.129973"
            }], {});
        }

        Promise.all([
            fetch("https://pub.orcid.org/v3.0/" + ORCID_ID + "/works", { headers: { Accept: "application/json" } })
                .then(function (res) { return res.ok ? res.json() : null; }),
            fetch("https://api.semanticscholar.org/graph/v1/author/" + S2_AUTHOR_ID + "?fields=hIndex,paperCount,citationCount,papers.citationCount,papers.externalIds")
                .then(function (res) { return res.ok ? res.json() : null; })
        ]).then(function (results) {
            var orcid = results[0];
            var s2 = results[1];

            var groups = (orcid && Array.isArray(orcid.group)) ? orcid.group : [];
            var works = groups.map(function (g) {
                var summary = g["work-summary"] && g["work-summary"][0];
                if (!summary) return null;
                var doi = null;
                var ids = (summary["external-ids"] && summary["external-ids"]["external-id"]) || [];
                for (var i = 0; i < ids.length; i += 1) {
                    if (ids[i]["external-id-type"] === "doi") doi = ids[i]["external-id-value"];
                }
                return {
                    title: (summary.title && summary.title.title) ? summary.title.title.value : "Untitled",
                    venue: summary["journal-title"] ? summary["journal-title"].value : "",
                    year: (summary["publication-date"] && summary["publication-date"].year) ? summary["publication-date"].year.value : null,
                    type: summary.type,
                    doi: doi ? doi.toLowerCase() : null,
                    url: summary.url ? summary.url.value : (doi ? "https://doi.org/" + doi : null)
                };
            }).filter(Boolean).sort(function (a, b) { return (b.year || 0) - (a.year || 0); });

            if (!works.length) throw new Error("no works on record");

            var citationsByDoi = {};
            if (s2 && Array.isArray(s2.papers)) {
                s2.papers.forEach(function (p) {
                    var pDoi = p.externalIds && p.externalIds.DOI;
                    if (pDoi) citationsByDoi[pDoi.toLowerCase()] = p.citationCount;
                });
            }

            hIndexEl.textContent = "Semantic Scholar: " + ((s2 && typeof s2.hIndex === "number") ? s2.hIndex : "--");
            citationsEl.textContent = "Semantic Scholar: " + ((s2 && typeof s2.citationCount === "number") ? s2.citationCount : "--");
            papersEl.textContent = String(works.length);
            statusEl.textContent = "Live // ORCID + Semantic Scholar";
            statStatusEl.textContent = "Live";
            renderPapers(works, citationsByDoi);
        }).catch(function () {
            setFallback();
        });
    })();
</script>
