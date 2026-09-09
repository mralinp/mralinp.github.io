---
title: Research
layout: main
---

<section class="research-page">
    {% include page-header.html kicker="Academic Impact" title="Research & Publications" status="Feed status: syncing..." status_id="researchStatus" %}
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
                var cites = w.doi && citationsByDoi[w.doi] !== undefined ? citationsByDoi[w.doi] : 1;
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
                title: "Breast mass classification in 3D ABUS based on Laplace-Beltrami spectra and Dual Path CNN",
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
