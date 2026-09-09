---
layout: main
title: Home
---

<section id="hero" class="d-flex flex-column justify-content-center">
    <div class="container" data-aos="zoom-in" data-aos-delay="100">
      <h1>Ali Naderi Parizi</h1>
      <p>I'm <span class="typed" data-typed-items="SoftWare Engineer, Developer, Hacker, a Free Geek"></span></p>
    </div>
</section>

{% include github-metrics.html %}

<section class="publication-monitor" data-aos="fade-up" data-aos-delay="180">
    <span class="material-symbols-outlined metrics-bg-icon">library_books</span>
    <div class="pub-shell">
        <div>
            <p class="pub-label">Academic Impact</p>
            <h2 class="pub-title">Publication Impact Monitor</h2>
            <p class="pub-subtitle">H-Index and citations per <a href="https://scholar.google.com/citations?user=2zycrawAAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Google Scholar</a>; publication record pulled live from ORCID.</p>
            <div class="pub-stats">
                <article class="pub-stat">
                    <p class="value">1</p>
                    <p class="label">H-Index</p>
                </article>
                <article class="pub-stat">
                    <p class="value">1</p>
                    <p class="label">Citations</p>
                </article>
                <article class="pub-stat">
                    <p class="value" id="homeStatPapers">--</p>
                    <p class="label">Published Works</p>
                </article>
                <article class="pub-stat">
                    <p class="value" id="homePubStatus">--</p>
                    <p class="label">Monitor Status</p>
                </article>
            </div>
        </div>

        <div>
            <div class="paper-list" id="homePaperList">
                <p class="pub-subtitle" style="margin:0;">Loading publications...</p>
            </div>
            <p class="pub-subtitle" style="margin-top: 0.7rem;">
                <a href="/research">View full research record &rarr;</a>
                &nbsp;&middot;&nbsp;
                <a href="https://scholar.google.com/citations?user=2zycrawAAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Google Scholar</a>
            </p>
        </div>
    </div>
</section>

<script>
    (function () {
        var ORCID_ID = "0009-0004-6491-8950";
        var MAX_PREVIEW = 3;

        var statusEl = document.getElementById("homePubStatus");
        var papersEl = document.getElementById("homeStatPapers");
        var listEl = document.getElementById("homePaperList");

        function escapeHtml(str) {
            var div = document.createElement("div");
            div.textContent = str || "";
            return div.innerHTML;
        }

        function renderPreview(works) {
            if (!works.length) {
                listEl.innerHTML = '<p class="pub-subtitle" style="margin:0;">No publications on record yet.</p>';
                return;
            }
            listEl.innerHTML = "";
            works.slice(0, MAX_PREVIEW).forEach(function (w) {
                var titleHtml = w.url
                    ? '<a href="' + w.url + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(w.title) + '</a>'
                    : escapeHtml(w.title);
                var meta = [w.venue, w.year].filter(Boolean).join(" &middot; ");
                var item = document.createElement("article");
                item.className = "paper-item";
                item.innerHTML =
                    '<div class="paper-header"><h3 class="paper-title" style="color:#f3f4f6;">' + titleHtml + "</h3></div>" +
                    '<div class="paper-meta"><span>' + meta + "</span></div>";
                listEl.appendChild(item);
            });
        }

        function setFallback() {
            statusEl.textContent = "Offline";
            papersEl.textContent = "1";
            renderPreview([{
                title: "Breast mass classification in 3D ABUS based on Laplace-Beltrami spectra and Dual Path CNN",
                venue: "Expert Systems with Applications",
                year: 2025,
                url: "https://doi.org/10.1016/j.eswa.2025.129973"
            }]);
        }

        fetch("https://pub.orcid.org/v3.0/" + ORCID_ID + "/works", { headers: { Accept: "application/json" } })
            .then(function (res) { return res.ok ? res.json() : null; })
            .then(function (orcid) {
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
                    url: summary.url ? summary.url.value : (doi ? "https://doi.org/" + doi : null)
                };
            }).filter(Boolean).sort(function (a, b) { return (b.year || 0) - (a.year || 0); });

            if (!works.length) throw new Error("no works on record");

            papersEl.textContent = String(works.length);
            statusEl.textContent = "Live";
            renderPreview(works);
        }).catch(function () {
            setFallback();
        });
    })();
</script>
