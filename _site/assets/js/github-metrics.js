(function () {
    var roots = document.querySelectorAll("[data-metrics-root]");
    if (!roots.length) return;

    function formatCount(value) {
        if (value >= 1000000) return (value / 1000000).toFixed(1).replace(".0", "") + "M";
        if (value >= 1000) return (value / 1000).toFixed(1).replace(".0", "") + "K";
        return String(value);
    }

    function paintGrid(grid, contributions) {
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

    function fetchAllRepos(username) {
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

    roots.forEach(function (root) {
        var username = root.getAttribute("data-username") || "mralinp";
        var grid = root.querySelector('[data-metric="grid"]');
        var reposEl = root.querySelector('[data-metric="repos"]');
        var starsEl = root.querySelector('[data-metric="stars"]');
        var commitsEl = root.querySelector('[data-metric="commits"]');
        var statusEl = root.querySelector('[data-metric="status"]');
        if (!grid) return;

        function setFallbackMetrics() {
            if (reposEl) reposEl.textContent = "55";
            if (starsEl) starsEl.textContent = "254";
            if (commitsEl) commitsEl.textContent = "226";
            if (statusEl) statusEl.textContent = "Fallback";
        }

        Promise.all([
            fetch("https://api.github.com/users/" + username).then(function (res) { return res.ok ? res.json() : null; }),
            fetchAllRepos(username),
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

            paintGrid(grid, contributions && Array.isArray(contributions.contributions) ? contributions.contributions : []);
            if (statusEl) statusEl.textContent = "Live // GitHub";
        }).catch(function () {
            setFallbackMetrics();
            paintGrid(grid, []);
        });
    });
})();
