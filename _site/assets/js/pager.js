(function () {
    var grids = document.querySelectorAll("[data-pager-grid]");
    if (!grids.length) return;

    function pageLink(label, page, disabled, active, onClick) {
        var li = document.createElement("li");
        li.className = "page-item" + (disabled ? " disabled" : "") + (active ? " active" : "");
        var a = document.createElement(active ? "span" : "a");
        a.className = "page-link";
        a.textContent = label;
        if (!disabled && !active) {
            a.href = "#";
            a.addEventListener("click", function (e) {
                e.preventDefault();
                onClick(page);
            });
        }
        li.appendChild(a);
        return li;
    }

    grids.forEach(function (grid) {
        var perPage = parseInt(grid.getAttribute("data-per-page"), 10) || 6;
        var items = Array.prototype.slice.call(grid.querySelectorAll(".js-page-item"));
        var pager = document.querySelector('[data-pager-for="' + grid.id + '"]');
        if (!items.length || !pager) return;

        var totalPages = Math.ceil(items.length / perPage);
        var current = 1;

        function renderPager() {
            pager.innerHTML = "";
            pager.appendChild(pageLink("Previous", current - 1, current === 1, false, onPage));
            for (var p = 1; p <= totalPages; p += 1) {
                pager.appendChild(pageLink(String(p), p, false, p === current, onPage));
            }
            pager.appendChild(pageLink("Next", current + 1, current === totalPages, false, onPage));
        }

        function renderPage(page, scroll) {
            current = Math.max(1, Math.min(totalPages, page));
            var start = (current - 1) * perPage;
            var end = start + perPage;
            for (var i = 0; i < items.length; i += 1) {
                items[i].style.display = i >= start && i < end ? "" : "none";
            }
            renderPager();
            if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
        }

        function onPage(page) {
            renderPage(page, true);
        }

        renderPage(1, false);
    });
})();
