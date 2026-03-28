(function () {
    "use strict";

    function normalizePath(path) {
        if (!path) return "/";
        return path.length > 1 ? path.replace(/\/+$/, "") : path;
    }

    function setActiveSidebarItem() {
        const items = document.querySelectorAll("#sidebar .sidebar-item");
        if (!items.length) return;

        const pagePath = normalizePath(window.location.pathname);
        items.forEach((item) => item.classList.remove("active"));

        let activeItem = null;
        items.forEach((item) => {
            const link = item.querySelector("a");
            if (!link) return;
            const href = normalizePath(link.getAttribute("href"));
            if (
                pagePath === href ||
                (href !== "/" && pagePath.startsWith(href + "/"))
            ) {
                activeItem = item;
            }
        });

        if (!activeItem && items[0]) activeItem = items[0];
        activeItem.classList.add("active");
    }

    function setupMobileSidebar() {
        const sidebar = document.getElementById("sidebar");
        const toggle = document.getElementById("mobileMenuToggle");
        if (!sidebar || !toggle) return;

        toggle.addEventListener("click", function () {
            sidebar.classList.toggle("open");
        });

        document.addEventListener("click", function (event) {
            const clickedInsideSidebar = sidebar.contains(event.target);
            const clickedToggle = toggle.contains(event.target);
            if (!clickedInsideSidebar && !clickedToggle) {
                sidebar.classList.remove("open");
            }
        });
    }

    function setupTypedAnimation() {
        const typedNode = document.querySelector(".typed");
        if (!typedNode || typeof Typed === "undefined") return;
        const typedItems = typedNode.getAttribute("data-typed-items");
        if (!typedItems) return;

        new Typed(".typed", {
            strings: typedItems.split(","),
            loop: true,
            typeSpeed: 100,
            backSpeed: 50,
            backDelay: 2000,
        });
    }

    setActiveSidebarItem();
    setupMobileSidebar();
    setupTypedAnimation();
})();
