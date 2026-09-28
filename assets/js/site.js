// The site's only script. Everything works without it; this adds the theme toggle, the phone menu,
// search, blog filters, the post's "On this page", copy buttons, and the hero quote cycling.
(function () {
  "use strict";
  var root = document.documentElement;
  var $ = function (s, el) { return (el || document).querySelector(s); };
  var $$ = function (s, el) { return [].slice.call((el || document).querySelectorAll(s)); };
  var reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- theme: follows the system until the visitor picks one; the pick is remembered ----
  var toggle = $("#theme");
  if (toggle) toggle.onclick = function () {
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.theme = root.dataset.theme; } catch (e) {}
  };

  // ---- phone menu ----
  var menu = $("#menu"), nav = $("#nav");
  if (menu) menu.onclick = function () {
    nav.classList.toggle("open");
    menu.setAttribute("aria-expanded", nav.classList.contains("open"));
  };

  // ---- hero epigraph: the first quote is rendered by Jekyll; clicking shows the next ----
  var ep = $("#epigraph");
  if (ep) {
    var quotes = JSON.parse(ep.dataset.quotes || "[]"), qi = 0;
    ep.onclick = function () {
      if (!quotes.length) return;
      qi = (qi + 1) % quotes.length;
      $(".q", ep).textContent = quotes[qi].text;
      $("cite", ep).textContent = quotes[qi].by;
    };
  }

  // ---- blog filter: chips toggle rows by tag; with JS off every post shows ----
  $$(".filter").forEach(function (group) {
    group.addEventListener("click", function (e) {
      var chip = e.target.closest(".chip");
      if (!chip) return;
      $$(".chip", group).forEach(function (c) { c.setAttribute("aria-pressed", c === chip); });
      var tag = chip.dataset.tag;
      $$("[data-tags]").forEach(function (row) {
        row.hidden = tag !== "all" && (" " + row.dataset.tags + " ").indexOf(" " + tag + " ") < 0;
      });
      $$(".year-group").forEach(function (g) { g.hidden = !$$("[data-tags]", g).some(function (r) { return !r.hidden; }); });
    });
  });

  // ---- code blocks: a copy button on each ----
  $$("div.highlighter-rouge").forEach(function (b) {
    var btn = document.createElement("button");
    btn.className = "copy"; btn.type = "button"; btn.textContent = "Copy";
    btn.onclick = function () {
      navigator.clipboard.writeText($("code", b).innerText).then(function () {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = "Copy"; }, 1500);
      });
    };
    b.appendChild(btn);
  });

  // ---- On this page: a sticky rail on wide screens, a pinned bar on narrow ones ----
  var art = $(".post-layout .prose"), rail = $(".toc"), bar = $(".toc-bar");
  if (art && rail && bar) {
    var heads = $$("h1, h2, h3", art);
    if (heads.length < 3) { rail.remove(); bar.remove(); }
    else {
      var links = heads.map(function (h, i) {
        h.id = h.id || "s" + i;
        return '<a href="#' + h.id + '" data-to="' + h.id + '"' + (h.tagName === "H3" ? ' class="sub"' : "") + ">" +
          h.textContent.replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }) + "</a>";
      }).join("");
      rail.innerHTML = '<p>On this page</p><div class="progress"><i></i></div>' + links;
      $(".links", bar).innerHTML = links;
      var go = function (e) {
        var a = e.target.closest("a[data-to]");
        if (!a) return;
        e.preventDefault(); bar.open = false;
        document.getElementById(a.dataset.to).scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
        history.replaceState(null, "", "#" + a.dataset.to);
      };
      rail.onclick = go; $(".links", bar).onclick = go;
      var ticking = false;
      var spy = function () {
        ticking = false;
        var line = innerHeight * 0.25, cur = heads[0];
        heads.forEach(function (h) { if (h.getBoundingClientRect().top < line) cur = h; });
        $$("[data-to]").forEach(function (a) { a.classList.toggle("on", a.dataset.to === cur.id); });
        $(".cur", bar).textContent = cur.textContent;
        var r = art.getBoundingClientRect(), pct = Math.min(1, Math.max(0, (line - r.top) / r.height));
        $$(".progress i").forEach(function (i) { i.style.width = pct * 100 + "%"; });
        var on = $("a.on", rail);
        if (on && rail.scrollHeight > rail.clientHeight) on.scrollIntoView({ block: "nearest" });
      };
      addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(spy); } }, { passive: true });
      spy();
    }
  }

  // ---- comments: commentbox.io is only loaded when the reader asks for it ----
  var cbtn = $("#loadComments");
  if (cbtn) cbtn.onclick = function () {
    cbtn.disabled = true; cbtn.textContent = "Loading…";
    var s = document.createElement("script");
    s.src = "https://unpkg.com/commentbox.io/dist/commentBox.min.js";
    s.onload = function () {
      cbtn.remove();
      var c = getComputedStyle(document.body).color;
      window.commentBox(cbtn.dataset.project, { textColor: c, subtextColor: c });
    };
    s.onerror = function () { cbtn.disabled = false; cbtn.textContent = "Couldn't load comments. Try again"; };
    document.head.appendChild(s);
  };

  // ---- search: /search.json is built by Jekyll and fetched on first open ----
  var dlg = $("#search"), q = $("#q"), hits = $("#hits");
  if (dlg) {
    var index = null, sel = 0, shown = [];
    var esc = function (t) { return String(t || "").replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
    var mark = function (text, terms) {
      var h = esc(text);
      terms.forEach(function (w) { h = h.replace(new RegExp("(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>"); });
      return h;
    };
    var fmt = function (d) { return d ? new Date(d + "T12:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : ""; };
    var render = function () {
      var terms = q.value.toLowerCase().split(/\s+/).filter(Boolean);
      shown = !terms.length ? index.slice(0, 6) : index.map(function (e) {
        var t = e.t.toLowerCase(), g = (e.tags || "").toLowerCase(), d = (e.d || "").toLowerCase(), s = 0;
        for (var i = 0; i < terms.length; i++) {
          var w = terms[i];
          if (t.indexOf(w) >= 0) s += 10; else if (g.indexOf(w) >= 0) s += 5; else if (d.indexOf(w) >= 0) s += 2; else return null;
        }
        return { e: e, s: s };
      }).filter(Boolean).sort(function (a, b) { return b.s - a.s || ((b.e.date || "") > (a.e.date || "") ? 1 : -1); })
        .map(function (x) { return x.e; }).slice(0, 8);
      sel = 0;
      hits.innerHTML = shown.length ? shown.map(function (e, i) {
        return '<li role="option" id="hit' + i + '" aria-selected="' + (i === 0) + '"><a href="' + esc(e.u) + '">' +
          (e.img ? '<img src="' + esc(e.img) + '" alt="" loading="lazy">' : "<span></span>") +
          '<div><div class="k">' + esc([e.kind, (e.tags || "").split(" ").slice(0, 2).join(" · "), fmt(e.date)].filter(Boolean).join(" · ")) +
          '</div><div class="t">' + mark(e.t, terms) + '</div><div class="d">' + mark(e.d, terms) + "</div></div></a></li>";
      }).join("") : '<li class="empty"><b>No hits.</b>Not even a B-side. Try another word.</li>';
      q.setAttribute("aria-activedescendant", shown.length ? "hit0" : "");
    };
    var move = function (n) {
      if (!shown.length) return;
      sel = (sel + n + shown.length) % shown.length;
      $$("li", hits).forEach(function (li, i) { li.setAttribute("aria-selected", i === sel); });
      q.setAttribute("aria-activedescendant", "hit" + sel);
      document.getElementById("hit" + sel).scrollIntoView({ block: "nearest" });
    };
    var open = function () {
      if (dlg.open) return;
      dlg.showModal(); q.value = ""; q.focus();
      if (index) return render();
      hits.innerHTML = '<li class="empty">Loading…</li>';
      fetch("/search.json").then(function (r) { return r.json(); }).then(function (data) { index = data; render(); })
        .catch(function () { hits.innerHTML = '<li class="empty"><b>Offline?</b>The search index didn\'t load.</li>'; });
    };
    $("#searchBtn").onclick = open;
    q.addEventListener("input", function () { if (index) render(); });
    q.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") { e.preventDefault(); if (shown[sel]) location.href = shown[sel].u; }
    });
    dlg.addEventListener("click", function (e) {
      if (e.target === dlg || e.target.closest("[data-close]")) dlg.close();
    });
    addEventListener("keydown", function (e) {
      var el = document.activeElement, typing = /INPUT|TEXTAREA|SELECT/.test(el.tagName) || el.isContentEditable;
      if ((e.key === "/" && !typing) || (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey))) { e.preventDefault(); open(); }
    });
  }

  console.log("%c// Reading the source? Good.\n// It's free software. Take it, change it, share it.\n// github.com/mralinp/mralinp.github.io",
    "font: 14px monospace; color: #e879f9");
})();
