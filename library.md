---
title: Books library
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
        background: #22c55e;
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

    .blog-card p {
        color: #94a3b8;
        margin: 0 0 0.65rem;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .book-rating {
        display: flex;
        align-items: center;
        gap: 0.12rem;
        margin: 0.1rem 0 0.55rem;
        color: #f59e0b;
        font-size: 0.7rem;
    }

    .book-rating .book-star-dim {
        opacity: 0.25;
    }

    .book-rating .book-rating-value {
        margin-left: 0.3rem;
        color: #9ca3af;
        font: 500 0.54rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .book-review-snippet {
        color: #94a3b8;
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
        display: block;
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
</style>

<section>
    <div class="blog-matrix-head">
        <div>
            <p>Reading Archive</p>
            <h2>Library Modules</h2>
        </div>
        <span class="blog-matrix-status">Feed status: live</span>
    </div>

    {% assign books = site.posts | where_exp: "post", "post.categories.first == 'book'" %}
    <div class="row" id="libraryGrid">
        {% for post in books %}
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
                    <span class="blog-card-badge">{{ post.categories[0] | default: "Book" | upcase }}</span>
                    <span class="blog-card-open material-symbols-outlined">north_east</span>
                </div>
                <div class="blog-card-content">
                    <div class="blog-card-top">
                        <h3 class="blog-card-title"><a href="{{ post.url }}" title="{{ post.title }}">{{ post.title | strip_html | truncate: 62 }}</a></h3>
                        <span class="blog-card-date">{{ post.date | date: "%b %Y" }}</span>
                    </div>
                    {% assign rating = post.book_rating | default: 0 | plus: 0 %}
                    <div class="book-rating" aria-label="Book rating">
                        {% for i in (1..5) %}
                        <i class="fa fa-star{% if i > rating %} book-star-dim{% endif %}"></i>
                        {% endfor %}
                        <span class="book-rating-value">{{ rating }}/5</span>
                    </div>
                    <p class="book-review-snippet">{{ post.book_review | default: post.brief | default: post.excerpt | strip_html | truncate: 130 }}</p>
                    <div class="blog-card-tags">
                        {% for category in post.categories limit: 3 %}
                        <span>{{ category }}</span>
                        {% endfor %}
                        {% if post.book_resources %}
                        <span>resources: {{ post.book_resources | size }}</span>
                        {% endif %}
                    </div>
                    <div class="blog-card-actions">
                        <a href="{{ post.url }}" class="blog-card-btn">Read review</a>
                    </div>
                </div>
            </article>
        </div>
        {% endfor %}
    </div>

    <nav aria-label="Page navigation" id="PageNav">
        <ul class="pagination justify-content-center" id="libraryPager"></ul>
    </nav>
</section>

<script>
    (function () {
        var perPage = 6;
        var items = Array.prototype.slice.call(document.querySelectorAll("#libraryGrid .js-page-item"));
        var pager = document.getElementById("libraryPager");
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