---
layout: main
---

<style>
    #content {
        background-color: #06070f;
    }

    .post-shell {
        display: grid;
        gap: 0.9rem;
    }

    .post-head {
        border: 1px solid #1a1a1a;
        background: rgba(5, 5, 5, 0.78);
        padding: 1rem;
    }

    .post-kicker {
        margin: 0;
        color: #ff5a00;
        font: 500 0.66rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    .post-head h1 {
        margin: 0.35rem 0 0.5rem;
        font-size: clamp(1.3rem, 3vw, 2rem);
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .post-brief {
        margin: 0 0 0.7rem;
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .post-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .post-chip {
        border: 1px solid #2b2b2b;
        background: rgba(0, 0, 0, 0.35);
        color: #cbd5e1;
        padding: 0.18rem 0.48rem;
        font: 500 0.6rem/1.2 "JetBrains Mono", monospace;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .post-media {
        border: 1px solid #1a1a1a;
        background: rgba(5, 5, 5, 0.75);
        overflow: hidden;
    }

    .post-media img {
        width: 100%;
        display: block;
    }

    .post-body {
        border: 1px solid #1a1a1a;
        background: rgba(5, 5, 5, 0.82);
        padding: 1rem;
    }

    .post-content {
        background: transparent;
        border: 0;
        padding: 0;
        max-width: 100%;
    }

    .post-nav {
        display: flex;
        justify-content: space-between;
        gap: 0.6rem;
        margin-bottom: 0.9rem;
    }

    .post-nav a {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        text-decoration: none;
        border: 1px solid #2b2b2b;
        color: #cbd5e1;
        padding: 0.3rem 0.55rem;
        font: 600 0.6rem/1 "JetBrains Mono", monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .post-nav a:hover {
        border-color: #ff5a00;
        color: #ff5a00;
    }

    .book-panel {
        margin-bottom: 0.95rem;
        border: 1px solid #2b2b2b;
        background: rgba(0, 0, 0, 0.35);
        padding: 0.8rem;
    }

    .book-panel h2 {
        margin: 0 0 0.55rem;
        color: #f8fafc;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.76rem;
    }

    .book-rating {
        display: flex;
        align-items: center;
        gap: 0.18rem;
        color: #f59e0b;
        margin-bottom: 0.45rem;
    }

    .book-rating .book-star-dim {
        opacity: 0.25;
    }

    .book-rating .book-rating-value {
        margin-left: 0.35rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font: 500 0.58rem/1 "JetBrains Mono", monospace;
    }

    .book-review {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .book-resources {
        margin: 0;
        padding-left: 1rem;
    }

    .book-resources li {
        margin-bottom: 0.35rem;
        color: #cbd5e1;
    }

    .book-resources a {
        color: #ff8a44;
    }

    .book-resources a:hover {
        color: #ff5a00;
    }

    .post-comments {
        border: 1px solid #1a1a1a;
        background: rgba(5, 5, 5, 0.78);
        padding: 0.9rem;
    }

    .post-comments h2 {
        margin: 0 0 0.75rem;
        color: #f8fafc;
        font-size: 0.9rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    @media (max-width: 720px) {
        .post-head,
        .post-body,
        .post-comments {
            padding: 0.8rem;
        }
    }
</style>

<article class="post-shell">
    {% assign back_url = "/blog" %}
    {% assign back_label = "Back to blog" %}
    {% if page.categories contains "book" %}
        {% assign back_url = "/library" %}
        {% assign back_label = "Back to library" %}
    {% elsif page.categories contains "project" %}
        {% assign back_url = "/projects" %}
        {% assign back_label = "Back to projects" %}
    {% endif %}

    <section class="post-head">
        <p class="post-kicker">Article Dispatch</p>
        <h1>{{ page.title }}</h1>
        {% if page.brief %}
        <p class="post-brief">{{ page.brief }}</p>
        {% endif %}
        <div class="post-chip-row">
            <span class="post-chip"><i class="fa fa-user-circle-o"></i> {{ page.author }}</span>
            <span class="post-chip"><i class="fa fa-calendar-o"></i> {{ page.date | date: "%-d %b %Y" }}</span>
            {% for cat in page.categories %}
            <span class="post-chip"><i class="fa fa-tag"></i> {{ cat }}</span>
            {% endfor %}
        </div>
    </section>

    {% if page.img %}
    <div class="post-media">
        <a class="image-zoom cboxElement" href="{{ page.img }}">
        <img src="{{ page.img }}" alt="{{ page.title | escape }}">
        <div class="image-overlay"></div>
        </a>
    </div>
    {% endif %}

    <section class="post-body">
        <div class="post-nav">
            <a href="{{ back_url }}"><span class="material-symbols-outlined">arrow_back</span> {{ back_label }}</a>
            <a href="{{ page.url }}">Permalink <span class="material-symbols-outlined">north_east</span></a>
        </div>

        {% if page.categories contains "book" %}
        <div class="book-panel">
            <h2>Book Review Snapshot</h2>
            {% assign rating = page.book_rating | default: 0 | plus: 0 %}
            <div class="book-rating" aria-label="Book rating">
                {% for i in (1..5) %}
                <i class="fa fa-star{% if i > rating %} book-star-dim{% endif %}"></i>
                {% endfor %}
                <span class="book-rating-value">{{ rating }}/5</span>
            </div>
            {% if page.book_review %}
            <p class="book-review">{{ page.book_review }}</p>
            {% else %}
            <p class="book-review">{{ page.brief }}</p>
            {% endif %}
        </div>

        {% if page.book_resources %}
        <div class="book-panel">
            <h2>Related Files and Notes</h2>
            <ul class="book-resources">
                {% for resource in page.book_resources %}
                <li>
                    <a href="{{ resource.url }}" target="_blank" rel="noopener noreferrer">{{ resource.title }}</a>
                    {% if resource.type %} <span class="text-secondary">({{ resource.type }})</span>{% endif %}
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        {% endif %}

        <div class="post-content">
            {{ page.content }}
        </div>
    </section>

    <section class="post-comments">
        <h2>Comment Channel</h2>
        <div class="commentbox"></div>
    </section>
</article>

<script src="https://unpkg.com/commentbox.io/dist/commentBox.min.js"></script>
<script>
    function arrToHex(arr) {
        let s = '#';
        for (let i = 0; i < arr.length; i++) {
            s += ((arr[i] / 16) | 0).toString(16);
            s += ((arr[i] % 16) | 0).toString(16);
        }
        return s;
    }
    var element = document.body;
    var style;
    var color = "black";
    if (window.getComputedStyle) {
        style = window.getComputedStyle(element);
    } else {
        style = element.currentStyle;
    }
    if (!style) {
        // ...seriously old browser...
    } else {
        color = style.color;
    }
    rgb = color.match(/\d+/g);
    commentBox('5076933226266624-proj', {
        textColor: arrToHex(rgb),
        subtextColor: arrToHex(rgb),
    })
</script>
