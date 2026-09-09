---
title: Books library
layout: main
---

<section>
    {% include page-header.html kicker="Reading Archive" title="Library Modules" %}
    {% assign books = site.posts | where_exp: "post", "post.categories.first == 'book'" %}
    {% include card-grid.html posts=books variant="book" grid_id="libraryGrid" %}
</section>
