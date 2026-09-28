---
layout: main
title: Library
permalink: /library/
brief: "Books read, and what stuck."
---
{%- assign books = site.posts | where_exp: "p", "p.categories.first == 'book'" -%}
<div class="wrap">
  <h1 class="page-title">Library</h1>
  <p class="page-lede">Books read, and what stuck.</p>
  <div class="grid books">{% for p in books %}{% include card.html post=p %}{% endfor %}</div>
</div>
