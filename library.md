---
layout: main
title: Library
permalink: /library/
brief: "Not everything I read is here. Only what left a mark."
---
{%- assign books = site.posts | where_exp: "p", "p.categories.first == 'book'" -%}
<div class="wrap">
  <h1 class="page-title">Library</h1>
  <p class="page-lede">Not everything I read is here. Only what left a mark.</p>
  <div class="grid books">{% for p in books %}{% include card.html post=p %}{% endfor %}</div>
</div>
