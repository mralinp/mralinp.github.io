---
layout: main
title: Projects
permalink: /projects/
brief: "Things built, shipped and documented: medical imaging, embedded hardware, games, security and tooling."
---
{%- assign projects = site.posts | where_exp: "p", "p.categories.first == 'project'" -%}
<div class="wrap">
  <h1 class="page-title">Projects</h1>
  <p class="page-lede">Built, shipped, documented.</p>
  <div style="margin-bottom:40px">{% include stats.html only="github" %}</div>
  <div class="grid">{% for p in projects %}{% include card.html post=p %}{% endfor %}</div>
</div>
