---
layout: main
title: Notes
permalink: /notes/
published: false
---
{%- assign notes = site.notes | sort: "date" | reverse -%}
<div class="wrap narrow">
  <h1 class="page-title">Notes</h1>
  <p class="page-lede">Short notes and a personal knowledge base.</p>
  <ul class="rows notes-list">{% for n in notes %}{% include post-row.html post=n year=true %}{% endfor %}</ul>
</div>
