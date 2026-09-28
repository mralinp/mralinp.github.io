---
layout: main
title: Projects
permalink: /projects/
brief: "Case studies and projects: authentication systems, production platforms, 3D medical imaging research and open-source tools, with how they work and the evidence."
---
{%- assign projects = site.posts | where_exp: "p", "p.categories.first == 'project'" -%}
{%- assign selected = projects | where: "featured", true -%}
{%- assign others = projects | where_exp: "p", "p.featured != true" -%}
<div class="wrap">
  <h1 class="page-title">Projects</h1>
  <p class="page-lede">What I built, how it works, and the evidence.</p>
  <div class="section-head"><h2>Selected work</h2></div>
  <div class="grid">{% for p in selected %}{% include card.html post=p %}{% endfor %}</div>
  <section>
    <div class="section-head"><h2>Open source</h2></div>
    {% include stats.html only="github" %}
  </section>
  <section>
    <div class="section-head"><h2>More projects</h2></div>
    <div class="grid">{% for p in others %}{% include card.html post=p %}{% endfor %}</div>
  </section>
</div>
