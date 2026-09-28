---
layout: main
title: Research
permalink: /research/
brief: "Medical image analysis, 3D ultrasound and representation learning: publications, current work and academic service."
---
{%- assign pubs = site.data.publications -%}
<div class="wrap narrow">
  <h1 class="page-title">Research</h1>
  <p class="page-lede">Medical image analysis, 3D ultrasound, and representation learning.</p>
  <div style="margin:24px 0 16px">{% include stats.html only="academic" %}</div>
  {%- assign ongoing = pubs | where: "status", "ongoing" %}
  {%- if ongoing.size > 0 %}
  <div class="year">Current work</div>
  {% for p in ongoing %}{% include publication.html pub=p %}{% endfor %}
  {%- endif %}
  {%- assign done = pubs | where_exp: "p", "p.status != 'ongoing'" | group_by: "year" %}
  {%- for y in done %}
  <div class="year">{{ y.name }}</div>
  {% for p in y.items %}{% include publication.html pub=p %}{% endfor %}
  {%- endfor %}
  <h2 style="font-size:22px; margin:48px 0 8px">Service</h2>
  <ul class="timeline">
    {%- for s in site.data.service %}
    <li><span class="when">{{ s.year }}</span><div><div class="what">{{ s.what }}</div><div class="where">{{ s.where }}</div></div></li>
    {%- endfor %}
  </ul>
</div>
