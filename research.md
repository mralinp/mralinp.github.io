---
layout: main
title: Research
permalink: /research/
brief: "Medical image analysis, 3D ultrasound and representation learning: publications, current work and academic service."
---
{%- assign pubs = site.data.publications -%}
<div class="wrap narrow">
  <h1 class="page-title">Research</h1>
  <p class="page-lede">If we knew what it was we were doing, it would not be called research. — Albert Einstein</p>
  <div style="margin:24px 0 16px">{% include stats.html only="academic" %}</div>
  <section class="program">
    <h2 class="program-title">3D breast ultrasound: telling malignant lesions from benign</h2>
    <p>Automated 3D breast ultrasound (ABUS) images the whole breast in one sweep, but reading the volumes is slow and separating malignant from benign lesions is hard. My research asks what the <em>shape</em> of a lesion's surface says about it.</p>
    <ol class="chain">
      <li><b>Thesis.</b> MSc at IUST (2023): classification of breast cancer lesions in 3D-ABUS images, supervised by Dr. Mohsen Soryani and Dr. Ehsan Kozegar.</li>
      <li><b>Published method.</b> Laplace–Beltrami spectra of the lesion surface, fed to a dual-path CNN: <strong>AUC 0.935, accuracy 84.3%</strong> on the official TDSC-ABUS test split (70 lesions), as reported in the paper. <a href="https://doi.org/10.1016/j.eswa.2025.129973">Paper (DOI)</a></li>
      <li><b>Current work.</b> Why the spectrum works: the mathematics of Laplace–Beltrami shape representation, and better descriptors derived from it. It starts from a fair benchmark: classical baselines reproduced under one protocol (Tan et al. 2012 reaches AUC 0.86 on the same split). <a href="/abus-classification/">Project page</a> · <a href="https://github.com/mralinp/abus-classification">Code</a></li>
      <li><b>Tooling.</b> <a href="/blog/abus-classification/medical-imaging/pytorch/python/2026/09/23/tdsc-abus2023-pytorch.html">tdsc-abus2023-pytorch</a>, the dataset loader everything above runs on: <a href="https://github.com/mralinp/tdsc-abus2023-pytorch">GitHub</a> · <a href="https://pypi.org/project/tdsc-abus2023-pytorch/">PyPI</a></li>
      <li><b>Write-ups.</b> <a href="/project/abus-classification/medical-imaging/ultrasound/mammography/breast-cancer/2026/09/16/abus-classification-1-imaging-modalities.html">Part 1: mammography, ultrasound, and why 3D</a></li>
    </ol>
    <p class="meta">A caveat that matters: the reproduced baselines use the ground-truth lesion masks, while challenge entries had to find and segment lesions themselves (the winner reached AUC 0.889), so those numbers are not directly comparable. Running every method on predicted segmentations is part of the current work.</p>
  </section>

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
