---
layout: main
title: Notes
permalink: /notes/
published: false
---

Short notes and personal knowledge base.

{% assign sorted_notes = site.notes | sort: "date" | reverse %}
{% if sorted_notes.size > 0 %}
{% for note in sorted_notes %}

- [{{ note.title }}]({{ note.url }}) - {{ note.date | date: "%-d %B %Y" }}
{% if note.brief %}
{{ note.brief }}
{% endif %}

{% endfor %}
{% else %}
No notes published yet.
{% endif %}
