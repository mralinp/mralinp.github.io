---
title: Projects
layout: main
---

<section>
    {% include github-metrics.html subtitle="Live telemetry from your GitHub profile and yearly contribution channel." %}
    {% include page-header.html kicker="Portfolio Matrix" title="Project Deployments" %}
    {% assign projects = site.posts | where_exp: "post", "post.categories.first == 'project'" %}
    {% include card-grid.html posts=projects variant="project" grid_id="projectsGrid" %}
</section>
