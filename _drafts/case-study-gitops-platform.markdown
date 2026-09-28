---
layout: post
title:  "Case Study: A Self-Service GitOps Platform for an AI Startup"
author: "Ali Naderi"
img: "/assets/images/posts/projects/gitops-platform/architecture.png"  # TODO: architecture diagram or a Grafana screenshot (no internal IPs in it)
date:   2026-09-28 12:00:00 +0330  # TODO: set the publish date
categories:  project platform kubernetes gitops devops
brief: "Designed and built the whole platform at SciNext in about a month: push to main, and your app is built, deployed, monitored and logged, with no tickets and no kubectl. k3s, Argo CD, Gitea Actions, Prometheus, Loki and Tempo."
---
{% comment %} Review before publishing:
     - The company is SciNext. Never mention the internal infrastructure domain or any
       internal IPs, hostnames, product URLs, secrets or colleagues' names: SciNext is still private.
       Everything below is architecture-level on purpose.
     - Replace every TODO with a real number or delete the sentence. No guesses. {% endcomment %}

At SciNext I designed and built the platform every product runs on. The goal was simple to state: an engineer pushes to `main`, and a few minutes later the new version is running, with metrics, logs and traces already flowing into one Grafana. No ticket to an admin, no `kubectl`, no copy-pasted YAML per app.

This is the problem, the design, the decisions behind it, and what it delivers.

# 1. The problem

> **TODO:** the situation before: how apps were deployed (by hand on VMs? docker compose?), how long a new service took to go live, what broke. Even rough numbers help ("a new service took a day of an admin's time").

We were a small team shipping several products at once: a chemistry database with a RAG assistant, an API gateway, an LLM-powered crawler, internal docs. Each needed hosting, a database, storage, TLS, monitoring. Doing that by hand per product doesn't scale, and it concentrates every deploy on whoever holds the keys.

# 2. Constraints

- **One physical server.** Everything runs as VMs on a single Proxmox host, so the design had to be lean and the blast radius of mistakes small.
- **Self-hosted end to end.** Git, CI, the container registry, dashboards: no SaaS dependency in the deploy path.
- **Engineers are not cluster admins.** Onboarding a new app must need zero platform knowledge beyond one template.
- **Secrets never go in git.**

# 3. Architecture

```text
 developer ── git push ──► Gitea (git + container registry)
                              │
                              ├─► Gitea Actions ── build & push image ──► registry
                              │        └─ write-only CI identity ── app Secret ──► app namespace
                              │
                              └─► Argo CD (on k3s) pulls git
                                     ├─ platform-infra          ingress, image updater, onboarding
                                     ├─ platform-observability  Prometheus, Grafana, Loki, Tempo, Alloy
                                     └─ every app repo          Deployment + Service + Ingress
                                                │
                   argocd-image-updater sees the new tag, commits it to the app repo,
                   Argo CD rolls the pods
```

Around the cluster, on their own VMs: a shared PostgreSQL/MongoDB server, MinIO for object storage, an nginx reverse proxy that terminates TLS behind a CDN, and a WireGuard VPN for reaching internal services from a laptop. Nightly VM snapshots cover backups.

## 3.1 The platform is itself GitOps

The cluster matches git. One bootstrap manifest is applied once; from then on Argo CD manages everything else, including itself, from the `platform-infra` repo (the app-of-apps pattern). A CI workflow can rebuild the whole thing on a fresh host, and because it's idempotent it doubles as a self-healing check on every push.

## 3.2 Apps onboard themselves

An Argo CD **ApplicationSet** watches every repo in each allow-listed Gitea organization. Any repo that contains `deploy/kustomization.yaml` becomes an Argo CD Application automatically, with its own image-updater scoped to it. Shipping a new service is: copy the template, fill in two names, push. Adding a whole new organization (team or project) is a small, reviewed change of two files.

## 3.3 Releases without a human in the loop

Gitea Actions builds an image tagged with the git SHA. `argocd-image-updater` notices it, **commits the new tag back into the app's repo**, and Argo CD rolls it out. Every deploy is therefore a git commit: auditable, and reverted with `git revert`.

## 3.4 One place to look

One Grafana for everyone. Logs need no app changes (Alloy tails every pod into Loki); metrics are one `ServiceMonitor` away; traces go to Tempo over OTLP, with a pre-wired endpoint in the app template. Logging a `traceID` gives click-through from a log line to its trace.

# 4. Security decisions

- **The CI identity can only write, never read.** Gitea Actions writes each app's settings into a Kubernetes Secret as a dedicated ServiceAccount allowed only `create` and `patch` on Secrets, only in onboarded namespaces. No read access, no cluster role. With read access, any repo's workflow could read every other app's secrets.
- **Protected namespaces.** The onboarding chart refuses to render into `kube-system`, `argocd` and other platform namespaces, so a badly named organization can't grant itself access there.
- **Per-app data isolation.** Every app gets its own database and user, and its own MinIO bucket and access key, scoped to that app alone. No shared superuser is handed out.
- **Secrets never touch argv, files or logs** in CI; multi-line values like PEM keys survive byte for byte.
- **Written-down trade-offs.** Namespaces are per organization, not per app, so one app's workflow could overwrite (not read) a sibling's Secret. That's documented as a known limit, with the fix (one namespace per app) described.

> **TODO:** anything you'd call out as the hardest bug or the most important decision you reversed.

# 5. Results

- **Built in about a month** (the platform repos' first commit to today) by one engineer: me. {% comment %} TODO: confirm, and whether anyone else contributed to the platform itself {% endcomment %}
- **5 organizations** onboarded, each with its own namespace and RBAC.
- **In production on it:** a Persian-first chemical compound database with a RAG assistant, the API gateway behind it, an LLM-powered crawler's frontend, and the engineering docs site.
- **New service to production:** > **TODO:** minutes from first push to a live URL.
- **Deploys:** > **TODO:** deploys per week, or total since launch (count image-updater commits).
- **Engineers using it:** > **TODO:** how many.
- **Onboarding docs:** the platform README and an MkDocs engineering site double as the onboarding path for new engineers.

# 6. What's next: GPUs

A GPU cluster has been bought and is being brought up. The plan is for it to join the same platform, so training and inference jobs ship the same way web apps do.

> **TODO:** the hardware (GPU model and count), and the plan: node pools/taints for GPU workloads, how models will be served (e.g. vLLM/Triton), and what goes on it first. Publish this section once it's real, or keep it to one line.

# 7. What I'd do differently

> **TODO:** one or two honest lessons (e.g. per-app namespaces from day one, cert-manager in-cluster, a second node for HA).

{% comment %} The repos and product URLs are private; this post links nothing internal. {% endcomment %}
