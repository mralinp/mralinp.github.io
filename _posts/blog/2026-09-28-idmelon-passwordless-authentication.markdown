---
layout: post
title:  "Case Study: Engineering Passwordless Authentication at IDmelon"
author: "Ali Naderi"
img: "/assets/images/posts/projects/idmelon/windows-signin.svg"
hero: false
date:   2026-09-28 20:00:00 +0330
categories:  blog authentication fido2 security windows
brief: "Two years building FIDO2 passwordless authentication at IDmelon, from R&D and prototypes to the first production release, and then leading the engineering team. With one component in depth: Windows sign-in with a security key for local accounts."
---
{% comment %}
Written from what Ali told me and from his private IDmelon repositories (architecture-level only:
no code, internals, customers or anything proprietary). Every statement below is either his account
or visible in those repos. Things to confirm or fill in before relying on this page:
  - [TODO: official title(s) and when the lead role started]
  - [TODO: team size you led]
  - [TODO: exact components you personally implemented beyond Windows sign-in]
  - [TODO: which other products you worked on: mobile app, desktop app, hardware reader, backend/admin]
  - [TODO: transports you worked with (USB HID, NFC, BLE), if any]
  - [TODO: certification involvement (e.g. FIDO certification), if any]
  - [TODO: production users/customers, only if publishable]
  - [TODO: a reliability or performance number you can defend]
  - [TODO: confirm the Windows sign-in component can be described publicly at this level]
{% endcomment %}
From 2022 to 2024 I worked at [IDmelon](https://idmelon.com), a Vancouver company building passwordless authentication on the FIDO2 standard. I was involved from the product's earliest stages and worked on it along the whole path from research to a product in production: the R&D, the prototypes, the demos, the first production version, the iterations after it, and later leading the engineering team that built it.

This page is about what I did there and how. It stays at the architecture level: nothing here is proprietary.

# Context

A password is a shared secret: the user knows it, the server stores something derived from it, and anyone who phishes or leaks it can use it. FIDO2 replaces it with public-key cryptography. An **authenticator** (a security key, or a phone acting as one) creates a key pair per site; the site stores only the public key. To sign in, the site sends a random challenge and the authenticator signs it with the private key, which never leaves the device. There is nothing reusable to phish or leak.

The standard handles the cryptography. The engineering problem is everything around it: getting real operating systems, applications and people to use it, reliably, every day.

# My role

- **R&D and prototypes.** I did the initial R&D, built the early prototypes, and presented and demonstrated them.
- **First production version.** I built the first production version, then kept improving it as it met real use.
- **Components and architecture.** I designed and implemented major technical components, and helped evolve the architecture as the product matured.
- **Leadership.** I led the engineering team, trained and mentored junior engineers, and worked closely with QC on quality.

My responsibilities grew with the product: from building the first working thing, to making it production-grade, to leading the people building it.

# From R&D to production

```text
research ─► prototype ─► demonstration ─► first production version ─► iteration in production ─► leading the team
```

Each step changes what "done" means. A prototype has to prove the idea in a demo. A production version has to work on machines you've never seen, fail safely, and be diagnosable when it doesn't. Iteration means taking what QC and users find and folding it back in without breaking what already works.

# In depth: Windows sign-in with a security key

One component shows the path well. At the time, Windows 10 supported signing in with a FIDO security key only for business accounts; people signing in to a **local** account couldn't use one at all. I designed and built the missing piece: sign-in to Windows with IDmelon, or any other FIDO2 security key, for local accounts.

![Windows sign-in with a FIDO2 security key: the sign-in screen loads a credential provider, which asks a background authentication service to authenticate the user; the service uses a local FIDO server as the relying party and talks to the security key, then returns a Windows credential](/assets/images/posts/projects/idmelon/windows-signin.svg)

It has three parts:

1. **A credential provider** (a C++ DLL). Windows loads credential providers into its sign-in screen; ours adds an IDmelon sign-in option for users who have registered a key. It's deliberately small: a sign-in button, and a call to the service.
2. **An authentication service** (a Windows service that starts at boot). It handles two requests, *register* and *authenticate*. When a user is authenticated with their key, it returns a Windows credential to the credential provider, which hands it to the sign-in screen.
3. **A local FIDO server.** FIDO2 always needs a relying party to issue challenges and verify the signed responses. With no web server in the picture, the relying party runs locally.

The split keeps the code that runs inside Windows' own sign-in screen minimal; the authentication work lives in a separate service. The first working version took about a week.
{% comment %}[TODO: confirm "about a week": the repo shows 26 → 31 March 2022, and whether this component shipped to customers]{% endcomment %}

# Production and quality

Getting from "works on my machine" to "works in production" was a large part of the job. I worked closely with QC to identify, reproduce, diagnose and fix bugs. Authentication is unforgiving here: a sign-in bug doesn't degrade the experience, it locks someone out of their computer.

# Leadership

As the product matured I led the engineering team: designing and implementing major components, helping evolve the architecture, and training and mentoring junior engineers.
{% comment %}[TODO: team size; whether you ran code/design review, planning, hiring or interviews]{% endcomment %}

# Outcome

The technology went from R&D and prototypes to production authentication products that kept developing as part of a real security company. IDmelon later became part of HID.

# Related

- [Smart Card: An Introduction to smart card development](/blog/embeded/smart-card/2023/07/19/smart-card-intro.html) and [Building Your First Applet with jCardSim](/blog/embeded/smart-card/2026/09/04/smart-card-dev.html): my own series on the hardware side of the same idea, building a FIDO security key on a Java Card, step by step.
