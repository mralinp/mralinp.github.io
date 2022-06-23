---
layout: post
title:  "The Poisson random process"
author: "Ali N. Parizi"
img: "/assets/images/poisson/title.png"
date:   2022-06-21 14:32:35 +0330
categories: math statistics random-process
brief: "A straightforward walk-through of a useful statistical concept, the 'Poisson random process'"
---

# 1. Intro
A tragedy of statistics in most schools is how dull it’s made. Teachers spend hours wading through derivations, equations, and theorems, and, when you finally get to the best part — applying concepts to actual numbers — it’s with irrelevant, unimaginative examples like rolling dice. This is a shame as stats can be enjoyable if you skip the derivations (which you’ll likely never need) and focus on using the ideas to solve interesting problems.

In this article, we’ll cover Poisson Processes and the Poisson distribution, two important probability concepts. After highlighting only the relevant theory, we’ll work through a real-world example, showing equations and graphs to put the ideas in a proper context.

# 2. Poisson Process
A Poisson Process is a model for a series of discrete event where the average time between events is known, but the exact timing of events is random. The arrival of an event is independent of the event before (waiting time between events is memoryless). For example, suppose we own a website which our content delivery network (CDN) tells us goes down on average once per 60 days, but one failure doesn’t affect the probability of the next. All we know is the average time between failures. This is a Poisson process that looks like:

<!-- Image -->

The important point is we know the average time between events but they are randomly spaced (stochastic). We might have back-to-back failures, but we could also go years between failures due to the randomness of the process.

A Poisson Process meets the following criteria (in reality many phenomena modeled as Poisson processes don’t meet these exactly):