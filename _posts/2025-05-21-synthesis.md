---
layout: post
title: "Synthesis: Trial Simulation"
category: Clinical Programming
tag: Python
length: 3
image: synthesis/synthesis-listing-card.avif
lead: Clinical programming depends heavily on data.
---

But realistic clinical trial data isn't always easy to access — particularly when developing tools, testing workflows, building examples, or experimenting outside a live study environment.

**Synthesis is my attempt to make that easier.**

Synthesis is an open-source **synthetic clinical trial simulator** currently under development. It is designed to generate realistic synthetic clinical trial data that can support development, testing, education, and experimentation without relying on real patient data.

## From Dataset Generator to Trial Simulator

Synthesis originally started with a narrower goal: generating synthetic SDTM datasets.

As the project evolved, I realized that generating datasets independently was only part of the problem.

Clinical trial datasets exist within a much larger context. Subjects participate in a study. They are assigned to treatment arms. They attend visits. Treatments are administered. Events occur over time. Those events eventually become the data represented across clinical datasets.

That shifted the direction of the project.

Instead of asking:

> **How can I generate an SDTM dataset?**

The more interesting question became:

> **How can I simulate the clinical trial that produces the data?**

That is the idea behind Synthesis.

## Simulating the Study, Not Just the Data

The goal is to model enough of a clinical trial's structure and subject journey that datasets can emerge from a coherent simulated study rather than being generated as isolated tables.

At a high level:

```text
Study Design
      ↓
Synthetic Trial
      ↓
Subject Journeys
      ↓
Clinical Events
      ↓
Clinical Trial Data
```

This distinction matters.

If the underlying study is coherent, the resulting data can better reflect relationships that exist across subjects, visits, treatments, events, and datasets.

## Why Build It?

Synthetic clinical trial data has several useful applications.

It can provide realistic data for developing and testing clinical programming tools, creating reproducible examples, experimenting with statistical workflows, demonstrating software, and learning clinical data standards without exposing actual participant information.

For me, Synthesis also creates something particularly valuable:

**a controlled environment for building and testing other clinical programming ideas.**

Rather than designing tools around small handcrafted examples, I want to be able to simulate a study and use its resulting data as a realistic development environment.

## Currently Under Development

Synthesis is an active project and its design is still evolving.

The current work is focused on establishing the foundations of the simulator and progressively expanding the clinical scenarios it can represent.

The project is being developed in **Python**, with the intention of making the resulting simulator reusable, extensible, and open source.

Implementation details are intentionally omitted here while the architecture continues to evolve.

## The Bigger Idea

Synthesis started as an SDTM dataset generator.

It is becoming something broader.

Instead of treating synthetic clinical data as a collection of independently generated records, the project approaches it as the **output of a simulated clinical trial**.

That gives the project a simple long-term direction:

> **Simulate the trial. Let the data follow.**
