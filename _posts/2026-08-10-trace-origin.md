---
layout: post
title: Transparent Reporting and Auditable Code Execution
subtitle: Reporting / Python
tag: Clinical Programming
length: 5
image: trace/trace-cover.jpg
lead: A statistical program shouldn't only produce the right output. It should leave behind a clear record of how it got there.
next_post_title: Install POSROG V3U8 (PhoenixOS) Alongside Solus
next_post_slug: 2022-12-21-posrog-install
---

Python gives statistical programmers powerful tools for working with clinical data. But when it comes to generating Tables, Listings, and Figures, there is one part of the workflow that can become inconsistent very quickly:

**The log.**

One program uses `print()` statements. Another uses Python's `logging` module. Another records only errors. Some programs produce detailed execution histories; others leave almost no record of what happened between reading the source data and producing the final output.

The problem isn't that Python lacks logging.

The problem is that general-purpose logging doesn't tell statistical programmers **what should be logged, when it should be logged, or how those messages should be structured.**

That is the problem I'm exploring with **TRACE**.

## The Idea: Make the Journey Visible

**TRACE — Transparent Reporting and Auditable Code Execution** — is an open-source project designed to make structured logging easier and more consistent for Python-based statistical programming.

The idea is simple:

> **A statistical program should leave behind a clear record of how its output was produced.**

TRACE is being designed around workflows such as the generation of **Tables, Listings, and Figures (TLFs)**, where understanding what happened during execution can be just as important as knowing whether the program completed successfully.

Rather than replacing Python's existing logging infrastructure, TRACE builds a statistical-programming layer on top of it.

**Python handles the logging machinery.**

**TRACE defines the language and conventions.**

## From Data to Output

Consider a typical TLF program.

It may:

1. Read an analysis dataset.
2. Validate required variables.
3. Filter the analysis population.
4. Derive variables.
5. Merge additional datasets.
6. Calculate statistics.
7. Generate a table.
8. Write the final output.

The final table tells us the result.

But it doesn't necessarily tell us the story of how we got there.

TRACE is designed to make that story visible.

A TRACE-enabled execution might produce messages such as:

```text
INFO: [READ] [ADSL] loaded – N=754, Vars=16
INFO: [FILTER] [ADSL] SAFFL == 'Y' applied – N=754 → 720
INFO: [DERIVE] [AGEGR1] created
INFO: [SUMMARY] [T14_01] completed – Treatment groups=3
INFO: [OUTPUT] [T14_01] saved – outputs/tables/T14_01.rtf
```

Read from top to bottom, the log becomes a concise narrative of the program's execution.

## Creating a Common Language

At the center of TRACE is a lightweight logging framework.

The framework defines how statistical programs should communicate important execution events using a consistent message structure:

```text
[STEP] [OBJECT] [ACTION/STATUS] – [DETAILS]
```

For example:

```text
[READ] [ADSL] loaded – N=754, Vars=16
```

Each component answers a simple question:

* **STEP** — Where are we in the workflow?
* **OBJECT** — What dataset, variable, analysis, or output are we working with?
* **ACTION/STATUS** — What happened?
* **DETAILS** — What information would help someone understand or review the event?

TRACE also defines conventions around Python's standard logging levels — `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` — so that severity has a consistent meaning across statistical programs.

The goal isn't to log everything.

It's to log **what matters**.

## Turning the Framework Into a Tool

Documentation alone would still leave every programmer responsible for implementing the framework themselves.

That creates another problem.

If every project contains code such as:

```python
logging.info(...)
logging.warning(...)
logging.error(...)
```

with manually constructed messages, consistency still depends on individual programmers remembering and applying the TRACE conventions correctly.

So TRACE is being developed as more than a written framework.

The project will include a **Python library that implements the framework directly**.

Instead of repeatedly constructing logging messages, the intended experience is closer to:

```python
trace.read(adsl, "ADSL")
trace.filter("SAFFL == 'Y'", before=754, after=720)
trace.derive("AGEGR1")
trace.output("T14_01", "outputs/tables/T14_01.rtf")
```

TRACE can then translate those operations into standardized log messages.

This shifts responsibility away from individual programs.

The programmer describes **what happened**.

TRACE determines **how it should be logged**.

## Designed for Statistical Programming

TRACE is deliberately not intended to become another general-purpose Python logging library.

Python already has one.

Instead, TRACE is being designed around concepts statistical programmers encounter repeatedly:

```text
READ
CHECK
FILTER
TRANSFORM
MERGE
DERIVE
ANALYZE
SUMMARY
VALIDATE
OUTPUT
```

These operations form a vocabulary for describing the journey from analysis data to final statistical output.

The library can then provide domain-aware utilities around that vocabulary.

For example, when working with a DataFrame, useful execution context may include observations, variables, filtering results, merge characteristics, or missing values.

That is where TRACE can provide value beyond simply formatting log messages.

## Keep the Simple Things Simple

One of the main design goals for TRACE is **low-friction adoption**.

Adding structured logging to an existing TLF program shouldn't require programmers to become experts in Python handlers, formatters, propagation, configuration, or custom logging classes.

The ideal experience is:

```python
from trace_tlf import Trace

trace = Trace("T14_01")
```

From there, programmers should be able to progressively add TRACE wherever execution visibility matters.

A team shouldn't need to redesign its programming environment to adopt TRACE.

It should be possible to introduce it into an existing script in minutes.

That principle is influencing the API currently being designed.

## Built on Python, Not Against It

TRACE will not attempt to recreate functionality already provided by Python's standard `logging` module.

Instead, the architecture is intended to sit above it:

```text
Statistical Program
        │
        ▼
   TRACE Python
        │
        ▼
 TRACE Framework
        │
        ▼
 Python logging
```

Python continues to handle the underlying logging infrastructure.

TRACE provides the statistical-programming semantics, conventions, formatting, and developer experience.

This keeps the project lightweight while allowing it to benefit from Python's mature logging ecosystem.

## Framework, Library, Package

TRACE has three closely related parts.

### TRACE Framework

The framework defines the methodology:

* Logging conventions
* Message structure
* Logging levels
* Statistical workflow terminology
* Recommended logging locations
* Good logging practices

### TRACE Python

The Python library turns those conventions into reusable software.

It provides the API programmers interact with when adding TRACE to their statistical programs.

### PyPI Distribution

The library is intended to be distributed as an installable Python package, making adoption as simple as installing it into an existing environment and importing TRACE into a program.

Together, these components separate **the idea from the implementation**.

That distinction also leaves the framework open to implementations beyond Python in the future.

## Designing the API Before the Implementation

TRACE is currently in the **API design phase**.

Before implementing the library, I'm defining the interface from the statistical programmer's perspective:

What should reading a dataset look like?

How should filtering be recorded?

What information should a merge capture?

How should derivations, validations, analyses, and outputs be represented?

Which operations should TRACE automate, and which should remain explicitly controlled by the programmer?

The aim is to make those decisions before building the internal architecture.

For an API intended to simplify logging, simplicity needs to exist at the design level — not be added after implementation.

## Building TRACE in the Open

TRACE is being developed as an **open-source project**.

The project is structured around three core pieces:

```text
TRACE
├── Framework documentation
├── Python implementation
└── PyPI package
```

The repository will also document the API, provide practical TLF examples, tests, implementation guidance, and contribution information.

The public implementation will evolve alongside the API as TRACE moves from design into development.

## Where TRACE Is Going

The initial goal is intentionally focused:

> **Make it extremely easy to add useful, consistent logging to Python statistical programs.**

From there, TRACE could support richer capabilities around DataFrame-aware logging, validation, execution summaries, structured logs, output metadata, and integrations with tools increasingly being used for clinical statistical programming.

But those features are secondary.

The fundamental problem TRACE needs to solve first is much smaller.

A programmer should be able to open an existing Python TLF script, add TRACE with minimal effort, run the program, and come away with a log that clearly explains what happened.

## The Bigger Idea

TRACE started with logging, but the underlying problem is really about **visibility**.

A statistical output may be correct, but the output alone doesn't communicate every decision and transformation that happened during execution.

Structured logging creates a narrative alongside the result.

It makes it easier to understand what data was read, what changed, what was validated, what was produced, and where something unexpected happened.

TRACE is my attempt to make that kind of execution visibility a natural part of Python statistical programming rather than something each programmer has to reinvent.

Because a statistical output shouldn't only be reproducible.

**Its journey should be traceable.**
