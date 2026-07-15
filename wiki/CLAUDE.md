# ReMo3D Research Wiki — Schema & Operating Manual

This directory is an **LLM-maintained research wiki** for the ReMo3D project. It
is a persistent, interlinked knowledge base about the *science and methods*
behind ReMo3D — the physics of resistivity logging, the numerical methods,
relevant papers, and the evolving findings from sensitivity analysis and
performance work.

**You (the LLM) own this directory.** You create pages, update them when new
sources arrive, maintain cross-references, and keep everything consistent. The
user curates sources, directs analysis, and asks questions. You do the
bookkeeping.

> This wiki is deliberately distinct from `../docs/`, which documents the
> **code**. When implementation detail is relevant, **link out** to the
> matching `../docs/*.md` page rather than duplicating it here.

## Layers

1. **Raw sources** (`raw/`) — immutable source documents the user drops in
   (papers, articles, notes, exported data, images in `raw/assets/`). Read from
   these; never modify them. Source of truth.
2. **The wiki** (everything else) — the markdown pages you generate and maintain.
3. **The schema** — this file. Co-evolve it with the user as conventions settle.

## Directory layout

| Dir / file      | Holds                                                                 |
|-----------------|-----------------------------------------------------------------------|
| `raw/`          | Immutable source documents. `raw/assets/` for images.                 |
| `sources/`      | One summary page per ingested source. Filename mirrors the source.    |
| `concepts/`     | Concept / theory pages (e.g. resistivity logging, mesh convergence).  |
| `entities/`     | Named things: methods, tools, datasets, people, institutions.         |
| `findings/`     | Results & synthesis from our own analysis (sensitivity, benchmarks).  |
| `index.md`      | Content catalog. Every page listed with a one-line summary. Curated.  |
| `log.md`        | Append-only chronological record of ingests / queries / lints.        |
| `overview.md`   | The evolving synthesis / thesis. The one page to read first.          |

## Page conventions

- **Frontmatter** on every page:
  ```yaml
  ---
  title: <human title>
  type: source | concept | entity | finding | overview
  tags: [ ... ]
  sources: [ <source-page-slugs this page draws on> ]
  updated: <YYYY-MM-DD>   # today's date is given in the session context
  ---
  ```
- **Cross-link liberally** with `[text](../concepts/foo.md)` relative links. A
  link to a page that doesn't exist yet is a *fine* signal that the page is worth
  creating — treat it as a to-do, not an error.
- **Cite sources inline** as `[[source-slug]]` referencing a page in `sources/`.
  Every non-obvious claim traces to a source page (or is marked `(seeded from
  repo)` / `(inference)` when it doesn't).
- **Flag contradictions explicitly.** When a new source disagrees with an
  existing claim, don't silently overwrite — add a `> ⚠️ Contradiction:` note
  citing both sides, and reconcile with the user.
- Keep pages focused and short. Split when a page sprawls.
- Dates: convert relative dates ("last week") to absolute `YYYY-MM-DD` using the
  session date.

## Operations

### Ingest (user drops a source into `raw/` and says "ingest it")
1. Read the source fully (for image-heavy markdown, read text first, then view
   referenced images in `raw/assets/`).
2. Briefly discuss key takeaways with the user.
3. Write/refresh a summary page in `sources/` (frontmatter, key points,
   verbatim-worthy quotes, figures, your assessment).
4. Update **relevant** `concepts/`, `entities/`, `findings/` pages — integrate
   the new information, add cross-links, flag contradictions. One source may
   touch 10–15 pages; touch every page it actually bears on, none it doesn't.
5. Update `index.md`.
6. Append an entry to `log.md`.

### Query (user asks a question)
1. Read `index.md` (and `overview.md`) to locate relevant pages, then drill in.
2. Answer with citations to wiki pages / source slugs.
3. **File good answers back** — a comparison, analysis, or discovered connection
   worth keeping becomes a new page (usually in `concepts/` or `findings/`),
   indexed and logged. Explorations should compound, not vanish into chat.

### Lint (user asks for a health check)
Scan for: contradictions between pages; stale claims newer sources supersede;
orphan pages (no inbound links); concepts mentioned but lacking a page; missing
cross-references; data gaps a web search could fill. Report findings and
proposed fixes; suggest new questions and sources worth pursuing.

## Log format

Append entries with a consistent prefix so `grep '^## \[' log.md | tail` works:

```
## [YYYY-MM-DD] ingest | <source title>
- pages touched: ...
```
`ingest` / `query` / `lint` / `scaffold` are the entry kinds.

## Style

- Neutral, precise, source-grounded. Distinguish established fact, this
  project's findings, and open questions.
- Prefer tables and short prose over walls of text. Use LaTeX for math
  (`$...$`), Mermaid for structure when it earns its place.
- When unsure whether something is settled, say so rather than asserting.
