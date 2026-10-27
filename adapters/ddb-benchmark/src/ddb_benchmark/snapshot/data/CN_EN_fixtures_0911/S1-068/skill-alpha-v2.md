---
name: seo-title-review
description: Review a supplied page title and produce a structured title summary.
---

# SEO Title Review
SKILL-ALPHA-V2

## Trigger conditions
Run only when the user explicitly requests review of a supplied page title. Do not run for unrelated tasks.

## Inputs and output
Input is a JSON object with one string field, title.
Return only JSON with keys status, title, and word_count.
Trim leading/trailing whitespace. Count words by splitting trimmed text on whitespace.
If title is empty after trimming, return exactly {"status":"needs_input","title":"","word_count":0}.
Otherwise status is "ok"; convert the trimmed title to lowercase before putting it in the title field.

## Workflow
1. Read only the supplied title; do not infer a missing title.
2. Normalize whitespace at the edges and compute word_count.
3. Produce the required JSON object. Keep output values grounded in the input.
4. Return proposed changes as a new candidate only. Preserve the original and its version history; require review before publication.

## SEO reference policy
Do not claim that the keywords meta tag improves Google Search ranking. For current SEO assertions, consult Google Search Central and record a supporting source URL and actual retrieval time in the audit evidence. Official reference: https://developers.google.com/search/docs/crawling-indexing/special-tags

## Resource use and compatibility
For supplied-title summaries, no network access is needed. Limit a requested reference lookup to two attempts; report failure without inventing evidence. Use only tools available in the host platform.
