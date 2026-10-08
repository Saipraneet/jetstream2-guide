# Guidance for coding agents working in this repository

This repo is the CSCI 567 guide for running class-project experiments on Jetstream2. The operational knowledge
lives in the skill at `.agents/skills/jetstream2-experiments/SKILL.md`; load it before doing anything that
touches an instance, a volume, or the `js2` helper.

Rules that apply in every session:

- **Budget:** each team has a hard cap of 10,000 SUs (about 90 GPU-hours) with no top-up. Before `js2 launch`,
  state the flavor, the SU rate (h100 128/h, a100 64/h, a100-20 32/h, a100-10 16/h) and the estimated SU cost,
  and get the user's confirmation. Prefer the smallest flavor that fits; smoke-test with a few steps first.
- **Never leave a GPU instance ACTIVE when work is done.** `js2 stop` shelves (free); `js2 finish` pulls results
  and deletes. Do not shelve while a job is still running in tmux.
- **Code goes up, results come down, never the reverse.** Use `js2 push-code` and `js2 pull-results`; `js2 up`
  and `js2 down` overwrite whole trees and need an explicit request.
- Run `js2` by full path (`~/bin/js2`) if `~/bin` is not on PATH. Account config is in `~/.jetstream2/js2.conf`.
- The README is the student-facing document; keep it and SKILL.md consistent when changing either.
