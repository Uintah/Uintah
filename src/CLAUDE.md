# Uintah — guidance for Claude agents

This file is auto-loaded into every Claude Code session rooted in the Uintah tree. It captures the
general, reusable knowledge for developing and running Uintah (ICE / MPM / MPMICE) so an agent does
not have to re-derive the workflow or re-discover the common API gotchas.

Everything here is machine-agnostic. Refer to locations by role (the *source tree*, the *build
directory*), never by absolute path — each developer's checkout lives somewhere different.

## Orientation

- **Uintah** is a computational framework; the components you will most often touch are **ICE**
  (compressible CFD), **MPM** (material point method / solid mechanics), and **MPMICE** (coupled
  fluid–solid).
- `src/` is the **source tree** (this directory). The **build directory** is created by
  `configure` as a *sibling* of `src/` (common names: `opt`, `dbg`, or a `*_gcc` variant). It holds
  the compiled `sus` executable and the post-processing tools under its own `StandAlone/`.
- Simulation inputs are XML `.ups` files under `src/StandAlone/inputs/{ICE,MPM,MPMICE}/`.

## Build & run, in one line

- **Build** from the configured build directory (the sibling of `src/`), *not* from `src/`:
  `make -jN uintah`.
- **Run** `sus` from that build directory's `StandAlone/`:
  `mpirun -n N sus inputs/<component>/<case>.ups > run.log 2>&1`.

For the full workflow — authoring a `.ups`, running, and post-processing the `.uda` output with
`lineextract` / `timeextract` / `puda` — **invoke the `uintah-workflow` skill**. It carries the
step-by-step detail and the extractor gotchas.

## Code style

General rules:

- **camelCase** for local variable names (`lowX`, `hiX`, `invMwMix`). Do **not** use
  underscore-separated names (`low_x`, `grad_x`). Exception: physical-quantity locals keep the
  grid-location suffix below (`temp_CC`, `rho_CC`).
- **Always** use `{}` braces on loops and conditionals, even for a single-line body.
- Match the surrounding file's conventions (comment density, naming, idiom) when they are stricter.

Naming conventions the components actually use (verified in ICE and MPM source):

- **`d_` prefix = a class member variable** — persistent per-component state, i.e. a "global" of
  that component, as opposed to a local. Examples from `CCA/Components/ICE/ICE.h`: `d_ref_press`,
  `d_gravity`, `d_impICE`, `d_CFL`, `d_advector`, `d_SMALL_NUM`. Same convention in MPM
  (`CCA/Components/MPM/Core/MPMFlags.h`: `d_gravity`, `d_axisymmetric`, …). A plain (non-`d_`) name
  is a local or a struct field. When you add member state to a component, prefix it `d_`.
- **`...Label` suffix = a `VarLabel*`** (a DataWarehouse variable handle), declared centrally in the
  component's label class (e.g. `CCA/Components/ICE/Core/ICELabel.h`: `press_CCLabel`, `delTLabel`,
  `gammaLabel`). Don't scatter `new VarLabel(...)` calls; add the label to that class.
- **Grid-location / role suffixes** on variable and label names:
  - `_CC` — cell-centered (`temp_CC`, `rho_CC`, `press_CC`)
  - `_FC`, `_FCX` / `_FCY` / `_FCZ` — face-centered (per axis)
  - `_FCME` — face-centered value *after* the momentum exchange
  - `_L` — Lagrangian-phase quantity (`eng_L_ME_CC`)
  - `_adv` — advected quantity; `_src` — a source/sink term; `dTdt` / `dVdt` — time derivatives.
- **Scheduler pairing** — a `schedule<Foo>(...)` method schedules the task whose callback is
  `<Foo>(...)` (e.g. `scheduleComputePressure` schedules `computePressure`). Follow the pair when
  adding a task.

## Debugging methodology — the Four C's

For any nontrivial numerical discrepancy (wrong flame speed, conservation violation, unexpected
profile), work through **Concern → Cause → Correction → Confirm** rather than jumping straight to a
patch:

1. **Concern** — state the observed discrepancy precisely, including the diagnostic that will tell
   you it is resolved (e.g. a conservation residual, a reference-solution overlay).
2. **Cause** — propose a single candidate root cause and get agreement on it *before* implementing.
   This avoids burning effort coding fixes for causes that were never plausible.
3. **Correction** — implement the fix. Back it with a physics/derivation argument for *why* this
   cause produces the Concern, and cite every code change by file + line.
4. **Confirm** — run the case and produce plots showing whether the Concern resolved, compared
   against the reference.

## Component development

When editing component source (Scheduler tasks, DataWarehouse variables, ghost cells), see
`CCA/Components/CLAUDE.md` for the task-ordering and ghost-iterator rules that most often bite.
