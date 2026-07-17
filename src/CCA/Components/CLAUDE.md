# Uintah component development — task graph & DataWarehouse gotchas

Auto-loaded when an agent edits component source under `CCA/Components/`. These two Uintah-wide
rules cause the most confusing runtime failures.

## Scheduler task ordering

A task's `NewDW` `requires` binds **only** to a `computes` from a task added **earlier** in
`addTask` / schedule order. The task graph does **not** reorder a task to satisfy a dependency that
is produced later in the same timestep.

- Symptom when you get it wrong: `InternalError: Failed to find comp for dep!` at graph compile.
- **OldDW** requires are exempt — they are satisfied by a send-old-data task and are always
  available. (But note: not every variable persists into OldDW; face-centered velocities, for
  example, do not.)
- If task B needs a variable that task A computes, A must be scheduled before B. If B is currently
  scheduled first, move A's `schedule*` call ahead of B's — this is legal as long as A does not
  itself require something B produces.
- Do **not** rely on "the DAG will reorder it" — it will not.

**Inspect the schedule order** before the graph compiles:

```bash
export SCI_DEBUG='<component>_tasks:+,ICE_tasks:+'
```

This prints where each task lands relative to the others, which is the fastest way to see whether a
require can actually bind.

## Ghost-cell iterator

When you fill a temporary `CCVariable` from a DataWarehouse variable fetched with
`Ghost::AroundCells, 1`, iterate with **`patch->getExtraCellIterator(1)`**, not the no-arg
`getExtraCellIterator()`.

- The no-arg form (`ngc = 0`) does **not** extend over inter-patch (Neighbor) faces, so the ghost
  layer at patch boundaries is skipped and stays zero-initialized.
- Consequence: quantities built from that temporary (fluxes, gradients) are wrong at patch
  boundaries, silently breaking conservation on multi-patch runs while single-patch runs look fine.
- Rule of thumb: if a temporary is allocated with `gac, 1` and filled from a DW variable also
  fetched with `gac, 1`, pass the matching ghost count to the iterator so the fill covers the layer
  the scheduler already populated.
