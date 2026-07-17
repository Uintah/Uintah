---
name: uintah-workflow
description: >-
  Build Uintah, author a .ups input file, run a simulation with sus/mpirun, and post-process the
  .uda output with lineextract/timeextract/puda. Use for any Uintah (ICE/MPM/MPMICE) simulation
  setup, run, or data-analysis task.
---

# Uintah workflow

End-to-end guide for a Uintah simulation: **build → author a `.ups` input → run with `sus` →
post-process the `.uda` output**.

Locations are referred to by role. The **build directory** is created by `configure` as a *sibling*
of `src/` (common names: `opt`, `dbg`, a `*_gcc` variant); it holds `sus` and the post-processing
tools under its own `StandAlone/`. Run all `sus` and tool commands from the build directory's
`StandAlone/`.

## 1. Build

From the configured build directory (the sibling of `src/`), **not** from `src/`:

```bash
make -jN uintah
```

## 2. Author a `.ups` input file

A `.ups` is XML wrapped in `<Uintah_specification>`. **Don't start from scratch** — copy a close
template from `src/StandAlone/inputs/{ICE,MPM,MPMICE}/` and edit it.

### Required sections (in order)

| Section | Purpose |
|---------|---------|
| `<Meta>` | `<title>` for the run |
| `<SimulationComponent type="ice"/>` | component: `ice`, `mpm`, or `mpmice` |
| `<Time>` | `maxTime`, `delt_min`, `delt_max`, `delt_init`, `timestep_multiplier`; optional `<max_Timesteps>` to cap step count |
| `<Grid>` → `<BoundaryConditions>` | per-`<Face side="x-">` a list of `<BCType label=... var=...>`; var types include `LODI`, `Dirichlet`, `Neumann`, `symmetry`, `computeFromDensity` |
| `<Grid>` → `<Level><Box>` | `lower`/`upper` (domain extents), `extraCells` (usually `[1,1,1]`), `patches`, `resolution` (cells per axis) |
| `<DataArchiver>` | `<filebase>` → `<name>.uda`, `<outputInterval>`, a list of `<save label="..."/>`, `<checkpoint>` |
| `<CFD><ICE>` | `cfl`, `<advection type="SecondOrder"/>`; optional `<ImplicitSolver>` |
| `<PhysicalConstants>` | `gravity`, `reference_pressure` |
| `<MaterialProperties>` | per-material EOS, viscosity, model config, initial conditions |

### Two rules that bite

- **You can only extract variables you saved.** Post-processing reads only labels that appear as
  `<save label="..."/>` in `<DataArchiver>`. Common CC labels: `temp_CC`, `vel_CC`, `rho_CC`,
  `press_CC`, `specific_heat`, `viscosity`, `thermalCond`. For MPM particle work, save
  `p.particleID` too (see the puda gotcha below).
- **Patches must tile the resolution**, and the `patches` product should equal or divide the MPI
  rank count (e.g. `patches [16,1,1]` with `mpirun -n 16` or `-n 8`).

## 3. Run

From the build directory's `StandAlone/`:

```bash
# Parallel (typical)
mpirun -n N sus inputs/ICE/<case>.ups > run.log 2>&1

# Serial
sus inputs/ICE/<case>.ups > run.log 2>&1
```

**Output:** a `<filebase>.uda` symlink pointing at the newest run, plus numbered
`<filebase>.uda.000`, `.001`, … for previous runs. Tail the log for timestep size, CFL, and restart
messages.

- **Restart:** `mpirun -n N sus -restart <name>.uda` or `sus -restart <name>.uda` (confirm the exact flag with `sus --help` first).
- **Inspect task scheduling:** `export SCI_DEBUG='<component>_tasks:+'` prints the schedule order
  before the task graph compiles.

## 4. Data analysis / post-processing

Tool binaries live under the build directory's `StandAlone/tools/`. Run from `StandAlone/`.

### lineextract — spatial line of a cell-centered variable

```bash
tools/extractors/lineextract -v press_CC -timestep 10 \
  -istart 0 0 0 -iend 199 0 0 -uda <case>.uda -o press.txt
```

- **Gotcha:** `-o /dev/stdout` interleaves data with header lines — always write to a **real file**
  and parse it. Omit `-timestep` to extract **all** timesteps at once (parse in chunks).
- Output columns: `i j k value` for scalars — **column 4** is the value. Parse with `awk`/Python,
  e.g. `awk 'NR>1{print $4}' press.txt`.

### timeextract — time series at a single cell

```bash
tools/extractors/timeextract -v temp_CC -istart 100 0 0 -iend 100 0 0 \
  -uda <case>.uda -o Tcenter.txt
```

### puda — uda inspector (grid & particle data)

```bash
tools/puda/puda -timesteps <case>.uda          # list timestep indices/times
tools/puda/puda --help                         # full flag list
tools/puda/puda -partvar p.temperature <uda>   # MPM particle variable dump
tools/puda/puda -partextract <uda>             # extract particle data
```

- **Known crash:** puda's Vector branch dereferences `p.particleID` even when it wasn't saved →
  segfault / garbage. **Save `p.particleID`** in the `.ups`, or guard your parser accordingly.

### Other tools

`compare_uda` (diff two udas), `partextract`, `faceextract`, `extractV`/`extractS`/`extractF`,
`pfs`.

### Analysis & plotting

Reuse the project's existing analysis scripts rather than rewriting extractor-parsing + plotting
from scratch. When validating a physics model, overlay Uintah output against a trusted reference
solution evaluated at the *same* states, and pick a conservation or steady-state diagnostic as the
pass/fail criterion.
