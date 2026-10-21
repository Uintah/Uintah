# Uintah Buildbot Configuration

Oct 21, 2026 · @T

## Overview

The Uintah buildbot on `uintah-build.chpc.utah.edu` (Buildbot 2.0.1, Python 3.7, Debian 10) is defined by three files: `master.cfg` holds the configuration, and `commonBuildSteps.py` holds the reusable build steps that `master.cfg` assembles into factories, and `emailFormatters.py` formats the failure emails.

| File | Role |
| --- | --- |
| `master.cfg` | The `BuildmasterConfig` dictionary: worker, change source, schedulers, compiler environments and configure lines, factories, builders, mail notifiers, web UI, database. Imports everything from `commonBuildSteps.py` with `from commonBuildSteps import *`. |
| `commonBuildSteps.py` | Helper functions that each add one or more steps to a factory (`gitUpdate`, `configure`, `compile`, `makeclean`, `copyBuildDir`, `runComponentTests`, `rm_localRT_dir`), plus `runOrSkip_BuildBot`, which decides whether a commit should trigger the builders. |
| `emailFormatters.py` | `CompileErrorFormatter`, which adds the compiler errors to failure emails. Imported in `master.cfg` with `from emailFormatters import CompileErrorFormatter`. |

The flow from commit to email:

1. `GitPoller` checks GitHub every minute for new commits on `master` and `kokkos_dev`.
2. `runOrSkip_BuildBot` drops commits that only touch docs or scripts, or that say `skipRT`.
3. A scheduler starts one or more builders, after a 1-minute quiet period.
4. Each builder runs its factory's steps on the worker `buster`, using the builder's properties (compiler, configure line, build directory, branch).
5. If a build fails, a copy of the cleaned build tree is saved under `/data/buildbot/<builder>/<build number>/`.
6. Mail notifiers email the commit authors and/or `t.harman@utah.edu`.

This document describes the files in this directory (`src/scripts/buildbot`). The live copies are in `/var/lib/buildbot/masters` (see Operating the master).

## Change detection

A commit triggers builds only if `GitPoller` sees it and `runOrSkip_BuildBot` marks it important.

**GitPoller** (`master.cfg`, `repo_git`)

| Setting | Value |
| --- | --- |
| Repository | `https://github.com/Uintah/Uintah.git` |
| Branches | `master`, `kokkos_dev` |
| Project label | `Uintah` |
| Poll interval | 60 s |
| Work directory | `gitpoller-work` (inside the master's base directory) |

**runOrSkip\_BuildBot** (`commonBuildSteps.py`) is the `fileIsImportant` function of the commit-triggered schedulers. It returns False (skip) when either rule matches:

- The commit message contains `skiprt`, `skip_rt` or `skip-rt`, case-insensitive.
- Every changed file lives under one of these directories: `doc`, `src/parametricStudies`, `src/scripts`, `src/testprograms`, `src/CCA/Components/FVM`, `src/CCA/Components/Arches/Attic`, `src/CCA/Components/ICE/Docs`, `src/CCA/Components/ICE/Matlab`, `src/CCA/Components/ICE/PressureSolve`, `src/CCA/Components/MPMICE/Docs`, `src/CCA/Components/PhaseField`.

A commit that changes any file outside those directories triggers the builders. A file matches an ignored directory only if it is in that directory or below it, so `doc` does not match `docs/` or `doc_old/`. A change with no file list is skipped.

## Schedulers

A commit on `master` starts `Trunk:dbg-compile`; if it passes, `Trunk:dbg-compile-static` runs, and if that passes, `Trunk:opt-full` runs. A commit that touches `doc/` also starts `Documentation-Builder`. Every other builder runs nightly or only when started by hand.

| Scheduler | Type | Trigger | Builders |
| --- | --- | --- | --- |
| `compile-dbg` | SingleBranchScheduler | Commit on `master` that passes `runOrSkip_BuildBot`, after 60 s with no further commits | `Trunk:dbg-compile` |
| `compile-dbg-static` | Dependent | `compile-dbg` build succeeds | `Trunk:dbg-compile-static` |
| `full_opt_cpu` | Dependent | `compile-dbg-static` build succeeds | `Trunk:opt-full` |
| `nightly-compile` | Nightly | 22:01 daily, only if `master` changed since the last run (`onlyIfChanged = True`) | `Trunk:dbg-compile-clang` |
| `doc-build` | SingleBranchScheduler | Commit on `master` that changes a file under `doc/` (`docChanged`), after 60 s with no further commits | `Documentation-Builder` |
| `force` | ForceScheduler | "PushMe!" button in the web UI | `Documentation-Builder`, `Trunk:dbg-compile`, `Trunk:dbg-compile-clang`, `Trunk:dbg-full`, `Trunk:dbg-compile-static`, `Trunk:opt-full`, `Trunk:opt-full-sanitize` |
| `try` | Try\_Userpass | `buildbot try` client on port 8031 | `Trunk:dbg-full`, `Trunk:opt-full`, `Trunk:opt-full-sanitize` |
| `buildbot_testing` | ForceScheduler | Testing form in the web UI, with a choice of components (`ICE`, `MPM`) | `buildbot_testing` |

The Kokkos schedulers (`kokkos-compile-omp` → `kokkos-compile-cuda` → `kokkos-compile-omp-cuda` → `kokkos-full`, on `kokkos_dev`) and `full-opt-gpu` are defined but not registered. The force form's branch, revision, repository and project are fixed (`master`, `Uintah`), so a forced build always uses the latest `master`.

## Builders and factories

Eight builders are active, all on the worker `buster` (`max_builds=6`). Each builder pairs a factory (the list of steps) with properties that tell those steps what to build.

### Factories

| Factory | Steps, in order |
| --- | --- |
| `f_doc` | Git checkout of `master` → `./runLatex` in `build/doc` → remove `src/` |
| `f_compile` | `gitUpdate` → `configure` → `compile` (16 procs) → `makeclean` → `copyBuildDir` |
| `f_compile_runTests` | `gitUpdate` → `configure` → `compile` (16 procs) → `runComponentTests` → `makeclean` → `copyBuildDir` |
| `f_compile_runTests_gpu` | Same as above, with `./R_Tester/modifySrc` after `gitUpdate` (used only by a disabled builder) |
| `f_bb_test` | `runComponentTests` only (`recompile = False`), against the build left in `buildbot_testing/build/dbg` on 2022-10-20 |
| `f_runTests` | `rm_localRT_dir` → `runComponentTests` (defined, not used by any builder) |

`makeclean` runs before `copyBuildDir` on purpose: the saved copy of a failed build is cleaned first to save space.

### Active builders

| Builder | Factory | Compiler env | Configure line | `wrkDir` | Branch |
| --- | --- | --- | --- | --- | --- |
| `Documentation-Builder` | `f_doc` | none | none | `none (fixed workdirs)` | `master` |
| `Trunk:dbg-compile` | `f_compile` | `compiler_env` (mpicc/mpic++) | `dbg_configCmd` | `build/dbg` | `master` |
| `Trunk:dbg-compile-clang` | `f_compile` | `clang_compiler_env` (clang-10) | `dbg_configCmd` | `build/dbg` | `master` |
| `Trunk:dbg-compile-static` | `f_compile` | `compiler_env` | `dbg_static_configCmd` | `build/dbg_static` | `master` |
| `Trunk:dbg-full` | `f_compile_runTests` | `compiler_env` | `dbg_configCmd` | `build/dbg` | `master` |
| `Trunk:opt-full` | `f_compile_runTests` | `compiler_env` | `opt_configCmd` | `build/opt` | `master` |
| `Trunk:opt-full-sanitize` | `f_compile_runTests` | `compiler_env` | `opt_sanitize_configCmd` | `build/opt` | `master` |
| `buildbot_testing` | `f_bb_test` | `compiler_env` | `dbg_configCmd` | `build/dbg` | `master` |

Disabled builders (defined but not in `master['builders']`): `Trunk:opt-full-gpu`, `Kokkos:opt-full`, `Kokkos:opt-omp-compile`, `Kokkos:opt-cuda-compile`, `Kokkos:opt-omp-cuda-compile`.

### Builder properties

The helpers in `commonBuildSteps.py` read these with `util.Property(...)` at build time, so one factory serves many builders.

| Property | Used by | Meaning |
| --- | --- | --- |
| `compiler_env` | `configure` | Environment for `configure`: `CC`, `CXX`, `F77`, `SCI_MAKE_BE_QUIET`, `LANG` |
| `configCmd` | `configure` | The full `../src/configure ...` command as a list |
| `wrkDir` | `configure`, `compile`, `makeclean`, `runComponentTests` | Build directory, relative to the builder directory |
| `myBranch` | `gitUpdate` | Git branch to check out |
| `defaultTestComps` | `getTestComponents()` | Components to test when the try client sets no `test_components` |

### Configure lines

Each configure line is built by adding lists together, e.g. `dbg_configCmd = dbg_base + all_components + mpi + hypre + petsc + boost`.

| Configure line | Base | Differences |
| --- | --- | --- |
| `dbg_configCmd` | `--enable-debug --enable-sci-malloc` | All components, hypre, PETSc 3.10.5, Boost 1.65.1 |
| `dbg_static_configCmd` | Debug + `--enable-static`, `CXXFLAGS=-pthread` | Static hypre, `--without-petsc` (PETSc does not link statically) |
| `opt_configCmd` | `--enable-optimize=-O3 -mfpmath=sse`, assertion level 0 | All components, hypre, PETSc, Boost |
| `opt_sanitize_configCmd` | Same as `opt_configCmd` | Adds `CXXFLAGS=-fsanitize=address,undefined ...` |

### Regression-test environment

The `*-full` builders and `buildbot_testing` set these environment variables for `make runLocalRT`:

| Variable | Value |
| --- | --- |
| `TEST_COMPONENTS` | `getTestComponents()`: the try client's `test_components`, else the builder's `defaultTestComps` (`ARCHES Examples ICE IMPM Models MPMICE MPM UCF Wasatch`). For `buildbot_testing`, the components picked in the force form, joined with spaces (`ICE MPM`). |
| `WHICH_TESTS` | `BUILDBOTTESTS` |
| `GOLD_STANDARDS` | `/home/rt/Linux/TestData/dbg` or `.../opt` |
| `RT_CHECKPOINTS` | `/home/rt/CheckPoints/` |

## commonBuildSteps.py

Each helper takes a factory and appends steps to it. Arguments left as `None` are read from the builder's properties at build time through `getValue()`.

| Function | Steps it adds | Notes |
| --- | --- | --- |
| `gitUpdate( factory )` | Git checkout of `myBranch` into `build/` (incremental, shallow, 10 retries, clobber on failure); `rm -rf doc/` | The doc removal halts the build if it fails |
| `configure( factory, wrkDir, configCmd )` | `rm -rf *` in `wrkDir`; `Configure` with `configCmd` and `compiler_env` | Configure has `haltOnFailure=True`: a configure failure stops the build; `makeclean` and `copyBuildDir` still run (`alwaysRun=True`) |
| `compile( factory, numProcs, wrkDir )` | `Compile` in `wrkDir` | Runs `python ../src/scripts/buildbot/make.py <numProcs>`. Has `haltOnFailure=True`: a compile failure stops the build, but `makeclean` and `copyBuildDir` still run (`alwaysRun=True`). |
| `makeclean( factory, wrkDir )` | `make reallyclean` in `wrkDir` | Uses the `wrkDir` argument, or the `wrkDir` property when none is passed |
| `copyBuildDir( factory )` | `mkdir` → tar `build/` → untar into `/data/buildbot/<builder>/<build number>/` → change permissions there (group `users`, `g+rwX`) | Every step has `alwaysRun=True`, so it runs even after a step halts the build, but only if `hasStepFailed()` is True; it is hidden when skipped |
| `runComponentTests( factory )` | `make runLocalRT` in `wrkDir` | 1-hour timeout; does not halt the build on failure |
| `rm_localRT_dir( factory, wrkDir )` | `rm -rf local_RT` in `wrkDir` | Used only by the unused `f_runTests` |

### Supporting functions

| Function | Purpose |
| --- | --- |
| `getValue( Q, Qname )` | Returns `Q` if given, else `util.Property( Qname )`. Uses `is None`, because `==` on a Buildbot property returns a renderer that is always true. |
| `getTestComponents()` | `util.Property( 'test_components' )`, defaulting to `util.Property( 'defaultTestComps' )` |
| `hasStepFailed( step )` | True when the build's result so far (`step.build.results`) is `FAILURE` or `EXCEPTION`. Used as `doStepIf` for the copy steps. |
| `skipped( results, s )` | True when a step was skipped. Used as `hideStepIf` so skipped copy steps don't clutter the web UI. |
| `runOrSkip_BuildBot( change )` | Decides whether a commit triggers builds (see Change detection) |
| `docChanged( change )` | True when any changed file is under `doc/`. The fileIsImportant function of `sched_doc`. |

`hasStepFailed` previously read a `build_status` property that Buildbot never sets, so it always returned False and failed builds were never saved. It was fixed on 2026-10-20.

## Email notifications

Commit authors are emailed only when their commit breaks a passing compile; `t.harman@utah.edu` (the `admin_email` variable, set once near the top of `master.cfg`) is emailed on every static or clang compile failure and when the worker goes missing. Four notifiers are registered in `master['services']`, all sending from `buildbot@uintah-build.chpc.utah.edu` (the `from_email` variable, next to `admin_email`).

| Notifier | Builders | Recipients | Mode |
| --- | --- | --- | --- |
| `mailNotify_compile_authors` | `Trunk:dbg-compile`, `Trunk:dbg-compile-static`, `Trunk:dbg-compile-clang` | Commit authors | `problem`: only when the builder's previous build passed |
| `mailNotify_compile_admin` | `Trunk:dbg-compile-static`, `Trunk:dbg-compile-clang` | `admin_email` (`t.harman@utah.edu`) | `failing`: every failure |
| `mailNotify_std` | `Trunk:opt-full` | Commit authors | `failing`: every failure |
| `mailNotify_try` | `Trunk:opt-full`, `Trunk:dbg-full`, try builds only | Build owner | `failing` |

All four also send on `exception` (an internal Buildbot error during the build). `mailNotify_kokkos` is defined but disabled.

### Who counts as a commit author

With `sendToInterestedUsers = True`, Buildbot emails the authors of the commits in the build plus the build's `owner` property:

- **Commit builds:** the git author string, e.g. `Todd Harman <t.harman@utah.edu>`.
- **Forced builds:** the address typed in the force form's Email field. It defaults to `email@email.com`, so type a real address.
- **Try builds:** the `--who` value, which `buildbot_try.sh` sets to your `git config user.email`.
- **Nightly clang build:** the authors of every commit since the last nightly build.

`lookup = "sci.utah.edu"` turns a bare name into an address: `bob` becomes `bob@sci.utah.edu`. Names that already contain `@` pass through unchanged.

### Compile errors in the email

When the compile step fails, every notifier puts the first compiler errors in the email body. `CompileErrorFormatter` (in `emailFormatters.py`) reads the compile step's `stdio` log only while writing a failure email, and only if that step failed; other emails are unchanged.

- **What it includes:** each `error:` or `undefined reference` line, with the gcc/ld context before it and the source line, caret and notes after it. Duplicates from parallel compiles appear once. The `make: *** [...] Error N` lines name the failed targets.
- **What it leaves out:** warnings, and `make cleanreally` errors (older versions of `make.py` produced them on every run).
- **Limits:** 60 lines; extra errors are counted, not shown.
- **No error lines found** (a timeout, a killed job): the last 20 lines of the log instead.
- **Log can't be read:** the email still goes out with a one-line note, and the error is logged in `twistd.log`.

### Notes

- `problem` mode compares with the builder's previous build, forced builds included. A failed forced build followed by a failed commit build sends no author email.
- A commit that breaks both `dbg-compile` and `dbg-compile-static` gets one author email, from `dbg-compile`: the static build only runs after `dbg-compile` passes.
- The sender was `uintah-developers@lists.utah.edu` until 2026-10-20. utah.edu's Cisco mail gateway accepted those messages but they never arrived, most likely filtered because the build host isn't allowed to send for that list.

## Operating the master

Run only one master at a time, from one base directory, and restart it fully after editing either file.

### Base directory

The master runs from `/var/lib/buildbot/masters`, set by `WorkingDirectory` in `/lib/systemd/system/buildbot-master.service`. The worker runs from `/var/lib/buildbot/worker` (`buildbot-worker.service`). No other base directory exists; the old notes pointing to `/var/lib/buildbot/master` were wrong.

Start and stop the master only through systemd. The 2026-10-20 `Address already in use` failure came from starting a second master by hand while the service was running. To check what's running:

```
systemctl status buildbot-master
ps -ef | grep -i [b]uildbot
sudo ss -ltnp | grep -E ':(8010|8031|9989) '
```

### Ports

| Port | Used by |
| --- | --- |
| 8010 | Web UI, `http://uintah-build.chpc.utah.edu:8010/` |
| 9989 | Worker connections (`pb` protocol) |
| 8031 | Try server (`Try_Userpass`) |

`Couldn't listen on any:8010: [Errno 98] Address already in use` means another master is already running.

### Applying changes

1. Check the configuration: `buildbot checkconfig master.cfg`. A clean run prints only `Config file is good!`.
2. Restart: `sudo systemctl stop buildbot-master ; sudo systemctl start buildbot-master ; sudo systemctl status buildbot-master`.
3. Confirm the startup in `twistd.log`: look for `adding N new builders` and no `BuildMaster startup failed`.

Use a full restart, not `buildbot reconfig`. `reconfig` re-reads `master.cfg` but keeps the already-imported `commonBuildSteps.py` and `emailFormatters.py`, so changes to them are ignored.

### Logs

| Log | What to look for |
| --- | --- |
| `<basedir>/twistd.log` | `sending mail (N bytes) to [...]` for each notifier email; startup errors; `gitpoller: processing changes` every minute |
| `/var/log/exim4/mainlog` | `<=` received from Buildbot, `=>` delivered to the gateway with `250 ok ... accepted`, `**` bounced. Search by exim message ID to follow one message. |

### Testing email

1. Send a test message through `localhost:25` with the Buildbot sender and confirm it arrives. This was verified on 2026-10-20: two test messages were accepted by the utah.edu gateway.
2. Force `Trunk:dbg-compile-static` with your address in the Email field. To make it fail, temporarily set the compile command to `["false"]` and restart; then `mailNotify_compile_admin` emails `t.harman@utah.edu`.
3. To test `problem` mode, force a passing build with the real compile command. Then set it to `["false"]`, restart, and force again: you should get two emails (authors + admin). A third forced failure sends only the admin email.

## Open items

The first one puts the changes described in this document into service.

- [ ] Deploy: copy `master.cfg`, `commonBuildSteps.py` and `emailFormatters.py` from this directory to `/var/lib/buildbot/masters`, run `buildbot checkconfig`, and restart with `systemctl`.
- [ ] After deploying, test email end to end: force a passing build, then a failing one, and check that both the author and admin emails arrive.
- [ ] `buildbot_testing`: keep reusing the 2022-10-20 build, or set `recompile = True` and clean up that block.
- [ ] Confirm the LaTeX sources live in top-level `doc/`, which `docChanged` assumes.
- [ ] Remove the leftover try login `("bob", "pw2")` and change the worker password from `password`.
- [ ] Replace `projectName` / `projectURL` with `title` / `titleURL` (the Buildbot 2.x names).
- [ ] `buildbot_try.sh` offers `trunk-gpu` and `kokkos-opt`, but those builders are disabled, so the try server drops the requests; it doesn't offer `Trunk:opt-full-sanitize`, which the try server accepts.
- [ ] The nightly clang build runs after any `master` commit, doc-only ones included; adding `fileIsImportant = runOrSkip_BuildBot` to its scheduler would skip those.
