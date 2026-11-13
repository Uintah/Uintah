# Uintah buildbot

Configuration for the Uintah buildbot on `uintah-build.chpc.utah.edu`
(Buildbot 2.0.1, Python 3.7, Debian 10).

Web UI: <http://uintah-build.chpc.utah.edu:8010/>

## Files

| File | Purpose |
| --- | --- |
| `master.cfg` | The buildmaster configuration: worker, change source, schedulers, compiler environments and configure lines, factories, builders, mail notifiers. The admin and sender email addresses (`admin_email`, `from_email`) are set once near the top. |
| `commonBuildSteps.py` | Helpers that add steps to a factory (`gitUpdate`, `configure`, `compile`, `makeclean`, `copyBuildDir`, `runComponentTests`), plus `runOrSkip_BuildBot` and `docChanged`, which decide whether a commit triggers builds. |
| `emailFormatters.py` | `CompileErrorFormatter`: puts the compiler errors in the body of failure emails. |
| `make.py` | Run by the compile step: `make -j <n>` then `make link_inputs`. Works under Python 2 and 3. |
| `../../buildbot_try.sh` | Developer script that submits a local patch to the try server. |

## Where it runs

| | Master | Worker |
| --- | --- | --- |
| Base directory | `/var/lib/buildbot/masters` | `/var/lib/buildbot/worker` |
| systemd unit | `buildbot-master.service` | `buildbot-worker.service` |
| Log | `twistd.log` in the base directory | `twistd.log` in the base directory |

Ports: 8010 web UI, 9989 worker connection, 8031 try server.

## Deploying a change

1. Copy `master.cfg`, `commonBuildSteps.py` and `emailFormatters.py` into
   `/var/lib/buildbot/masters`.
2. Check it: `cd /var/lib/buildbot/masters ; buildbot checkconfig master.cfg`
   should print only `Config file is good!`.
3. Restart through systemd:

   ```
   sudo systemctl stop buildbot-master ; sudo systemctl start buildbot-master ; sudo systemctl status buildbot-master
   ```

- Use a full restart, not `buildbot reconfig`: `reconfig` re-reads
  `master.cfg` but keeps the already-imported `commonBuildSteps.py` and
  `emailFormatters.py`.
- Don't start or restart the master by hand while the service is running.
  The second master fails with `Address already in use` on ports 8010 and 8031.
- `make.py` is read from each builder's checkout, so changes to it take
  effect once they are committed.

## What triggers builds

`GitPoller` checks GitHub every minute. `runOrSkip_BuildBot` skips a commit
when its message contains `skiprt`, `skip_rt` or `skip-rt`, or when every
changed file is under an ignored directory (`doc`, `src/scripts`,
`src/testprograms`, ...; the list is in the function).

| Trigger | Builders |
| --- | --- |
| Commit on `master` | `Trunk:dbg-compile`; if it passes, `Trunk:dbg-compile-static`; if that passes, `Trunk:opt-full` |
| Commit on `master` that changes `doc/` | `Documentation-Builder` |
| 22:01 nightly, only if `master` changed | `Trunk:dbg-compile-clang` |
| Force button in the web UI | Any of the trunk builders and `Documentation-Builder` |
| `buildbot_try.sh` (try server) | `Trunk:dbg-full`, `Trunk:opt-full`, `Trunk:opt-full-sanitize` |
| Testing form in the web UI | `buildbot_testing` (runs the tests against an existing build) |

## Email

All mail is sent from `from_email`. It must be an address on the build host's
own domain: utah.edu's mail gateway accepts mail sent from this host as
`uintah-developers@lists.utah.edu` but never delivers it.

| Builders | Who | When |
| --- | --- | --- |
| `Trunk:dbg-compile`, `-static`, `-clang` | Commit authors | Only when the commit breaks a build that was passing |
| `Trunk:dbg-compile-static`, `-clang` | `admin_email` | Every failure |
| `Trunk:opt-full` | Commit authors | Every failure |
| Try builds | The `--who` address (the developer's `git config user.email`) | Every failure |

When the compile step fails, the email includes its first compiler errors
with their context and the failed make targets (up to 60 lines). If no error
lines are found, it shows the end of the compile log instead.

## Failed builds

When a build fails, the cleaned build directory is saved to
`/data/buildbot/<builder>/<build number>/`, readable by group `users`. The
compile step halts the build on failure, so the clean and copy steps are
marked `alwaysRun`.

## Try server

From the top of a Uintah checkout, with your changes not yet committed:

```
src/buildbot_try.sh trunk-opt createPatch
src/buildbot_try.sh trunk-dbg myPatch my.patch testComponents "ICE MPM"
```

The results are emailed to your `git config user.email`. The script stops
if that isn't set.

## Testing a change

- Run `buildbot checkconfig master.cfg` before every restart.
- Force a build from the web UI and put your own address in the Email
  field; it defaults to `email@email.com`.
- To test the "breaks a passing build" email, force a passing build first,
  then a failing one.
- Mail delivery: `twistd.log` shows `sending mail (N bytes) to [...]`, and
  `/var/log/exim4/mainlog` shows whether utah.edu's gateway accepted it
  (`250 ok ... accepted`).
