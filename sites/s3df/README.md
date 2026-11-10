# S3DF nightly builds

This directory builds the HEAD of t1-software and of every tracked
package once a day on S3DF. It keeps a ledger of every build and renders
a static dashboard. Another site can copy the directory and change
`site.env`.

## How it works

```
cron (sdfcron001, daily)
  -> nightly.py poll      resolve HEADs -> build key -> new key? -> sbatch
  -> build.sbatch          Slurm job on a GPU node
  -> nightly.py run KEY   scratch clone at the recorded commit -> scripts/build.sh
  -> ledger + dashboard
```

- **What is tracked.** The HEAD of `T1_SOFTWARE_BRANCH` on
  `T1_SOFTWARE_REPO`, and the default-branch HEAD of every package whose
  `<PKG>_COMMIT` key has a `<PKG>_REPO` key in `base/versions.env` or
  `subgroups/*/versions.env`. The poll reads those files at the resolved
  t1-software commit. Version pins (Torch, JAX, ROOT, ...) are not tracked.
- **Only pushed commits build.** The poll and the job read t1-software
  from the repo URL, never from a local checkout. They never edit
  `versions.env`, never commit, and never move a ref. The pins of a
  nightly reach `build.sh` as `--pin KEY=SHA`.
- **Build key.** The first 12 hex characters of the sha256 of
  `{variant, t1_software_commit, pins}`. The image tag is
  `<YYYYMMDD>-<key>`, with the date of the first submission. A key that
  is already in the ledger is never built again automatically, whatever
  its status. One build runs at a time.
- **Images** go to `<IMAGES_DIR>/nightly/`, with `-latest` links, so
  nightly images never mix with release images. A nightly saves the
  t1-lartpc `.oci.tar` and `.sif` and both manifests. It builds and tests
  t1-base but does not save its archive (`build.sh --no-base-archive`):
  the base changes only with a t1-software commit. Use a release build
  for a t1-base archive.

## Setup

1. Make a state directory and copy `site.env.example` to
   `<state>/site.env`. Set `IMAGES_DIR` and `SLURM_ACCOUNT`; set
   `CACHE_DIR` to a shared layer cache if you have one.
2. Check the poll. A dry run writes only the bare cache `<state>/t1-software.git/`:
   ```bash
   uv run --script sites/s3df/nightly.py poll --dry-run --state <state>
   ```
   It prints every tracked HEAD, the key, and `would submit <key>`.
3. Install the line in `crontab.example` on `sdfcron001`. The HTTPS repo
   needs no credentials.

`--state` defaults to `$T1_NIGHTLY_STATE`. The ledger:

```
<state>/site.env
<state>/last-poll.json            time, ok, error, heads of the last poll
<state>/poll.log                  one line per poll decision
<state>/t1-software.git/          bare cache, used to read versions.env
<state>/builds/<key>/record.json
<state>/builds/<key>/build.log    latest attempt; attempt-N.log for earlier
<state>/builds/<key>/review.json  written only by `review`
<state>/site/                     default dashboard
```

## Statuses

| Status | Set by | Next |
|---|---|---|
| `queued` | `poll` or `retry`, after `sbatch` succeeds | `running` |
| `running` | `run`, at job start | `success` or `failed` |
| `success` | `run`, `build.sh` exit 0 | final |
| `failed` | `run`, `build.sh` exit not 0; or `poll`, from `sacct` (`OUT_OF_MEMORY`, `FAILED`, ...) | waits for `review` and `retry` |
| `interrupted` | `poll`, from `sacct` (`PREEMPTED`, `TIMEOUT`, `NODE_FAIL`) | resubmitted once automatically; after a second interruption, waits for `retry` |
| `cancelled` | `poll`, from `sacct` (`CANCELLED`) | waits for `retry` |

`failed_stage` holds the last `== <stage> ==` line of `build.log`, or
`checkout`, `cache lock` (another build holds the shared cache), or
`slurm <STATE>` when the job ended without a result.

## Review workflow

```bash
nightly.py status --state <state>                 # one line per build, newest first
less <state>/builds/<key>/build.log               # find the cause
nightly.py review <key> --note "pimm-data HEAD drops a module" --state <state>
nightly.py retry <key> --state <state>            # after an upstream or recipe fix
```

A failed build shows "needs review" on the dashboard until a review
records the current attempt. `retry` submits one new attempt with the
same commit set and tag. A fix in t1-software itself is a new commit, so
it gives a new key and builds on the next poll.

## Retention

`poll` deletes the `.oci.tar` and `.sif` files of successful nightlies
older than the newest `NIGHTLY_KEEP` (default 7). Files that a `-latest`
link points at stay. Records, logs and manifests are never deleted.

## Nightly link

After each successful build, `run` points `<IMAGES_DIR>/t1-lartpc-<variant>-nightly.sif`
at that build's `.sif` (a relative link into `nightly/`). A failed build
leaves the link alone. Users can load the nightly image from this fixed
path. The newest build is never pruned, so the link does not dangle.

## Dashboard

`poll`, `run`, `review` and `retry` re-render the dashboard into
`DASHBOARD_DIR` (default `<state>/site`). To write it elsewhere:

```bash
nightly.py render --state <state> --out DIR
```

Open `DIR/index.html` in a browser. The pages work without JavaScript.
One small inline script adds a light/dark toggle and remembers the choice
in the browser; without it the pages follow the system theme. Links are
relative, except the commit links to github.com. The pages work from the
filesystem or from any static web server. The index shows a banner when
the last poll failed, or when it is older than twice `POLL_INTERVAL_HOURS`.

Each record stores `repos`, the clone URL of every tracked package. The
build page links each commit to `<repo>/commit/<sha>` when the URL is on
github.com. A record without `repos` shows plain commits.

On S3DF, `DASHBOARD_DIR` is `/sdf/data/neutrino/public_html/t1-software`,
published at <https://s3df.slac.stanford.edu/data/neutrino/t1-software/>.
The site is public. It does not publish logs: a build page shows the log
file location and the last 200 lines of `build.log`.

## Time limit

`SLURM_TIME` starts at 16 h. After the first nightly finishes, set it to
that build's wall time plus 50 % (`sacct -j <job> -o Elapsed`). A build
that hits the limit ends as `TIMEOUT` and is resubmitted once.

## Tests

```bash
uv run --python 3.11 python -m unittest discover -s sites/s3df/tests -v
```

They use local git remotes and fake `sbatch`/`sacct`, and run in seconds.
