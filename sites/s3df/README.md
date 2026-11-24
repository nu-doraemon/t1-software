# S3DF nightly builds

This directory builds the HEAD of t1-software and of every tracked
package once a day on S3DF. It keeps a ledger of every build and renders
a static dashboard. Another site can copy the directory and change
`site.env`.

## How it works

```
cron (sdfcron001, daily)
  -> nightly.py poll      resolve HEADs -> one build key per subgroup; list release tags
        |-> sbatch build.sbatch   GPU node: nightly.py run KEY...      (new subgroup keys)
        '-> sbatch build.sbatch   CPU node: nightly.py convert REL...  (new release tags)
  -> ledger + dashboard
```

Two kinds of images:

- **Nightly builds** (experimental): this repository's own images, built
  from the HEAD of every tracked package. One table per subgroup.
- **Releases**: the upstream project's own release image, converted to
  Apptainer. Today: t1-wc, from LUCiD's `ghcr.io/cider-ml/lucid`.
  T1-LArTPC has no release yet.

### Nightly builds

- **What is tracked.** The HEAD of `T1_SOFTWARE_BRANCH` on
  `T1_SOFTWARE_REPO`, and the default-branch HEAD of every package whose
  `<PKG>_COMMIT` key has a `<PKG>_REPO` key in `base/versions.env` or
  `subgroups/*/versions.env`. The poll reads those files at the resolved
  t1-software commit. Version pins (Torch, JAX, ROOT, ...) are not tracked.
- **Only pushed commits build.** The poll and the job read t1-software
  from the repo URL, never from a local checkout. They never edit
  `versions.env`, never commit, and never move a ref. The pins of a
  nightly reach `build.sh` as `--pin KEY=SHA`.
- **One record per subgroup.** The build key is the first 12 hex
  characters of the sha256 of `{variant, subgroup, t1_software_commit,
  pins}`. The pins are the base's and that subgroup's own. So a goop
  commit rebuilds t1-lartpc only, a LUCiD commit rebuilds t1-wc only, and
  a t1-software commit rebuilds both. The image tag is
  `<YYYYMMDD>-<key>`, with the date of the first submission. A key that
  is already in the ledger is never built again automatically, whatever
  its status.
- **One job, one `build.sh` run per record.** The new records of a poll
  share one Slurm job. It builds them in `SUBGROUPS` order (lartpc, then
  wc), each with `build.sh --subgroup`. The base comes from the layer
  cache on the second run. A failed subgroup never stops or fails the
  other one: each has its own status, review, retry and `-nightly.sif`
  link. Each record keeps its own `build.log`; the job's Slurm output is
  `<state>/jobs/<jobid>.log`. One build job runs at a time.
- **Images** go to `<IMAGES_DIR>/nightly/`, with `-latest` links, so
  nightly images never mix with release images. A nightly saves the
  subgroup's `.oci.tar` and `.sif` and all manifests. It builds and tests
  t1-base but does not save its archive (`build.sh --no-base-archive`):
  the base changes only with a t1-software commit. Use a release build
  for a t1-base archive.
- **t1-wc is experimental.** Its PhotonSim uses Geant4 11.3.2 on its own
  prefix, while the base stays on Geant4 10.7.4. The dashboard marks it
  "experimental, unofficial". The image runs on a GPU (`apptainer exec
  --nv`); its test checks that JAX uses the GPU.

### Releases

For each subgroup in `RELEASE_SUBGROUPS`, `poll` asks the image registry
(anonymous HTTPS) for the `vX.Y.Z` tags and takes the newest
`RELEASE_KEEP` (default 3). A tag with no record gets one conversion
record; all new records of a poll share one CPU job (`CONVERT_*` keys).
`nightly.py convert` then, per tag:

1. `apptainer pull` of the image at its digest (not the tag) into a scratch
   directory (`$LSCRATCH`), with Apptainer's cache and temp dirs there too;
2. inspects the image (`image_info.py`, including `/opt/VERSIONS`);
3. runs the subgroup's release test in the image
   (`subgroups/wc/tests/test_photonsim_lucid.sh`; with `--nv` when the node
   has a GPU);
4. writes the `.sif` and `<name>.sif.sha256` to
   `<IMAGES_DIR>/<name>/<name>-<tag>.sif` (`<name>` is `lucid`);
5. points `<IMAGES_DIR>/<name>-latest.sif` at the newest converted version.

A failed step leaves no `.sif`. A tag whose digest moved after its
conversion is logged and is not converted again. Files past the newest
`RELEASE_KEEP` versions are deleted; the `-latest` target stays.
Release images are public. Another HPC centre can run
`apptainer pull docker://<image>:<tag>` itself, or copy the `.sif` and
check its sha256.

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
<state>/last-release-check.json   time, ok, error of the last registry check
<state>/poll.log                  one line per poll decision
<state>/t1-software.git/          bare cache, used to read versions.env
<state>/jobs/<jobid>.log          Slurm output of each job
<state>/builds/<key>/record.json  one per subgroup commit set; has "subgroup"
<state>/builds/<key>/build.log    latest attempt; attempt-N.log for earlier
<state>/builds/<key>/review.json  written only by `review`
<state>/releases/<sub>/<tag>/record.json
<state>/releases/<sub>/<tag>/convert.log   latest attempt; attempt-N.log for earlier
<state>/site/                     default dashboard
```

## Statuses

| Status | Set by | Next |
|---|---|---|
| `queued` | `poll` or `retry`, after `sbatch` succeeds | `running` |
| `running` | `run` or `convert`, when that record starts | `success` or `failed` |
| `success` | `run`: `build.sh` exit 0; `convert`: every step passed | final |
| `failed` | `run`, `build.sh` exit not 0; or `poll`, from `sacct` (`OUT_OF_MEMORY`, `FAILED`, ...) | waits for `review` and `retry` |
| `interrupted` | `poll`, from `sacct` (`PREEMPTED`, `TIMEOUT`, `NODE_FAIL`) | resubmitted once automatically; after a second interruption, waits for `retry` |
| `cancelled` | `poll`, from `sacct` (`CANCELLED`) | waits for `retry` |

`failed_stage` holds the last `== <stage> ==` line of `build.log`, or
`checkout`, `cache lock` (another build holds the shared cache), or
`slurm <STATE>` when the job ended without a result.

## Review workflow

```bash
nightly.py status --state <state>                 # one line per build and release
less <state>/builds/<key>/build.log               # find the cause
nightly.py review <key> --note "pimm-data HEAD drops a module" --state <state>
nightly.py retry <key> --state <state>            # after an upstream or recipe fix
nightly.py retry-release wc/v0.6.0 --state <state>   # a failed conversion
```

A failed build shows "needs review" on the dashboard until a review
records the current attempt. `retry` submits one new attempt with the
same commit set and tag. A fix in t1-software itself is a new commit, so
it gives a new key and builds on the next poll.

## Retention

`poll` deletes the `.oci.tar` and `.sif` files of successful nightlies
older than the newest `NIGHTLY_KEEP` (default 7) of each subgroup. Files
that a `-latest` or `-nightly` link points at stay. Release `.sif` files
follow `RELEASE_KEEP` (see above). Records, logs and manifests are never deleted.

## Nightly link

After each successful build, `run` points
`<IMAGES_DIR>/t1-<subgroup>-<variant>-nightly.sif` at that build's `.sif`
(a relative link into `nightly/`; one link per subgroup in `SUBGROUPS` of
`nightly.py`: lartpc, wc). A failed build leaves its link alone; the
nightly page shows each link in its subgroup's section.
Users can load the nightly images from these fixed paths. The newest
build is never pruned, so the links do not dangle.

## Dashboard

`poll`, `run`, `convert`, `review`, `retry` and `retry-release` re-render the dashboard into
`DASHBOARD_DIR` (default `<state>/site`). To write it elsewhere:

```bash
nightly.py render --state <state> --out DIR
```

Two pages: `index.html` lists the releases, one section per subgroup (a
subgroup with no release says so and links its nightly). `nightly.html`
has one section per subgroup with the nightly image link, the main
versions, how to use a GPU, and the table of its builds. Open
`DIR/index.html` in a browser. The pages work without JavaScript.
One small inline script adds a light/dark toggle and remembers the choice
in the browser; without it the pages follow the system theme. Links are
relative, except the commit links to github.com. The pages work from the
filesystem or from any static web server. The nightly page shows a banner
when the last poll failed, or when it is older than twice
`POLL_INTERVAL_HOURS`; the index shows one when the last release check
failed.

After a successful build, `run` inspects the `.sif` with
`apptainer exec IMAGE python - < image_info.py` and writes
`builds/<key>/image-info.json`. The build page shows the main versions
(PyTorch, JAX, ROOT, Geant4, Python) and, for t1-wc, the LUCiD and
PhotonSim commits from `/opt/VERSIONS`. It links to `<key>-image.html`.
That page lists the core stack as pinned (from the base manifest) against
installed, marks a mismatch "differs", and lists every Python package. A
failed inspection is reported and does not fail the build; the page then
has no base-image section.

Each record stores `repos`, the clone URL of every tracked package. The
build page links each commit to `<repo>/commit/<sha>` when the URL is on
github.com. A record without `repos` shows plain commits.

On S3DF, `DASHBOARD_DIR` is `/sdf/data/neutrino/public_html/t1-software`,
published at <https://s3df.slac.stanford.edu/data/neutrino/t1-software/>.
The site is public. It does not publish logs: a build page shows the log
file location and the last 200 lines of `build.log`.

## Time limit

`SLURM_TIME` starts at 16 h; it covers all subgroups of one job. After the first nightly finishes, set it to
that build's wall time plus 50 % (`sacct -j <job> -o Elapsed`). A build
that hits the limit ends as `TIMEOUT` and is resubmitted once.

## Tests

```bash
uv run --python 3.11 python -m unittest discover -s sites/s3df/tests -v
```

They use local git remotes and fake `sbatch`/`sacct`, and run in seconds.
