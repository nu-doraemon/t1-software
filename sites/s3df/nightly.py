#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Nightly HEAD builds of t1-software on S3DF, and Apptainer copies of
upstream release images.

Two layers do the work. scripts/build.sh is the generic layer: it builds
one commit set and knows nothing about the nightly. This script is the
site layer. Once a day, cron runs `poll` on the cron node.

Nightly builds. `poll` resolves the HEAD of t1-software and of every
package that has a <PKG>_REPO key in versions.env. Each subgroup overlay
(lartpc, wc) has its own build key, from the variant, the t1-software
commit, the base pins and that subgroup's own pins. So a package change
rebuilds only the subgroup that uses it. Every subgroup key that is new
to the ledger gets a record; one Slurm job then runs `run KEY...` on a
GPU node. `run` builds the records one after another, in SUBGROUPS
order: it fetches t1-software at the recorded commit into a scratch
clone and calls build.sh with --subgroup, --pin and --tag. A failed
subgroup never stops the next one. No t1-software checkout is modified.

Releases. For each subgroup in RELEASE_SUBGROUPS, `poll` lists the
release tags (vX.Y.Z) of that subgroup's upstream image (RELEASES). The
newest RELEASE_KEEP tags that have no record get one CPU Slurm job,
which runs `convert SUBGROUP/TAG...`: apptainer pull of the image at
its digest, an inspection, the subgroup's release test inside the
image, a sha256 file, and the <name>-latest.sif link.

Ledger, under the state directory (--state or $T1_NIGHTLY_STATE):

    site.env                         site values (paths, Slurm account, ...)
    last-poll.json                   {time, ok, error, heads}
    last-release-check.json          {time, ok, error}
    poll.log                         one line per poll decision
    t1-software.git/                 bare cache, used to read versions.env
    jobs/<jobid>.log                 Slurm output of each job
    builds/<key>/record.json         one record per subgroup commit set
    builds/<key>/build.log           latest attempt; attempt-N.log for earlier
    builds/<key>/review.json         written only by `review`
    releases/<sub>/<tag>/record.json one record per converted release
    releases/<sub>/<tag>/convert.log latest attempt; attempt-N.log for earlier

Statuses (builds and releases):

    queued       poll/retry, after sbatch succeeds
    running      run/convert, at the start of that record
    success      run: build.sh exit 0; convert: every step passed
    failed       run/convert, a step failed; or poll, from sacct
    interrupted  poll, on PREEMPTED, TIMEOUT or NODE_FAIL; poll resubmits
                 a build once, then it waits for `retry`
    cancelled    poll, on CANCELLED; waits for `retry`

A key or a release tag that has a record is never built again
automatically, whatever its status. Only `retry` and `retry-release`
rebuild it. At most one build job and one conversion job are queued or
running.
"""
import argparse
import contextlib
import fcntl
import getpass
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import dashboard  # sites/s3df/dashboard.py; a script's own directory is on sys.path

T1 = "T1_SOFTWARE_COMMIT"
ACTIVE = ("queued", "running")
# Subgroup overlays, in build order. Each has its own build records and
# its own t1-<subgroup>-<variant>-nightly.sif link.
SUBGROUPS = ("lartpc", "wc")
RETRYABLE = ("failed", "interrupted", "cancelled")
SLURM_ALIVE = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING",
               "COMPLETING", "RESIZING"}
SLURM_INTERRUPT = {"PREEMPTED", "TIMEOUT", "NODE_FAIL"}
GIT_TIMEOUT = 120  # s; one ls-remote or shallow fetch takes a few seconds
REGISTRY_TIMEOUT = 60  # s per registry request
REPO_ROOT = Path(__file__).resolve().parents[2]

# Upstream release images, per subgroup, that RELEASE_SUBGROUPS converts:
#   image   public OCI image; its vX.Y.Z tags are the releases
#   repo    source repository, for the release, recipe and package links
#   recipe  the image recipe's directory in repo
#   test    script in this repository, run inside the converted image
#           (apptainer exec IMAGE bash -s < test)
RELEASES = {
    "wc": {"image": "ghcr.io/cider-ml/lucid", "repo": "https://github.com/CIDeR-ML/LUCiD",
           "recipe": "container", "test": "subgroups/wc/tests/test_photonsim_lucid.sh"},
}
RELEASE_TAG = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
MANIFEST_ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json"])

# site.env keys. None means: required (REQUIRED) or computed in load_site.
SITE_DEFAULTS = {
    "T1_SOFTWARE_REPO": "https://github.com/nu-doraemon/t1-software.git",
    "T1_SOFTWARE_BRANCH": "main",
    "VARIANT": "cuda12",
    "IMAGES_DIR": None,
    "CACHE_DIR": "",
    "WORK_DIR": "",
    "NIGHTLY_KEEP": "7",
    "POLL_INTERVAL_HOURS": "24",
    "DASHBOARD_DIR": None,
    "UV": None,
    "SLURM_ACCOUNT": None,
    "SLURM_PARTITION": "ampere",
    "SLURM_QOS": "normal",
    "SLURM_GPUS": "1",
    "SLURM_CPUS": "8",
    "SLURM_MEM": "32G",
    "SLURM_TIME": "16:00:00",
    "RELEASE_SUBGROUPS": "",
    "RELEASE_KEEP": "3",
    "CONVERT_PARTITION": "milano",
    "CONVERT_GPUS": "0",
    "CONVERT_CPUS": "8",
    "CONVERT_MEM": "32G",
    "CONVERT_TIME": "02:00:00",
    "SBATCH": "sbatch",
    "SACCT": "sacct",
    "APPTAINER": "apptainer",
}
REQUIRED = ("IMAGES_DIR", "SLURM_ACCOUNT")

KV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class UsageError(Exception):
    """Bad configuration or arguments; exit code 2."""


class PollError(Exception):
    """A HEAD or registry lookup failed; the poll submits nothing for it."""


class SubmitError(Exception):
    """sbatch failed."""


class SacctError(Exception):
    """sacct failed."""


class StepError(Exception):
    """A conversion step failed; the message is the step name."""


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_kv(text, strip_quotes=False):
    """KEY=VALUE lines of a versions.env or site.env; other lines are ignored."""
    out = {}
    for line in text.splitlines():
        m = KV_LINE.match(line.strip())
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if strip_quotes and len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load_site(state):
    path = state / "site.env"
    if not path.exists():
        raise UsageError(f"site.env: {path} not found")
    given = parse_kv(path.read_text(), strip_quotes=True)
    for key in given:
        if key not in SITE_DEFAULTS:
            raise UsageError(f"site.env: unknown key {key}")
    site = {}
    for key, default in SITE_DEFAULTS.items():
        value = given.get(key, "")
        site[key] = value if value != "" else (default or "")
    for key in REQUIRED:
        if not site[key]:
            raise UsageError(f"site.env: {key} is required")
    for sub in release_subgroups(site):
        if sub not in RELEASES:
            raise UsageError(f"site.env: RELEASE_SUBGROUPS: no release image for {sub} "
                             f"(known: {', '.join(sorted(RELEASES))})")
    if not site["DASHBOARD_DIR"]:
        site["DASHBOARD_DIR"] = str(state / "site")
    # uv sets $UV to its own path for the scripts it runs. Cron's PATH
    # lacks ~/.local/bin, so the Slurm job needs this absolute path.
    uv = site["UV"] or os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise UsageError("site.env: UV not found")
    site["UV"] = os.path.abspath(uv)
    return site


def release_subgroups(site):
    return [s.strip() for s in site["RELEASE_SUBGROUPS"].split(",") if s.strip()]


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def write_json(path, data):
    """Write to a temp file in the same directory, then rename over."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


@contextlib.contextmanager
def ledger_lock(state):
    """Exclusive lock on <state>/.lock. fcntl.flock is not reentrant."""
    state.mkdir(parents=True, exist_ok=True)
    with open(state / ".lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build_key(variant, t1_commit, pins, subgroup):
    blob = json.dumps({"variant": variant, "t1_software_commit": t1_commit, "pins": pins,
                       "subgroup": subgroup}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def build_dir(state, key):
    return state / "builds" / key


def record_path(state, key):
    return build_dir(state, key) / "record.json"


def _order(record):
    """Sort key: queued_at, then the last Slurm job id, then SUBGROUPS order.

    queued_at has one-second resolution, so two records can tie. Slurm job
    ids grow monotonically, which breaks the tie; records of one job keep
    their build order.
    """
    ids = record.get("slurm_job_ids") or []
    last = int(ids[-1]) if ids and str(ids[-1]).isdigit() else -1
    sub = record.get("subgroup")
    rank = -SUBGROUPS.index(sub) if sub in SUBGROUPS else -len(SUBGROUPS)
    return (record.get("queued_at") or "", last, rank)


def load_records(state):
    """All build records, newest first (see _order). A record from before
    per-subgroup builds has no subgroup; those built only t1-lartpc.
    """
    records = []
    for path in (state / "builds").glob("*/record.json"):
        rec = load_json(path)
        if rec is not None:
            rec.setdefault("subgroup", "lartpc")
            records.append(rec)
    records.sort(key=_order, reverse=True)
    return records


def save_record(state, record):
    write_json(record_path(state, record["key"]), record)


def review_state(record, review):
    if (record["status"] in RETRYABLE and review is not None
            and review.get("attempt") == record["attempts"]):
        return "reviewed"
    if record["status"] == "failed":
        return "needs review"
    return ""


def log(state, msg, *, dry_run=False):
    print(msg, flush=True)
    if not dry_run:
        with open(state / "poll.log", "a") as f:
            f.write(f"{now_iso()} {msg}\n")


def git(*args):
    """Run git; any failure is a PollError. Returns stdout."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    cmd = ["git", *args]
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=GIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise PollError(f"{' '.join(cmd)}: timed out after {GIT_TIMEOUT} s") from None
    if r.returncode != 0:
        raise PollError(f"{' '.join(cmd)}: {r.stderr.strip()}")
    return r.stdout


def ls_remote(repo, ref):
    out = git("ls-remote", repo, ref).split()
    if not out:
        raise PollError(f"git ls-remote {repo} {ref}: no such ref")
    return out[0]


def resolve_heads(state, site):
    """HEADs of t1-software and of every tracked package, at the HEAD of
    T1_SOFTWARE_BRANCH.

    Returns (heads, repos, owners, subgroups). heads and repos are keyed by
    <PKG>_COMMIT; owners maps each package key to "base" or the subgroup
    whose versions.env names it; subgroups lists the SUBGROUPS present at
    that commit, in build order.
    """
    repo, branch = site["T1_SOFTWARE_REPO"], site["T1_SOFTWARE_BRANCH"]
    commit = ls_remote(repo, f"refs/heads/{branch}")
    cache = state / "t1-software.git"
    if not cache.exists():
        git("init", "--bare", "--quiet", str(cache))
    git("-C", str(cache), "fetch", "--quiet", "--depth", "1", repo, commit)
    present = {line.rsplit("/", 1)[-1] for line in
               git("-C", str(cache), "ls-tree", "--name-only", commit, "subgroups/").splitlines()}
    subgroups = [s for s in SUBGROUPS if s in present]
    heads, repos, owners = {T1: commit}, {T1: repo}, {}
    for path, owner in [("base/versions.env", "base"),
                        *((f"subgroups/{s}/versions.env", s) for s in subgroups)]:
        try:
            text = git("-C", str(cache), "show", f"{commit}:{path}")
        except PollError:
            if owner == "base":
                raise
            continue  # a subgroup without its own pins
        pins = parse_kv(text)
        for key in sorted(pins):
            if key.endswith("_COMMIT") and pins.get(key[: -len("_COMMIT")] + "_REPO"):
                url = pins[key[: -len("_COMMIT")] + "_REPO"]
                heads[key] = ls_remote(url, "HEAD")
                repos[key] = url
                owners[key] = owner
    return heads, repos, owners, subgroups


def subgroup_pins(heads, owners, subgroup):
    """The pins one subgroup's build uses: the base's and its own."""
    return {k: v for k, v in heads.items() if k != T1 and owners.get(k) in ("base", subgroup)}


def nightly_link(site, variant, subgroup):
    """Path of the symlink to a subgroup's newest nightly .sif of a variant, next to nightly/."""
    return Path(site["IMAGES_DIR"]) / f"t1-{subgroup}-{variant}-nightly.sif"


def replace_link(link, target):
    """Point link at target with a relative path, atomically. Returns success."""
    tmp = link.with_name(f".{link.name}.tmp.{os.getpid()}")
    try:
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(os.path.relpath(target, link.parent))
        os.replace(tmp, link)
    except OSError as e:
        print(f"nightly: cannot update {link}: {e}", file=sys.stderr, flush=True)
        tmp.unlink(missing_ok=True)
        return False
    return True


def update_nightly_link(site, rec):
    """Point the record's subgroup nightly_link at its .sif. Returns False
    when the image file is missing or the link cannot be written.
    """
    sif = dashboard.image_file(rec)
    if not sif.is_file():
        return False
    return replace_link(nightly_link(site, rec["variant"], rec["subgroup"]), sif)


def _pip_version(pip, prefix):
    """Version of the pip package named prefix, else the first one that starts with it."""
    if prefix in pip:
        return pip[prefix]
    return next((v for n, v in sorted(pip.items()) if n.startswith(prefix)), "")


def _differs(pinned, installed):
    return bool(pinned) and not (installed == pinned or installed.startswith(pinned + "."))


def compose_image_info(raw, base_pins, size, built_at, sif):
    """Core-stack rows (pinned vs installed), provenance and the package
    list of one image.

    raw is the output of image_info.py; base_pins is the base manifest's
    pins ({} for an image not built here). A pin of "3.11" matches an
    installed "3.11.17". jax_gpu: the image's JAX has a CUDA backend
    (a jax-cuda* plugin, or a CUDA build of conda's jaxlib).
    """
    pip = dict(raw["pip"])
    conda = raw["conda"]
    jax = _pip_version(pip, "jax")
    spec = [
        ("OS", "", raw["os"], raw["os"], ""),
        ("Python", "python_version", raw["python"], raw["python"], "conda-forge"),
        ("PyTorch", "torch_version", _pip_version(pip, "torch"), _pip_version(pip, "torch"), ""),
        ("JAX", "jax_version", f"{jax} (jaxlib {_pip_version(pip, 'jaxlib')})" if jax else "", jax, ""),
        ("CUDA runtime", "", _pip_version(pip, "nvidia-cuda-runtime"), "", "pip"),
        ("cuDNN", "", _pip_version(pip, "nvidia-cudnn"), "", "pip"),
        ("Triton", "", _pip_version(pip, "triton"), "", ""),
        ("NumPy", "", _pip_version(pip, "numpy"), "", ""),
        ("h5py / HDF5", "", f"{_pip_version(pip, 'h5py')} / {conda.get('hdf5', '')}", "", ""),
        ("ROOT", "root_version", conda.get("root_base", ""), conda.get("root_base", ""), "root_base, PyROOT"),
        ("Geant4", "geant4_version", conda.get("geant4", ""), conda.get("geant4", ""), "noqt build"),
        ("g++ (conda)", "gxx_version", conda.get("gxx_linux-64", ""), conda.get("gxx_linux-64", ""),
         "the compiler under /opt/conda"),
        ("CMake", "", raw.get("cmake", ""), "", ""),
    ]
    core = []
    for name, pin_key, shown, compare, note in spec:
        pinned = base_pins.get(pin_key, "") if pin_key else ""
        core.append({"name": name, "pinned": pinned, "installed": shown or "-", "note": note,
                     "differs": _differs(pinned, compare)})
    jax_gpu = (any(n.startswith("jax-cuda") for n in pip)
               or "cuda" in (raw.get("conda_builds") or {}).get("jaxlib", ""))
    return {"core": core, "pip": raw["pip"], "conda_count": len(conda), "size": size,
            "built_at": built_at, "sif": str(sif), "versions": raw.get("versions") or {},
            "jax_gpu": jax_gpu}


def inspect_image(site, sif):
    """image_info.py's JSON for one .sif. Raises RuntimeError or ValueError."""
    script = Path(__file__).with_name("image_info.py").read_text()
    r = subprocess.run([site["APPTAINER"], "exec", str(sif), "python", "-"], input=script,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-300:])
    return json.loads(r.stdout)


def write_image_info(state, site, rec):
    """Inspect the record's .sif and write builds/<key>/image-info.json. A
    failure is reported and never changes the build result. Returns success.
    """
    images = Path(rec["images_dir"])
    sif = dashboard.image_file(rec)
    if not sif.is_file():
        return False
    try:
        raw = inspect_image(site, sif)
        base = load_json(images / f"t1-base-{rec['variant']}-{rec['tag']}.manifest.json") or {}
        own = load_json(images / f"t1-{rec['subgroup']}-{rec['variant']}-{rec['tag']}.manifest.json") or {}
        info = compose_image_info(raw, base.get("pins") or {}, sif.stat().st_size,
                                  own.get("built_at"), sif)
        write_json(build_dir(state, rec["key"]) / "image-info.json", info)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as e:
        print(f"nightly: image info failed: {e}", file=sys.stderr, flush=True)
        return False
    return True


def sacct_state(site, job_id):
    """First word of the job's Slurm state; "" when sacct does not know it yet."""
    cmd = [site["SACCT"], "-n", "-X", "-P", "-j", job_id, "-o", "State"]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise SacctError("timed out") from None
    if r.returncode != 0:
        raise SacctError(r.stderr.strip())
    lines = r.stdout.strip().splitlines()
    return lines[0].split()[0] if lines and lines[0].split() else ""


def sbatch(state, site, name, resources, command):
    """Submit build.sbatch running `nightly.py <command...>`; returns the job id.

    Each job's Slurm output goes to <state>/jobs/<jobid>.log; each record
    writes its own log besides.
    """
    state = state.resolve()
    here = Path(__file__).resolve()
    (state / "jobs").mkdir(parents=True, exist_ok=True)
    cmd = [
        site["SBATCH"], "--parsable",
        "--job-name", name,
        "--account", site["SLURM_ACCOUNT"],
        *resources,
        "--output", str(state / "jobs" / "%j.log"),
        "--open-mode", "append",
        # The poll owns retries; Slurm's own requeue would race it.
        "--no-requeue",
        "--export", "ALL",
        str(here.parent / "build.sbatch"),
        site["UV"], str(here), *command, "--state", str(state),
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SubmitError(str(e)) from None
    if r.returncode != 0:
        raise SubmitError(r.stderr.strip() or f"exit {r.returncode}")
    return r.stdout.strip().split(";", 1)[0].strip()


def submit(state, site, records):
    """One build job for the records, built in SUBGROUPS order."""
    records = sorted(records, key=lambda r: SUBGROUPS.index(r["subgroup"]))
    keys = [r["key"] for r in records]
    resources = ["--partition", site["SLURM_PARTITION"], "--qos", site["SLURM_QOS"],
                 "--gpus", site["SLURM_GPUS"], "--cpus-per-task", site["SLURM_CPUS"],
                 "--mem", site["SLURM_MEM"], "--time", site["SLURM_TIME"]]
    return sbatch(state, site, f"t1-nightly-{keys[0]}", resources, ["run", *keys])


def rotate_log(directory, name, attempt):
    """Move <name> to attempt-<attempt>.log; returns the undo, or None."""
    current, old = directory / name, directory / f"attempt-{attempt}.log"
    if not current.exists():
        return None
    os.replace(current, old)
    return lambda: os.replace(old, current)


def resubmit(state, site, records):
    """Submit a new attempt of existing build records in one job. Returns
    the job id. On sbatch failure the records and their logs stay as they were.
    """
    undo = [rotate_log(build_dir(state, r["key"]), "build.log", r["attempts"]) for r in records]
    try:
        job_id = submit(state, site, records)
    except SubmitError:
        for u in undo:
            if u:
                u()
        raise
    for rec in records:
        rec.update(attempts=rec["attempts"] + 1, status="queued", queued_at=now_iso(),
                   started_at=None, finished_at=None, failed_stage=None)
        rec["slurm_job_ids"].append(job_id)
        save_record(state, rec)
    return job_id


def new_record(site, subgroup, key, t1_commit, pins, repos, changed):
    queued_at = now_iso()
    return {
        "key": key,
        "subgroup": subgroup,
        "variant": site["VARIANT"],
        "t1_software_commit": t1_commit,
        "pins": pins,
        "changed": changed,
        "repos": {k: v for k, v in repos.items() if k == T1 or k in pins},
        "status": "queued",
        "failed_stage": None,
        "attempts": 1,
        "slurm_job_ids": [],
        "queued_at": queued_at,
        "started_at": None,
        "finished_at": None,
        "tag": f"{queued_at[:4]}{queued_at[5:7]}{queued_at[8:10]}-{key}",
        "images_dir": str(Path(site["IMAGES_DIR"]) / "nightly"),
    }


def reconcile(state, site, records, dry_run, save, name):
    """Settle queued/running records whose Slurm job ended without a result."""
    for rec in records:
        if rec["status"] not in ACTIVE or not rec["slurm_job_ids"]:
            continue
        try:
            st = sacct_state(site, rec["slurm_job_ids"][-1])
        except SacctError as e:
            log(state, f"sacct failed for {name(rec)}: {e}", dry_run=dry_run)
            continue
        if not st or st in SLURM_ALIVE:
            continue
        if st in SLURM_INTERRUPT:
            new = "interrupted"
        elif st == "CANCELLED":
            new = "cancelled"
        else:
            new = "failed"
        if dry_run:
            log(state, f"would reconcile {name(rec)}: {st} -> {new}", dry_run=True)
            continue
        rec["status"] = new
        if new == "failed" and rec["failed_stage"] is None:
            rec["failed_stage"] = f"slurm {st}"
        rec["finished_at"] = now_iso()
        save(state, rec)
        log(state, f"reconcile {name(rec)}: {st} -> {new}")


def _poll(state, site, dry_run):
    records = load_records(state)
    prev = load_json(state / "last-poll.json") or {}
    prev_heads = prev.get("heads") or {}

    reconcile(state, site, records, dry_run, save_record, lambda r: r["key"])
    active = [r for r in records if r["status"] in ACTIVE]

    try:
        heads, repos, owners, subgroups = resolve_heads(state, site)
    except PollError as e:
        if not dry_run:
            write_json(state / "last-poll.json",
                       {"time": now_iso(), "ok": False, "error": str(e), "heads": prev_heads})
        log(state, f"error: {e}", dry_run=dry_run)
        return 1
    if dry_run:
        for name, sha in heads.items():
            print(f"{name} {sha}")
    # Resolve HEADs before any submission: a failed lookup submits nothing,
    # not even the automatic retry of an interrupted build.
    if not active:
        pending = [r for r in records if r["status"] == "interrupted" and r["attempts"] == 1]
        if pending:
            keys = " ".join(r["key"] for r in pending)
            if dry_run:
                log(state, f"would auto-retry {keys}", dry_run=True)
            else:
                try:
                    job_id = resubmit(state, site, pending)
                except SubmitError as e:
                    log(state, f"sbatch failed: {e}")
                    return 1
                log(state, f"auto-retry {keys} job {job_id}")
            active = pending

    ok = {"time": now_iso(), "ok": True, "error": None, "heads": heads}
    if active:
        if not dry_run:
            write_json(state / "last-poll.json", ok)
        log(state, f"build active: {' '.join(r['key'] for r in active)}", dry_run=dry_run)
        return 0

    new, built = [], []
    for sub in subgroups:
        pins = subgroup_pins(heads, owners, sub)
        key = build_key(site["VARIANT"], heads[T1], pins, sub)
        if dry_run:
            print(f"key {sub} {key}")
        existing = load_json(record_path(state, key))
        if existing is not None:
            built.append(f"{sub} {key} ({existing['status']})")
            continue
        changed = {n: [prev_heads.get(n), sha] for n, sha in [(T1, heads[T1]), *pins.items()]
                   if prev_heads.get(n) != sha}
        new.append(new_record(site, sub, key, heads[T1], pins, repos, changed))
    if not new:
        log(state, "no change" if heads == prev_heads else f"already built {', '.join(built)}",
            dry_run=dry_run)
    elif dry_run:
        for rec in new:
            log(state, f"would submit {rec['subgroup']} {rec['key']} "
                       f"changed {','.join(rec['changed'])}", dry_run=True)
        return 0
    else:
        for rec in new:
            build_dir(state, rec["key"]).mkdir(parents=True, exist_ok=True)
            save_record(state, rec)
        try:
            job_id = submit(state, site, new)
        except SubmitError as e:
            for rec in new:
                shutil.rmtree(build_dir(state, rec["key"]), ignore_errors=True)
            log(state, f"sbatch failed: {e}")
            return 1
        for rec in new:
            rec["slurm_job_ids"].append(job_id)
            save_record(state, rec)
            log(state, f"submitted {rec['subgroup']} {rec['key']} job {job_id} "
                       f"changed {','.join(rec['changed'])}")

    if not dry_run:
        write_json(state / "last-poll.json", ok)
        prune(state, site)
    return 0


# -- releases ------------------------------------------------------------

def version_key(tag):
    return tuple(int(x) for x in RELEASE_TAG.match(tag).groups())


def release_dir(state, subgroup, tag):
    return state / "releases" / subgroup / tag


def release_ref(rec):
    return f"{rec['subgroup']}/{rec['tag']}"


def save_release(state, rec):
    write_json(release_dir(state, rec["subgroup"], rec["tag"]) / "record.json", rec)


def load_releases(state):
    """All release records, newest version first within each subgroup."""
    out = []
    for path in (state / "releases").glob("*/*/record.json"):
        rec = load_json(path)
        if rec is not None:
            out.append(rec)
    out.sort(key=lambda r: (r["subgroup"], version_key(r["tag"])), reverse=True)
    return out


def release_link(site, subgroup):
    """<IMAGES_DIR>/<name>-latest.sif, <name> being the image's last path part."""
    name = RELEASES[subgroup]["image"].rsplit("/", 1)[-1]
    return Path(site["IMAGES_DIR"]) / f"{name}-latest.sif"


def release_files(site, subgroup, tag):
    """(<IMAGES_DIR>/<name>/<name>-<tag>.sif, release_link)."""
    link = release_link(site, subgroup)
    name = link.name.removesuffix("-latest.sif")
    return link.parent / name / f"{name}-{tag}.sif", link


def _http(url, headers=None, method="GET"):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    return urllib.request.urlopen(req, timeout=REGISTRY_TIMEOUT)


def _registry_auth(host, name):
    """Headers for anonymous pull access, from the registry's Bearer challenge."""
    try:
        with _http(f"https://{host}/v2/"):
            return {}
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        challenge = e.headers.get("WWW-Authenticate", "")
    params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    query = urllib.parse.urlencode({"service": params.get("service", ""),
                                    "scope": f"repository:{name}:pull"})
    with _http(f"{params['realm']}?{query}") as r:
        data = json.load(r)
    return {"Authorization": f"Bearer {data.get('token') or data['access_token']}"}


def registry_releases(image, keep):
    """The newest `keep` release tags of a public image, newest first, as
    [(tag, digest)]. Raises PollError.
    """
    host, name = image.split("/", 1)
    try:
        auth = _registry_auth(host, name)
        with _http(f"https://{host}/v2/{name}/tags/list?n=10000", auth) as r:
            tags = json.load(r).get("tags") or []
        out = []
        for tag in sorted((t for t in tags if RELEASE_TAG.match(t)), key=version_key,
                          reverse=True)[:keep]:
            with _http(f"https://{host}/v2/{name}/manifests/{tag}",
                       {**auth, "Accept": MANIFEST_ACCEPT}, "HEAD") as r:
                out.append((tag, r.headers["Docker-Content-Digest"]))
    except (OSError, ValueError, KeyError) as e:  # URLError and HTTPError are OSErrors
        raise PollError(f"{image}: {e}") from None
    return out


def submit_convert(state, site, records):
    resources = ["--partition", site["CONVERT_PARTITION"], "--qos", site["SLURM_QOS"],
                 "--cpus-per-task", site["CONVERT_CPUS"], "--mem", site["CONVERT_MEM"],
                 "--time", site["CONVERT_TIME"]]
    if site["CONVERT_GPUS"] != "0":
        resources += ["--gpus", site["CONVERT_GPUS"]]
    return sbatch(state, site, "t1-convert", resources,
                  ["convert", *(release_ref(r) for r in records)])


def _poll_releases(state, site, dry_run):
    subgroups = release_subgroups(site)
    if not subgroups:
        return 0
    records = load_releases(state)
    reconcile(state, site, records, dry_run, save_release, release_ref)
    active = [r for r in records if r["status"] in ACTIVE]
    known = {release_ref(r): r for r in records}
    check = {"time": now_iso(), "ok": True, "error": None}
    plan = []
    for sub in subgroups:
        image = RELEASES[sub]["image"]
        try:
            tags = registry_releases(image, int(site["RELEASE_KEEP"]))
        except PollError as e:
            check.update(ok=False, error=str(e))
            log(state, f"release check failed: {e}", dry_run=dry_run)
            continue
        for tag, digest in tags:
            rec = known.get(f"{sub}/{tag}")
            if rec is None:
                plan.append((sub, image, tag, digest))
            elif rec["digest"] != digest:
                log(state, f"release {sub}/{tag}: {image}:{tag} moved from {rec['digest']} "
                           f"to {digest}; not converted again", dry_run=dry_run)
    if dry_run:
        for sub, image, tag, digest in plan:
            log(state, f"would convert {sub}/{tag} {image}@{digest}", dry_run=True)
        return 0 if check["ok"] else 1
    if plan and active:
        log(state, f"conversion active: {' '.join(release_ref(r) for r in active)}")
    elif plan:
        new = []
        for sub, image, tag, digest in plan:
            rec = {"subgroup": sub, "image": image, "tag": tag, "digest": digest,
                   "status": "queued", "failed_stage": None, "attempts": 1, "slurm_job_ids": [],
                   "queued_at": now_iso(), "started_at": None, "finished_at": None,
                   "sif": None, "size": None, "sha256": None}
            release_dir(state, sub, tag).mkdir(parents=True, exist_ok=True)
            save_release(state, rec)
            new.append(rec)
        try:
            job_id = submit_convert(state, site, new)
        except SubmitError as e:
            for rec in new:
                shutil.rmtree(release_dir(state, rec["subgroup"], rec["tag"]), ignore_errors=True)
            log(state, f"sbatch failed: {e}")
            return 1
        for rec in new:
            rec["slurm_job_ids"].append(job_id)
            save_release(state, rec)
        log(state, f"submitted conversion {' '.join(release_ref(r) for r in new)} job {job_id}")
    write_json(state / "last-release-check.json", check)
    prune_releases(state, site)
    return 0 if check["ok"] else 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def has_gpu():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class Tee:
    """Writes to stdout (the Slurm job log) and to one record's own log."""

    def __init__(self, path):
        self.f = open(path, "a")

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
        self.f.write(text)
        self.f.flush()

    def close(self):
        self.f.close()


def _stream(cmd, out, step, **kwargs):
    """Run cmd, copying its output to out; StepError(step) on failure."""
    out.write(f"== {step} ==\n$ {' '.join(map(str, cmd))}\n")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                **kwargs)
    except OSError as e:
        out.write(f"convert: {e}\n")
        raise StepError(step) from None
    for line in proc.stdout:
        out.write(line)
    if proc.wait() != 0:
        raise StepError(step)


def convert(state, refs):
    """Convert release records (SUBGROUP/TAG) inside their Slurm job, then
    re-render the dashboard. A failed release never stops the next one.
    """
    site = load_site(state)
    rc = 0
    for ref in refs:
        sub, _, tag = ref.partition("/")
        rc = _convert(state, site, sub, tag) or rc
    refresh(state, site)
    return rc


def _convert(state, site, sub, tag):
    d = release_dir(state, sub, tag)
    with ledger_lock(state):
        rec = load_json(d / "record.json")
        if rec is None:
            print(f"no release record {sub}/{tag}", file=sys.stderr)
            return 1
        rec.update(status="running", started_at=now_iso())
        save_release(state, rec)
    sif, _ = release_files(site, sub, tag)
    scratch = scratch_base(site) / "t1-convert" / f"{sub}-{tag}"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "tmp").mkdir(parents=True)
    env = dict(os.environ, APPTAINER_CACHEDIR=str(scratch / "cache"),
               APPTAINER_TMPDIR=str(scratch / "tmp"))
    tmp_sif = scratch / sif.name
    out = Tee(d / "convert.log")
    status, stage, raw, digest = "failed", None, None, None
    try:
        _stream([site["APPTAINER"], "pull", "--force", str(tmp_sif),
                 f"docker://{rec['image']}@{rec['digest']}"], out, "apptainer pull", env=env)
        out.write("== inspect ==\n")
        try:
            raw = inspect_image(site, tmp_sif)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
            out.write(f"convert: inspection failed: {e}\n")
            raise StepError("inspect") from None
        test = RELEASES[sub].get("test")
        if test:
            nv = ["--nv"] if has_gpu() else []
            with open(REPO_ROOT / test) as script:
                _stream([site["APPTAINER"], "exec", *nv, str(tmp_sif), "bash", "-s"], out,
                        f"release test {test}", stdin=script, env=env)
        out.write("== sha256 ==\n")
        digest = sha256_file(tmp_sif)
        out.write(f"{digest}  {sif.name}\n")
        out.write(f"== publish {sif} ==\n")
        try:
            sif.parent.mkdir(parents=True, exist_ok=True)
            part = sif.with_name(f".{sif.name}.part")
            shutil.move(tmp_sif, part)
            os.replace(part, sif)
            (sif.parent / f"{sif.name}.sha256").write_text(f"{digest}  {sif.name}\n")
        except OSError as e:
            out.write(f"convert: {e}\n")
            raise StepError("publish") from None
        status = "success"
    except StepError as e:
        stage = str(e)
        out.write(f"convert: FAILED at {stage}\n")
    finally:
        out.close()
        shutil.rmtree(scratch, ignore_errors=True)
    if status == "success":
        info = compose_image_info(raw, {}, sif.stat().st_size, None, sif)
        write_json(d / "image-info.json", info)
    with ledger_lock(state):
        rec = load_json(d / "record.json")
        rec.update(status=status, failed_stage=stage, finished_at=now_iso())
        if status == "success":
            rec.update(sif=str(sif), size=sif.stat().st_size, sha256=digest)
        save_release(state, rec)
        update_release_link(state, site, sub)
    return 0 if status == "success" else 1


def update_release_link(state, site, sub):
    """Point <name>-latest.sif at the newest release version with an image."""
    for rec in load_releases(state):
        if rec["subgroup"] == sub and rec["status"] == "success":
            sif, link = release_files(site, sub, rec["tag"])
            if sif.is_file():
                return replace_link(link, sif)
    return False


def prune_releases(state, site):
    """Delete the .sif files of converted releases past the newest RELEASE_KEEP
    versions of each subgroup. The file the -latest link points at stays.
    """
    keep = int(site["RELEASE_KEEP"])
    done = {}
    for rec in load_releases(state):
        if rec["status"] == "success":
            done.setdefault(rec["subgroup"], []).append(rec)
    for sub, recs in done.items():
        for rec in recs[keep:]:
            sif, link = release_files(site, sub, rec["tag"])
            if os.path.realpath(link) == os.path.realpath(sif):
                continue
            for path in (sif, sif.with_name(f"{sif.name}.sha256")):
                if path.exists():
                    os.remove(path)
                    log(state, f"pruned {path}")


def retry_release(state, ref):
    site = load_site(state)
    sub, _, tag = ref.partition("/")
    d = release_dir(state, sub, tag)
    with ledger_lock(state):
        rec = load_json(d / "record.json")
        if rec is None:
            print(f"no release record {ref}", file=sys.stderr)
            return 1
        if rec["status"] not in RETRYABLE:
            print(f"{ref} is {rec['status']}; retry needs failed, interrupted or cancelled",
                  file=sys.stderr)
            return 1
        others = [r for r in load_releases(state) if r["status"] in ACTIVE]
        if others:
            print(f"conversion active: {release_ref(others[0])}", file=sys.stderr)
            return 1
        undo = rotate_log(d, "convert.log", rec["attempts"])
        try:
            job_id = submit_convert(state, site, [rec])
        except SubmitError as e:
            if undo:
                undo()
            print(f"sbatch failed: {e}", file=sys.stderr)
            return 1
        rec.update(attempts=rec["attempts"] + 1, status="queued", queued_at=now_iso(),
                   started_at=None, finished_at=None, failed_stage=None)
        rec["slurm_job_ids"].append(job_id)
        save_release(state, rec)
    refresh(state, site)
    print(f"retry {ref} job {job_id} attempt {rec['attempts']}")
    return 0


# -- commands --------------------------------------------------------------

def refresh(state, site, out=None):
    """Re-render the dashboard. Takes the ledger lock itself, so callers
    must call it only after they released theirs (flock is not reentrant).
    A render error never changes the caller's result; it is reported.
    """
    try:
        with ledger_lock(state):
            records = load_records(state)
            reviews = {r["key"]: load_json(build_dir(state, r["key"]) / "review.json")
                       for r in records}
            releases = load_releases(state)
            dashboard.render(
                Path(out or site["DASHBOARD_DIR"]), site=site, records=records,
                reviews=reviews,
                review_states={r["key"]: review_state(r, reviews[r["key"]]) for r in records},
                last_poll=load_json(state / "last-poll.json"),
                build_dirs={r["key"]: build_dir(state, r["key"]) for r in records},
                image_infos={r["key"]: load_json(build_dir(state, r["key"]) / "image-info.json")
                             for r in records},
                nightly_links={s: nightly_link(site, site["VARIANT"], s) for s in SUBGROUPS},
                subgroups=SUBGROUPS,
                releases={s: {"config": RELEASES[s],
                              "link": release_link(site, s),
                              "records": [r for r in releases if r["subgroup"] == s],
                              "infos": {r["tag"]: load_json(release_dir(state, s, r["tag"])
                                                            / "image-info.json")
                                        for r in releases if r["subgroup"] == s}}
                          for s in release_subgroups(site)},
                release_check=load_json(state / "last-release-check.json"),
                now=datetime.now(timezone.utc))
    except Exception as e:  # the dashboard must never break a poll or a build
        print(f"render failed: {e}", file=sys.stderr, flush=True)
        return False
    return True


def poll(state, dry_run):
    site = load_site(state)
    if dry_run:
        return max(_poll(state, site, True), _poll_releases(state, site, True))
    with ledger_lock(state):
        rc = _poll(state, site, False)
        rc = max(rc, _poll_releases(state, site, False))
    refresh(state, site)
    return rc


def review(state, key, note):
    site = load_site(state)
    with ledger_lock(state):
        rec = load_json(record_path(state, key))
        if rec is None:
            print(f"no record {key}", file=sys.stderr)
            return 1
        write_json(build_dir(state, key) / "review.json", {
            "time": now_iso(), "user": getpass.getuser(), "note": note,
            "attempt": rec["attempts"],
        })
    refresh(state, site)
    print(f"reviewed {key}")
    return 0


def retry(state, key):
    site = load_site(state)
    with ledger_lock(state):
        rec = next((r for r in load_records(state) if r["key"] == key), None)
        if rec is None:
            print(f"no record {key}", file=sys.stderr)
            return 1
        if rec["status"] not in RETRYABLE:
            print(f"{key} is {rec['status']}; retry needs failed, interrupted or cancelled",
                  file=sys.stderr)
            return 1
        others = [r for r in load_records(state) if r["status"] in ACTIVE]
        if others:
            print(f"build active: {others[0]['key']}", file=sys.stderr)
            return 1
        try:
            job_id = resubmit(state, site, [rec])
        except SubmitError as e:
            print(f"sbatch failed: {e}", file=sys.stderr)
            return 1
    refresh(state, site)
    print(f"retry {key} job {job_id} attempt {rec['attempts']}")
    return 0


def status(state):
    for rec in load_records(state):
        rv = load_json(build_dir(state, rec["key"]) / "review.json")
        line = (f"{rec['key']}  {rec['subgroup']:<6}  {rec['status']:<11}  a{rec['attempts']}  "
                f"{rec['tag']}  t1={rec['t1_software_commit'][:8]}  "
                f"changed={len(rec['changed'])}  {review_state(rec, rv)}")
        print(line.rstrip())
    for rec in load_releases(state):
        print(f"{release_ref(rec)}  {rec['status']:<11}  a{rec['attempts']}  {rec['digest'][:19]}  "
              f"{rec.get('failed_stage') or ''}".rstrip())
    return 0


# The record and site.env alone decide a nightly build, not the
# environment of whoever ran poll or retry (sbatch --export ALL).
BUILD_ENV_DROP = ("T1_SOFTWARE_IMAGES_DIR", "T1_SOFTWARE_CACHE_DIR", "T1_SKIP_PIPELINE_TEST")
STAGE_LINE = re.compile(r"== (.*) ==")


def scratch_base(site):
    for value in (site["WORK_DIR"], os.environ.get("LSCRATCH"), os.environ.get("TMPDIR")):
        if value:
            return Path(value)
    return Path("/tmp")


def finish(state, key, status, stage):
    with ledger_lock(state):
        rec = load_json(record_path(state, key))
        rec.update(status=status, failed_stage=stage, finished_at=now_iso())
        save_record(state, rec)


def run(state, keys):
    """Build records one after another inside their Slurm job, then
    re-render the dashboard. A failed build never stops the next one; an
    interruption stops the job.
    """
    site = load_site(state)
    rc = 0
    for key in keys:
        one = _run(state, site, key)
        if one == 143:
            rc = one
            break
        rc = one or rc
    refresh(state, site)
    return rc


def _run(state, site, key):
    with ledger_lock(state):
        rec = load_json(record_path(state, key))
        if rec is None:
            print(f"no record {key}", file=sys.stderr)
            return 1
        rec.setdefault("subgroup", "lartpc")
        rec.update(status="running", started_at=now_iso())
        save_record(state, rec)

    out = Tee(build_dir(state, key) / "build.log")
    try:
        return _build(state, site, rec, out)
    finally:
        out.close()


def _build(state, site, rec, out):
    key = rec["key"]
    # A scratch clone at the recorded commit: the image matches the key
    # even if the branch moved since the poll, and no checkout is touched.
    src = scratch_base(site) / "t1-nightly" / f"{key}-src"
    shutil.rmtree(src, ignore_errors=True)
    src.parent.mkdir(parents=True, exist_ok=True)
    out.write(f"nightly: {rec['subgroup']}: fetching t1-software {rec['t1_software_commit']} "
              f"into {src}\n")
    try:
        git("init", "--quiet", str(src))
        git("-C", str(src), "fetch", "--quiet", "--depth", "1",
            site["T1_SOFTWARE_REPO"], rec["t1_software_commit"])
        git("-C", str(src), "checkout", "--quiet", "FETCH_HEAD")
    except PollError as e:
        out.write(f"nightly: checkout failed: {e}\n")
        finish(state, key, "failed", "checkout")
        shutil.rmtree(src, ignore_errors=True)
        return 1

    # The base layers change only with a t1-software commit, so a nightly
    # base archive is mostly a 13+ GB copy of the previous one. The base is
    # still built (every subgroup image builds FROM it) and tested.
    cmd = [str(src / "scripts" / "build.sh"), "--variant", rec["variant"],
           "--subgroup", rec["subgroup"],
           "--images-dir", rec["images_dir"], "--tag", rec["tag"], "--no-base-archive"]
    for k, v in sorted(rec["pins"].items()):
        cmd += ["--pin", f"{k}={v}"]
    if site["CACHE_DIR"]:
        cmd += ["--cache-dir", site["CACHE_DIR"]]
    if site["WORK_DIR"]:
        cmd += ["--work-dir", site["WORK_DIR"]]
    env = {k: v for k, v in os.environ.items() if k not in BUILD_ENV_DROP}

    # Slurm sends SIGTERM on scancel, preemption and time limit. Pass it to
    # build.sh and leave the record `running`: poll reads the real cause
    # from sacct.
    interrupted = False
    proc = None

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    stage = None
    lock_busy = False
    try:
        try:
            proc = subprocess.Popen(cmd, cwd=src, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
        except OSError as e:
            out.write(f"nightly: cannot start build.sh: {e}\n")
            finish(state, key, "failed", "start build.sh")
            shutil.rmtree(src, ignore_errors=True)
            return 1
        for line in proc.stdout:
            out.write(line)
            m = STAGE_LINE.fullmatch(line.rstrip("\n"))
            if m:
                stage = m.group(1)
            if "holds" in line and ".lock" in line:
                lock_busy = True
        rc = proc.wait()
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    shutil.rmtree(src, ignore_errors=True)
    if interrupted:
        out.write("nightly: interrupted; poll reconciles from sacct\n")
        return 143
    if rc == 0:
        finish(state, key, "success", None)
        update_nightly_link(site, rec)
        write_image_info(state, site, rec)
    else:
        finish(state, key, "failed", "cache lock" if lock_busy else stage)
    return rc


def prune(state, site):
    """Delete image files of successful nightlies past the newest NIGHTLY_KEEP
    of each subgroup.

    A file that a -latest or nightly link points at stays. Manifests,
    records and logs always stay.
    """
    keep = int(site["NIGHTLY_KEEP"])
    done = {}
    for rec in sorted((r for r in load_records(state) if r["status"] == "success"),
                      key=lambda r: r.get("finished_at") or "", reverse=True):
        done.setdefault(rec["subgroup"], []).append(rec)
    for recs in done.values():
        for rec in recs[keep:]:
            d = Path(rec["images_dir"])
            if not d.is_dir():
                continue
            linked = {os.path.realpath(p) for p in [*d.glob("*-latest.*"), *d.parent.glob("*-nightly.sif")]
                      if p.is_symlink()}
            for suffix in ("oci.tar", "sif"):
                for path in sorted(d.glob(f"t1-*-{rec['variant']}-{rec['tag']}.{suffix}")):
                    if path.is_symlink() or os.path.realpath(path) in linked:
                        continue
                    os.remove(path)
                    log(state, f"pruned {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nightly.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--state", help="state directory (default: $T1_NIGHTLY_STATE)")
        return p

    p = add("poll", "resolve HEADs and release tags; submit builds and conversions")
    p.add_argument("--dry-run", action="store_true",
                   help="print the decisions; write nothing but the bare cache")
    p = add("review", "record a review of a build")
    p.add_argument("key")
    p.add_argument("--note", required=True)
    p = add("retry", "submit a new attempt of a failed, interrupted or cancelled build")
    p.add_argument("key")
    p = add("retry-release", "submit a new attempt of a failed, interrupted or cancelled conversion")
    p.add_argument("ref", metavar="SUBGROUP/TAG")
    add("status", "one line per build and per release, newest first")
    p = add("run", "build records, one after another (runs inside the Slurm job)")
    p.add_argument("keys", nargs="+", metavar="key")
    p = add("convert", "convert release records (runs inside the Slurm job)")
    p.add_argument("refs", nargs="+", metavar="SUBGROUP/TAG")
    p = add("render", "write the static dashboard")
    p.add_argument("--out", help="output directory (default: DASHBOARD_DIR from site.env)")

    args = parser.parse_args(argv)
    state_arg = args.state or os.environ.get("T1_NIGHTLY_STATE")
    if not state_arg:
        print("nightly: --state or $T1_NIGHTLY_STATE is required", file=sys.stderr)
        return 2
    state = Path(state_arg).resolve()
    try:
        if args.cmd == "poll":
            return poll(state, args.dry_run)
        if args.cmd == "review":
            return review(state, args.key, args.note)
        if args.cmd == "retry":
            return retry(state, args.key)
        if args.cmd == "retry-release":
            return retry_release(state, args.ref)
        if args.cmd == "run":
            return run(state, args.keys)
        if args.cmd == "convert":
            return convert(state, args.refs)
        if args.cmd == "render":
            return 0 if refresh(state, load_site(state), args.out) else 1
        if args.cmd == "status":
            return status(state)
    except UsageError as e:
        print(e, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
