#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Nightly HEAD builds of t1-software on S3DF.

Two layers do the work. scripts/build.sh is the generic layer: it builds
one commit set and knows nothing about the nightly. This script is the
site layer. Once a day, cron runs `poll` on the cron node. `poll` resolves
the HEAD of t1-software and of every package that has a <PKG>_REPO key in
versions.env. The commit set gives a build key. A key that is new to the
ledger gets one Slurm job, which runs `run` on a GPU node. `run` fetches
t1-software at the recorded commit into a scratch clone and calls
build.sh with --pin and --tag. No t1-software checkout is ever modified.

Ledger, under the state directory (--state or $T1_NIGHTLY_STATE):

    site.env                      site values (paths, Slurm account, ...)
    last-poll.json                {time, ok, error, heads}
    poll.log                      one line per poll decision
    t1-software.git/              bare cache, used to read versions.env
    builds/<key>/record.json      one record per commit set
    builds/<key>/build.log        latest attempt; attempt-N.log for earlier
    builds/<key>/review.json      written only by `review`

Statuses:

    queued       poll/retry, after sbatch succeeds
    running      run, at job start
    success      run, build.sh exit 0
    failed       run, build.sh exit != 0; or poll, from sacct
    interrupted  poll, on PREEMPTED, TIMEOUT or NODE_FAIL; poll resubmits
                 it once, then it waits for `retry`
    cancelled    poll, on CANCELLED; waits for `retry`

A key that has a record is never built again automatically, whatever its
status. Only `retry` rebuilds it. At most one build is queued or running.
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
from datetime import datetime, timezone
from pathlib import Path

import dashboard  # sites/s3df/dashboard.py; a script's own directory is on sys.path

T1 = "T1_SOFTWARE_COMMIT"
ACTIVE = ("queued", "running")
RETRYABLE = ("failed", "interrupted", "cancelled")
SLURM_ALIVE = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING",
               "COMPLETING", "RESIZING"}
SLURM_INTERRUPT = {"PREEMPTED", "TIMEOUT", "NODE_FAIL"}
GIT_TIMEOUT = 120  # s; one ls-remote or shallow fetch takes a few seconds

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
    "SBATCH": "sbatch",
    "SACCT": "sacct",
    "APPTAINER": "apptainer",
}
REQUIRED = ("IMAGES_DIR", "SLURM_ACCOUNT")

KV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class UsageError(Exception):
    """Bad configuration or arguments; exit code 2."""


class PollError(Exception):
    """A HEAD lookup failed; the poll submits nothing."""


class SubmitError(Exception):
    """sbatch failed."""


class SacctError(Exception):
    """sacct failed."""


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
    if not site["DASHBOARD_DIR"]:
        site["DASHBOARD_DIR"] = str(state / "site")
    # uv sets $UV to its own path for the scripts it runs. Cron's PATH
    # lacks ~/.local/bin, so the Slurm job needs this absolute path.
    uv = site["UV"] or os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise UsageError("site.env: UV not found")
    site["UV"] = os.path.abspath(uv)
    return site


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


def build_key(variant, t1_commit, pins):
    blob = json.dumps({"variant": variant, "t1_software_commit": t1_commit, "pins": pins},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def build_dir(state, key):
    return state / "builds" / key


def record_path(state, key):
    return build_dir(state, key) / "record.json"


def _order(record):
    """Sort key: queued_at, then the last Slurm job id.

    queued_at has one-second resolution, so two records can tie. Slurm job
    ids grow monotonically, which breaks the tie.
    """
    ids = record.get("slurm_job_ids") or []
    last = int(ids[-1]) if ids and str(ids[-1]).isdigit() else -1
    return (record.get("queued_at") or "", last)


def load_records(state):
    """All records, newest first (see _order)."""
    records = []
    for path in (state / "builds").glob("*/record.json"):
        rec = load_json(path)
        if rec is not None:
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
    """HEADs and clone URLs of t1-software and of every package with a
    <PKG>_REPO key. Returns (heads, repos), both keyed by <PKG>_COMMIT.
    """
    repo, branch = site["T1_SOFTWARE_REPO"], site["T1_SOFTWARE_BRANCH"]
    commit = ls_remote(repo, f"refs/heads/{branch}")
    cache = state / "t1-software.git"
    if not cache.exists():
        git("init", "--bare", "--quiet", str(cache))
    git("-C", str(cache), "fetch", "--quiet", "--depth", "1", repo, commit)
    paths = ["base/versions.env"]
    for line in git("-C", str(cache), "ls-tree", "--name-only", commit, "subgroups/").splitlines():
        paths.append(f"{line}/versions.env")
    heads = {T1: commit}
    repos = {T1: repo}
    for path in paths:
        try:
            text = git("-C", str(cache), "show", f"{commit}:{path}")
        except PollError:
            if path == "base/versions.env":
                raise
            continue  # a subgroup without its own pins
        pins = parse_kv(text)
        for key in sorted(pins):
            if key.endswith("_COMMIT") and pins.get(key[: -len("_COMMIT")] + "_REPO"):
                url = pins[key[: -len("_COMMIT")] + "_REPO"]
                heads[key] = ls_remote(url, "HEAD")
                repos[key] = url
    return heads, repos


def nightly_link(site, variant):
    """Path of the symlink to the newest nightly .sif of a variant, next to nightly/."""
    return Path(site["IMAGES_DIR"]) / f"t1-lartpc-{variant}-nightly.sif"


def update_nightly_link(site, rec):
    """Point nightly_link at rec's .sif with a relative target. Returns
    False when the image file is missing or the link cannot be written.
    """
    sif = Path(rec["images_dir"]) / f"t1-lartpc-{rec['variant']}-{rec['tag']}.sif"
    link = nightly_link(site, rec["variant"])
    if not sif.is_file():
        return False
    tmp = link.with_name(f".{link.name}.tmp.{os.getpid()}")
    try:
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(os.path.relpath(sif, link.parent))
        os.replace(tmp, link)
    except OSError as e:
        print(f"nightly: cannot update {link}: {e}", file=sys.stderr, flush=True)
        tmp.unlink(missing_ok=True)
        return False
    return True


def _pip_version(pip, prefix):
    """Version of the pip package named prefix, else the first one that starts with it."""
    if prefix in pip:
        return pip[prefix]
    return next((v for n, v in sorted(pip.items()) if n.startswith(prefix)), "")


def _differs(pinned, installed):
    return bool(pinned) and not (installed == pinned or installed.startswith(pinned + "."))


def compose_image_info(raw, base_pins, size, built_at, sif):
    """Core-stack rows (pinned vs installed) and the package list of one image.

    raw is the output of image_info.py; base_pins is the base manifest's
    pins. A pin of "3.11" matches an installed "3.11.17".
    """
    pip = dict(raw["pip"])
    conda = raw["conda"]
    jax = _pip_version(pip, "jax")
    spec = [
        ("OS", "", raw["os"], raw["os"], ""),
        ("Python", "python_version", raw["python"], raw["python"], "conda-forge"),
        ("PyTorch", "torch_version", _pip_version(pip, "torch"), _pip_version(pip, "torch"), ""),
        ("JAX", "jax_version", f"{jax} (jaxlib {_pip_version(pip, 'jaxlib')})", jax, ""),
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
    return {"core": core, "pip": raw["pip"], "conda_count": len(conda), "size": size,
            "built_at": built_at, "sif": str(sif)}


def write_image_info(state, site, rec):
    """Inspect the built .sif and write builds/<key>/image-info.json. A failure
    is reported and never changes the build result. Returns success.
    """
    images = Path(rec["images_dir"])
    sif = images / f"t1-lartpc-{rec['variant']}-{rec['tag']}.sif"
    if not sif.is_file():
        return False
    try:
        script = Path(__file__).with_name("image_info.py").read_text()
        r = subprocess.run([site["APPTAINER"], "exec", str(sif), "python", "-"], input=script,
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-300:])
        raw = json.loads(r.stdout)
        base = load_json(images / f"t1-base-{rec['variant']}-{rec['tag']}.manifest.json") or {}
        lartpc = load_json(images / f"t1-lartpc-{rec['variant']}-{rec['tag']}.manifest.json") or {}
        info = compose_image_info(raw, base.get("pins") or {}, sif.stat().st_size,
                                  lartpc.get("built_at"), sif)
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


def submit(state, site, record):
    """sbatch one build job; returns its job id."""
    state = state.resolve()
    here = Path(__file__).resolve()
    key = record["key"]
    cmd = [
        site["SBATCH"], "--parsable",
        "--job-name", f"t1-nightly-{key}",
        "--account", site["SLURM_ACCOUNT"],
        "--partition", site["SLURM_PARTITION"],
        "--qos", site["SLURM_QOS"],
        "--gpus", site["SLURM_GPUS"],
        "--cpus-per-task", site["SLURM_CPUS"],
        "--mem", site["SLURM_MEM"],
        "--time", site["SLURM_TIME"],
        "--output", str(build_dir(state, key) / "build.log"),
        "--open-mode", "append",
        # The poll owns retries; Slurm's own requeue would race it.
        "--no-requeue",
        "--export", "ALL",
        str(here.parent / "build.sbatch"),
        site["UV"], str(here), "run", key, "--state", str(state),
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SubmitError(str(e)) from None
    if r.returncode != 0:
        raise SubmitError(r.stderr.strip() or f"exit {r.returncode}")
    return r.stdout.strip().split(";", 1)[0].strip()


def resubmit(state, site, record):
    """Submit a new attempt of an existing record. Returns the job id.

    On sbatch failure the record and its logs stay as they were.
    """
    d = build_dir(state, record["key"])
    log_path = d / "build.log"
    old = d / f"attempt-{record['attempts']}.log"
    moved = log_path.exists()
    if moved:
        os.replace(log_path, old)
    try:
        job_id = submit(state, site, record)
    except SubmitError:
        if moved:
            os.replace(old, log_path)
        raise
    record.update(attempts=record["attempts"] + 1, status="queued", queued_at=now_iso(),
                  started_at=None, finished_at=None, failed_stage=None)
    record["slurm_job_ids"].append(job_id)
    save_record(state, record)
    return job_id


def new_record(site, key, heads, repos, changed):
    queued_at = now_iso()
    return {
        "key": key,
        "variant": site["VARIANT"],
        "t1_software_commit": heads[T1],
        "pins": {k: v for k, v in heads.items() if k != T1},
        "changed": changed,
        "repos": repos,
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


def reconcile(state, site, records, dry_run):
    """Settle queued/running records whose Slurm job ended without a result."""
    for rec in records:
        if rec["status"] not in ACTIVE or not rec["slurm_job_ids"]:
            continue
        try:
            st = sacct_state(site, rec["slurm_job_ids"][-1])
        except SacctError as e:
            log(state, f"sacct failed for {rec['key']}: {e}", dry_run=dry_run)
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
            log(state, f"would reconcile {rec['key']}: {st} -> {new}", dry_run=True)
            continue
        rec["status"] = new
        if new == "failed" and rec["failed_stage"] is None:
            rec["failed_stage"] = f"slurm {st}"
        rec["finished_at"] = now_iso()
        save_record(state, rec)
        log(state, f"reconcile {rec['key']}: {st} -> {new}")


def _poll(state, site, dry_run):
    records = load_records(state)
    prev = load_json(state / "last-poll.json") or {}
    prev_heads = prev.get("heads") or {}

    reconcile(state, site, records, dry_run)
    active = [r for r in records if r["status"] in ACTIVE]

    try:
        heads, repos = resolve_heads(state, site)
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
            rec = pending[0]
            if dry_run:
                log(state, f"would auto-retry {rec['key']}", dry_run=True)
            else:
                try:
                    job_id = resubmit(state, site, rec)
                except SubmitError as e:
                    log(state, f"sbatch failed: {e}")
                    return 1
                log(state, f"auto-retry {rec['key']} job {job_id}")
            active = [rec]

    ok = {"time": now_iso(), "ok": True, "error": None, "heads": heads}
    if active:
        if not dry_run:
            write_json(state / "last-poll.json", ok)
        log(state, f"build active: {active[0]['key']}", dry_run=dry_run)
        return 0

    pins = {k: v for k, v in heads.items() if k != T1}
    key = build_key(site["VARIANT"], heads[T1], pins)
    if dry_run:
        print(f"key {key}")
    existing = load_json(record_path(state, key))
    if existing is not None:
        if heads == prev_heads:
            log(state, "no change", dry_run=dry_run)
        else:
            log(state, f"already built {key} ({existing['status']})", dry_run=dry_run)
    else:
        changed = {n: [prev_heads.get(n), sha] for n, sha in heads.items()
                   if prev_heads.get(n) != sha}
        if dry_run:
            log(state, f"would submit {key} changed {','.join(changed)}", dry_run=True)
            return 0
        rec = new_record(site, key, heads, repos, changed)
        d = build_dir(state, key)
        d.mkdir(parents=True, exist_ok=True)
        save_record(state, rec)
        try:
            job_id = submit(state, site, rec)
        except SubmitError as e:
            shutil.rmtree(d, ignore_errors=True)
            log(state, f"sbatch failed: {e}")
            return 1
        rec["slurm_job_ids"].append(job_id)
        save_record(state, rec)
        log(state, f"submitted {key} job {job_id} changed {','.join(changed)}")

    if not dry_run:
        write_json(state / "last-poll.json", ok)
        prune(state, site)
    return 0


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
            dashboard.render(
                Path(out or site["DASHBOARD_DIR"]), site=site, records=records,
                reviews=reviews,
                review_states={r["key"]: review_state(r, reviews[r["key"]]) for r in records},
                last_poll=load_json(state / "last-poll.json"),
                build_dirs={r["key"]: build_dir(state, r["key"]) for r in records},
                image_infos={r["key"]: load_json(build_dir(state, r["key"]) / "image-info.json")
                             for r in records},
                nightly_link=nightly_link(site, site["VARIANT"]),
                now=datetime.now(timezone.utc))
    except Exception as e:  # the dashboard must never break a poll or a build
        print(f"render failed: {e}", file=sys.stderr, flush=True)
        return False
    return True


def poll(state, dry_run):
    site = load_site(state)
    if dry_run:
        return _poll(state, site, True)
    with ledger_lock(state):
        rc = _poll(state, site, False)
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
        rec = load_json(record_path(state, key))
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
            job_id = resubmit(state, site, rec)
        except SubmitError as e:
            print(f"sbatch failed: {e}", file=sys.stderr)
            return 1
    refresh(state, site)
    print(f"retry {key} job {job_id} attempt {rec['attempts']}")
    return 0


def status(state):
    for rec in load_records(state):
        rv = load_json(build_dir(state, rec["key"]) / "review.json")
        line = (f"{rec['key']}  {rec['status']:<11}  a{rec['attempts']}  {rec['tag']}  "
                f"t1={rec['t1_software_commit'][:8]}  changed={len(rec['changed'])}  "
                f"{review_state(rec, rv)}")
        print(line.rstrip())
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


def run(state, key):
    """Build one record inside its Slurm job, then re-render the dashboard."""
    site = load_site(state)
    rc = _run(state, site, key)
    refresh(state, site)
    return rc


def _run(state, site, key):
    with ledger_lock(state):
        rec = load_json(record_path(state, key))
        if rec is None:
            print(f"no record {key}", file=sys.stderr)
            return 1
        rec.update(status="running", started_at=now_iso())
        save_record(state, rec)

    # A scratch clone at the recorded commit: the image matches the key
    # even if the branch moved since the poll, and no checkout is touched.
    src = scratch_base(site) / "t1-nightly" / f"{key}-src"
    shutil.rmtree(src, ignore_errors=True)
    src.parent.mkdir(parents=True, exist_ok=True)
    print(f"nightly: fetching t1-software {rec['t1_software_commit']} into {src}", flush=True)
    try:
        git("init", "--quiet", str(src))
        git("-C", str(src), "fetch", "--quiet", "--depth", "1",
            site["T1_SOFTWARE_REPO"], rec["t1_software_commit"])
        git("-C", str(src), "checkout", "--quiet", "FETCH_HEAD")
    except PollError as e:
        print(f"nightly: checkout failed: {e}", file=sys.stderr, flush=True)
        finish(state, key, "failed", "checkout")
        shutil.rmtree(src, ignore_errors=True)
        return 1

    # The base layers change only with a t1-software commit, so a nightly
    # base archive is mostly a 13+ GB copy of the previous one. The base is
    # still built (the LArTPC image builds FROM it) and tested.
    cmd = [str(src / "scripts" / "build.sh"), "--variant", rec["variant"],
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
            print(f"nightly: cannot start build.sh: {e}", file=sys.stderr, flush=True)
            finish(state, key, "failed", "start build.sh")
            shutil.rmtree(src, ignore_errors=True)
            return 1
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
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
        print("nightly: interrupted; poll reconciles from sacct", flush=True)
        return 143
    if rc == 0:
        finish(state, key, "success", None)
        update_nightly_link(site, rec)
        write_image_info(state, site, rec)
    else:
        finish(state, key, "failed", "cache lock" if lock_busy else stage)
    return rc


def prune(state, site):
    """Delete image files of successful nightlies past the newest NIGHTLY_KEEP.

    A file that a -latest link points at stays. Manifests, records and
    logs always stay.
    """
    keep = int(site["NIGHTLY_KEEP"])
    done = sorted((r for r in load_records(state) if r["status"] == "success"),
                  key=lambda r: r.get("finished_at") or "", reverse=True)
    for rec in done[keep:]:
        d = Path(rec["images_dir"])
        if not d.is_dir():
            continue
        latest = {os.path.realpath(p) for p in d.glob("*-latest.*") if p.is_symlink()}
        for suffix in ("oci.tar", "sif"):
            for path in sorted(d.glob(f"t1-*-{rec['variant']}-{rec['tag']}.{suffix}")):
                if path.is_symlink() or os.path.realpath(path) in latest:
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

    p = add("poll", "resolve HEADs; submit a build for a new commit set")
    p.add_argument("--dry-run", action="store_true",
                   help="print the decisions; write nothing but the bare cache")
    p = add("review", "record a review of a build")
    p.add_argument("key")
    p.add_argument("--note", required=True)
    p = add("retry", "submit a new attempt of a failed, interrupted or cancelled build")
    p.add_argument("key")
    add("status", "one line per build, newest first")
    p = add("run", "build one record (runs inside the Slurm job)")
    p.add_argument("key")
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
        if args.cmd == "run":
            return run(state, args.key)
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
