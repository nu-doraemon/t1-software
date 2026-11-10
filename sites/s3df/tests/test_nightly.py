"""Tests of sites/s3df/nightly.py against local git remotes and fake Slurm.

Each test builds its own fixture in a temporary directory:
  - pkg.git and t1.git: bare remotes over file://, fetchable by SHA;
  - a fake sbatch that logs its argv and prints a job id (101, 102, ...);
  - a fake sacct that prints <tmp>/sacct/<jobid> when that file exists;
  - a site.env pointing at all of them.

    uv run --python 3.11 python -m unittest discover -s sites/s3df/tests -v
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import datetime, timezone
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_DIR))
_spec = importlib.util.spec_from_file_location("nightly", SITE_DIR / "nightly.py")
nightly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nightly)

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
}

# Environment variables the tests set; setUp clears them, tearDown restores them.
FAKE_VARS = ("T1_NIGHTLY_STATE", "FAKE_SBATCH_RC", "FAKE_BUILD_RC", "FAKE_ARGV", "FAKE_SEEN",
             "FAKE_EXTRA_LINE", "FAKE_ENV_PROBE", "T1_SKIP_PIPELINE_TEST", "FAKE_APPTAINER_RC")

FAKE_SBATCH = textwrap.dedent("""\
    #!/bin/bash
    root="$(dirname "$(dirname "$0")")"
    if [[ "${FAKE_SBATCH_RC:-0}" != 0 ]]; then
        echo "fake sbatch: refused" >&2
        exit "$FAKE_SBATCH_RC"
    fi
    python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@" >> "$root/sbatch.calls"
    args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
        [[ "${args[i]}" == --output ]] && touch "${args[i+1]}"
    done
    n=$(wc -l < "$root/sbatch.calls")
    echo "$((100 + n));cluster"
    """)

FAKE_SACCT = textwrap.dedent("""\
    #!/bin/bash
    root="$(dirname "$(dirname "$0")")"
    args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
        if [[ "${args[i]}" == -j ]]; then
            f="$root/sacct/${args[i+1]}"
            [[ -f "$f" ]] && cat "$f"
        fi
    done
    exit 0
    """)

# What image_info.py prints for the fake image.
RAW_IMAGE = json.dumps({
    "os": "Test OS 1", "python": "3.11.17", "cmake": "3.31.8",
    "pip": [["h5py", "3.16.0"], ["jax", "0.6.2"], ["jaxlib", "0.6.2"], ["numpy", "2.4.6"],
            ["torch", "2.13.0+cu129"], ["triton", "3.1.0"]],
    "conda": {"root_base": "6.32.16", "geant4": "10.7.4", "hdf5": "1.14.6",
              "gxx_linux-64": "14.3.0"}})

# Stands in for `apptainer exec IMAGE python -`: reads the script, prints RAW_IMAGE.
FAKE_APPTAINER = textwrap.dedent("""\
    #!/bin/bash
    root="$(dirname "$(dirname "$0")")"
    cat > /dev/null
    if [[ "${FAKE_APPTAINER_RC:-0}" != 0 ]]; then
        echo "fake apptainer: refused" >&2
        exit "$FAKE_APPTAINER_RC"
    fi
    cat "$root/image_raw.json"
    """)

# Stands in for scripts/build.sh inside the fetched t1-software commit.
FAKE_BUILD = textwrap.dedent("""\
    #!/bin/bash
    [[ -n "${FAKE_ARGV:-}" ]] && printf '%s\\n' "$@" > "$FAKE_ARGV"
    [[ -n "${FAKE_SEEN:-}" ]] && cat marker.txt > "$FAKE_SEEN"
    [[ -n "${FAKE_ENV_PROBE:-}" ]] && env > "$FAKE_ENV_PROBE"
    echo "== stage-one =="
    [[ -n "${FAKE_EXTRA_LINE:-}" ]] && echo "$FAKE_EXTRA_LINE"
    echo "== stage-two =="
    exit "${FAKE_BUILD_RC:-0}"
    """)


def utc_today():
    return datetime.now(timezone.utc).strftime("%Y%m%d")


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.state = self.root / "state"
        self.state.mkdir()
        bindir = self.root / "bin"
        bindir.mkdir()
        (self.root / "sacct").mkdir()
        for name, body in [("sbatch", FAKE_SBATCH), ("sacct", FAKE_SACCT),
                           ("apptainer", FAKE_APPTAINER)]:
            f = bindir / name
            f.write_text(body)
            f.chmod(0o755)
        (self.root / "image_raw.json").write_text(RAW_IMAGE)

        self.pkg_src = self.root / "pkgsrc"
        self.pkg_git = self.root / "pkg.git"
        self.bare(self.pkg_git)
        self.git("init", "--quiet", "--initial-branch=main", str(self.pkg_src))
        self.A = self.commit_pkg("a")

        self.t1_src = self.root / "t1src"
        self.t1_git = self.root / "t1.git"
        self.bare(self.t1_git)
        self.git("init", "--quiet", "--initial-branch=main", str(self.t1_src))
        (self.t1_src / "base").mkdir()
        (self.t1_src / "base/versions.env").write_text("CUDA_VARIANT=cuda12\n")
        (self.t1_src / "subgroups/lartpc").mkdir(parents=True)
        self.write_pins(f"file://{self.pkg_git}", self.A)
        (self.t1_src / "scripts").mkdir()
        (self.t1_src / "scripts/build.sh").write_text(FAKE_BUILD)
        (self.t1_src / "marker.txt").write_text("committed\n")
        self.git("-C", str(self.t1_src), "add", "-A")
        self.git("-C", str(self.t1_src), "update-index", "--chmod=+x", "scripts/build.sh")
        self.t1_commit = self.commit_t1("t1 initial", add=False)

        self.site = {
            "T1_SOFTWARE_REPO": f"file://{self.t1_git}",
            "IMAGES_DIR": str(self.root / "img"),
            "SLURM_ACCOUNT": "test",
            "SBATCH": str(bindir / "sbatch"),
            "SACCT": str(bindir / "sacct"),
            "APPTAINER": str(bindir / "apptainer"),
            "WORK_DIR": str(self.root / "work"),
            "UV": "/bin/true",
        }
        self.write_site()

    def git(self, *args):
        return subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True).stdout.strip()

    def bare(self, path):
        self.git("init", "--quiet", "--bare", "--initial-branch=main", str(path))
        self.git("-C", str(path), "config", "uploadpack.allowAnySHA1InWant", "true")

    def commit_pkg(self, content):
        (self.pkg_src / "f.txt").write_text(content)
        self.git("-C", str(self.pkg_src), "add", "-A")
        self.git("-C", str(self.pkg_src), "commit", "--quiet", "-m", content)
        self.git("-C", str(self.pkg_src), "push", "--quiet", str(self.pkg_git), "HEAD:main")
        return self.git("-C", str(self.pkg_src), "rev-parse", "HEAD")

    def write_pins(self, repo, commit):
        (self.t1_src / "subgroups/lartpc/versions.env").write_text(
            f"# pins\nPKG_REPO={repo}\nPKG_COMMIT={commit}\n")

    def commit_t1(self, msg, add=True):
        if add:
            self.git("-C", str(self.t1_src), "add", "-A")
        self.git("-C", str(self.t1_src), "commit", "--quiet", "-m", msg)
        self.git("-C", str(self.t1_src), "push", "--quiet", str(self.t1_git), "HEAD:main")
        return self.git("-C", str(self.t1_src), "rev-parse", "HEAD")

    def write_site(self):
        (self.state / "site.env").write_text(
            "".join(f"{k}={v}\n" for k, v in self.site.items()))

    def main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = nightly.main([*args, "--state", str(self.state)])
        return rc, out.getvalue(), err.getvalue()

    def poll(self):
        return self.main("poll")[0]

    def calls(self):
        path = self.root / "sbatch.calls"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def records(self):
        return nightly.load_records(self.state)

    def only_record(self):
        recs = self.records()
        assert len(recs) == 1, recs
        return recs[0]

    def set_status(self, key, status, **extra):
        path = nightly.record_path(self.state, key)
        rec = json.loads(path.read_text())
        rec["status"] = status
        rec.update(extra)
        path.write_text(json.dumps(rec))

    def set_sacct(self, job_id, state):
        (self.root / "sacct" / str(job_id)).write_text(f"{state}\n")

    def poll_log_last(self):
        return (self.state / "poll.log").read_text().splitlines()[-1]


class NightlyTest(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in [*GIT_ENV, *FAKE_VARS]}
        os.environ.update(GIT_ENV)
        for k in FAKE_VARS:
            os.environ.pop(k, None)
        self._tmps = []

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for t in self._tmps:
            t.cleanup()

    def fixture(self):
        t = tempfile.TemporaryDirectory()
        self._tmps.append(t)
        return Fixture(t.name)

    def test_key(self):
        expected = hashlib.sha256(
            b'{"pins":{"A_COMMIT":"1","B_COMMIT":"2"},"t1_software_commit":"'
            + b"c" * 40 + b'","variant":"cuda12"}').hexdigest()[:12]
        got = nightly.build_key("cuda12", "c" * 40, {"B_COMMIT": "2", "A_COMMIT": "1"})
        self.assertEqual(got, expected)
        self.assertEqual(len(got), 12)

    def test_first_poll_and_no_change(self):
        f = self.fixture()
        self.assertEqual(f.poll(), 0)
        self.assertEqual(len(f.calls()), 1)
        rec = f.only_record()
        self.assertEqual(set(rec), {
            "key", "variant", "t1_software_commit", "pins", "changed", "status",
            "failed_stage", "attempts", "slurm_job_ids", "queued_at", "started_at",
            "finished_at", "tag", "images_dir", "repos"})
        self.assertEqual(rec["status"], "queued")
        self.assertEqual(rec["attempts"], 1)
        self.assertEqual(rec["slurm_job_ids"], ["101"])
        self.assertEqual(rec["images_dir"], str(f.root / "img" / "nightly"))
        self.assertEqual(rec["tag"], f"{utc_today()}-{rec['key']}")
        self.assertEqual(rec["changed"], {"T1_SOFTWARE_COMMIT": [None, f.t1_commit],
                                          "PKG_COMMIT": [None, f.A]})
        self.assertEqual(rec["repos"], {"T1_SOFTWARE_COMMIT": f"file://{f.t1_git}",
                                        "PKG_COMMIT": f"file://{f.pkg_git}"})
        self.assertTrue((nightly.build_dir(f.state, rec["key"]) / "build.log").exists())
        f.set_status(rec["key"], "success")
        self.assertEqual(f.poll(), 0)
        self.assertEqual(len(f.calls()), 1)
        self.assertTrue(f.poll_log_last().endswith("no change"))

    def test_head_moves(self):
        f = self.fixture()
        f.poll()
        f.set_status(f.only_record()["key"], "success")
        B = f.commit_pkg("b")
        f.poll()
        self.assertEqual(len(f.calls()), 2)
        newest = f.records()[0]
        self.assertEqual(newest["changed"], {"PKG_COMMIT": [f.A, B]})
        self.assertEqual(newest["pins"], {"PKG_COMMIT": B})

    def test_record_order_tiebreak(self):
        f = self.fixture()
        for key, job in [("aaaaaaaaaaaa", "120"), ("bbbbbbbbbbbb", "9"), ("cccccccccccc", "300")]:
            d = nightly.build_dir(f.state, key)
            d.mkdir(parents=True)
            (d / "record.json").write_text(json.dumps(
                {"key": key, "queued_at": "2026-11-09T10:00:00Z", "slurm_job_ids": [job]}))
        self.assertEqual([r["key"] for r in f.records()],
                         ["cccccccccccc", "aaaaaaaaaaaa", "bbbbbbbbbbbb"])

    def test_variant_change_builds(self):
        f = self.fixture()
        f.poll()
        f.set_status(f.only_record()["key"], "success")
        f.site["VARIANT"] = "cuda12x"
        f.write_site()
        f.poll()
        self.assertEqual(len(f.calls()), 2)

    def test_failed_not_rebuilt_review_retry(self):
        f = self.fixture()
        f.poll()
        key = f.only_record()["key"]
        f.set_status(key, "failed")
        for _ in range(3):
            f.poll()
        self.assertEqual(len(f.calls()), 1)
        rc, out, _ = f.main("review", key, "--note", "n")
        self.assertEqual(rc, 0)
        rv = json.loads((nightly.build_dir(f.state, key) / "review.json").read_text())
        self.assertEqual(rv["attempt"], 1)
        self.assertEqual(rv["note"], "n")
        self.assertEqual(nightly.review_state(f.only_record(), rv), "reviewed")
        tag = f.only_record()["tag"]
        rc, out, err = f.main("retry", key)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(f.calls()), 2)
        rec = f.only_record()
        self.assertEqual(rec["attempts"], 2)
        self.assertEqual(rec["status"], "queued")
        self.assertEqual(rec["tag"], tag)
        self.assertEqual(rec["slurm_job_ids"], ["101", "102"])
        self.assertTrue((nightly.build_dir(f.state, key) / "attempt-1.log").exists())
        self.assertEqual(nightly.review_state(rec, rv), "")

    def test_interrupt_retry_once(self):
        for state in ("PREEMPTED", "TIMEOUT", "NODE_FAIL"):
            with self.subTest(state=state):
                f = self.fixture()
                f.poll()
                f.set_sacct(101, state)
                f.poll()  # reconciles to interrupted, then resubmits it
                rec = f.only_record()
                self.assertEqual(rec["attempts"], 2)
                self.assertEqual(rec["status"], "queued")
                self.assertEqual(len(f.calls()), 2)
                f.set_sacct(102, "TIMEOUT")
                f.poll()
                self.assertEqual(f.only_record()["status"], "interrupted")
                self.assertEqual(len(f.calls()), 2)
                f.poll()
                self.assertEqual(len(f.calls()), 2)

    def test_cancel_and_oom_not_retried(self):
        for state, status, stage in [("CANCELLED", "cancelled", None),
                                     ("OUT_OF_MEMORY", "failed", "slurm OUT_OF_MEMORY")]:
            with self.subTest(state=state):
                f = self.fixture()
                f.poll()
                f.set_sacct(101, f"{state} by 123" if state == "CANCELLED" else state)
                f.poll()
                rec = f.only_record()
                self.assertEqual(rec["status"], status)
                self.assertEqual(rec["failed_stage"], stage)
                self.assertIsNotNone(rec["finished_at"])
                f.poll()
                self.assertEqual(len(f.calls()), 1)

    def test_active_blocks_submit(self):
        f = self.fixture()
        f.poll()
        key = f.only_record()["key"]
        f.set_sacct(101, "PENDING")
        B = f.commit_pkg("b")
        self.assertEqual(f.poll(), 0)
        self.assertEqual(len(f.calls()), 1)
        self.assertEqual(f.poll_log_last().split(" ", 1)[1], f"build active: {key}")
        last = json.loads((f.state / "last-poll.json").read_text())
        self.assertEqual(last["heads"]["PKG_COMMIT"], B)

    def test_lookup_failure(self):
        f = self.fixture()
        f.write_pins(f"file://{f.root / 'missing.git'}", f.A)
        f.commit_t1("broken repo url")
        self.assertEqual(f.poll(), 1)
        last = json.loads((f.state / "last-poll.json").read_text())
        self.assertIs(last["ok"], False)
        self.assertTrue(last["error"])
        self.assertEqual(f.calls(), [])
        self.assertEqual(f.records(), [])

    def test_lookup_failure_blocks_auto_retry(self):
        f = self.fixture()
        f.poll()
        f.set_sacct(101, "PREEMPTED")
        f.write_pins(f"file://{f.root / 'missing.git'}", f.A)
        f.commit_t1("broken repo url")
        self.assertEqual(f.poll(), 1)
        rec = f.only_record()
        self.assertEqual(rec["status"], "interrupted")
        self.assertEqual(rec["attempts"], 1)
        self.assertEqual(len(f.calls()), 1)

    def test_sbatch_failure(self):
        f = self.fixture()
        os.environ["FAKE_SBATCH_RC"] = "1"
        self.assertEqual(f.poll(), 1)
        self.assertEqual(list((f.state / "builds").iterdir()), [])
        self.assertFalse((f.state / "last-poll.json").exists())
        os.environ["FAKE_SBATCH_RC"] = "0"
        self.assertEqual(f.poll(), 0)
        self.assertEqual(len(f.records()), 1)

    def test_retry_sbatch_failure(self):
        f = self.fixture()
        f.poll()
        key = f.only_record()["key"]
        f.set_status(key, "failed")
        os.environ["FAKE_SBATCH_RC"] = "1"
        rc, _, err = f.main("retry", key)
        self.assertEqual(rc, 1)
        self.assertIn("sbatch failed", err)
        rec = f.only_record()
        self.assertEqual(rec["attempts"], 1)
        self.assertEqual(rec["status"], "failed")
        d = nightly.build_dir(f.state, key)
        self.assertTrue((d / "build.log").exists())
        self.assertFalse((d / "attempt-1.log").exists())

    def test_status_line(self):
        f = self.fixture()
        f.poll()
        rec = f.only_record()
        f.set_status(rec["key"], "failed")
        rc, out, _ = f.main("status")
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines(), [
            f"{rec['key']}  failed       a1  {rec['tag']}  t1={f.t1_commit[:8]}  "
            f"changed=2  needs review"])

    def test_dry_run(self):
        f = self.fixture()
        rc, out, _ = f.main("poll", "--dry-run")
        self.assertEqual(rc, 0)
        key = nightly.build_key("cuda12", f.t1_commit, {"PKG_COMMIT": f.A})
        self.assertIn(f"would submit {key}", out)
        self.assertFalse((f.state / "builds").exists())
        self.assertFalse((f.state / "last-poll.json").exists())
        self.assertFalse((f.state / "poll.log").exists())
        self.assertEqual(f.calls(), [])

    # -- run ---------------------------------------------------------------

    def polled(self):
        """Fixture after one poll, with FAKE_ARGV/FAKE_SEEN set."""
        f = self.fixture()
        f.poll()
        os.environ["FAKE_ARGV"] = str(f.root / "argv.txt")
        os.environ["FAKE_SEEN"] = str(f.root / "seen.txt")
        return f, f.only_record()

    def test_run_success_args(self):
        f, rec = self.polled()
        f.site["VARIANT"] = "other"
        f.write_site()
        os.environ["T1_SKIP_PIPELINE_TEST"] = "1"
        os.environ["FAKE_ENV_PROBE"] = str(f.root / "env.txt")
        rc, out, _ = f.main("run", rec["key"])
        self.assertEqual(rc, 0)
        done = f.only_record()
        self.assertEqual(done["status"], "success")
        self.assertIsNone(done["failed_stage"])
        self.assertIsNotNone(done["started_at"])
        self.assertIsNotNone(done["finished_at"])
        argv = (f.root / "argv.txt").read_text().splitlines()
        for a, b in [("--variant", "cuda12"), ("--images-dir", str(f.root / "img" / "nightly")),
                     ("--tag", rec["tag"]), ("--pin", f"PKG_COMMIT={f.A}"),
                     ("--work-dir", str(f.root / "work"))]:
            self.assertEqual(argv[argv.index(a) + 1], b)
        self.assertIn("--no-base-archive", argv)
        self.assertNotIn("T1_SKIP_PIPELINE_TEST=1", (f.root / "env.txt").read_text())
        self.assertIn("== stage-two ==", out)
        self.assertFalse((f.root / "work" / "t1-nightly" / f"{rec['key']}-src").exists())

    def test_run_updates_nightly_link(self):
        f, rec = self.polled()
        img = f.root / "img"
        link = img / "t1-lartpc-cuda12-nightly.sif"
        (img / "nightly").mkdir(parents=True)
        sif = img / "nightly" / f"t1-lartpc-cuda12-{rec['tag']}.sif"
        # The fake build.sh writes no image: no image, no link.
        f.main("run", rec["key"])
        self.assertFalse(os.path.lexists(link))
        sif.write_text("x")
        old = img / "nightly" / "t1-lartpc-cuda12-19700101-old.sif"
        old.write_text("y")
        link.symlink_to(f"nightly/{old.name}")
        f.set_status(rec["key"], "queued")
        f.main("run", rec["key"])
        self.assertEqual(os.readlink(link), f"nightly/{sif.name}")
        # A failed build leaves the link alone.
        link.unlink()
        link.symlink_to(f"nightly/{old.name}")
        os.environ["FAKE_BUILD_RC"] = "1"
        f.set_status(rec["key"], "queued")
        f.main("run", rec["key"])
        self.assertEqual(os.readlink(link), f"nightly/{old.name}")

    def test_compose_image_info(self):
        raw = json.loads(RAW_IMAGE)
        info = nightly.compose_image_info(
            raw, {"python_version": "3.11", "torch_version": "2.13.0+cu129", "jax_version": "0.7.0",
                  "geant4_version": "10.7", "gxx_version": "14.3", "root_version": "6.30"},
            1, None, "x.sif")
        rows = {r["name"]: r for r in info["core"]}
        self.assertEqual(rows["Python"]["installed"], "3.11.17")
        self.assertFalse(rows["Python"]["differs"])      # "3.11" matches "3.11.17"
        self.assertFalse(rows["PyTorch"]["differs"])     # exact, with the local version
        self.assertEqual(rows["JAX"]["installed"], "0.6.2 (jaxlib 0.6.2)")
        self.assertTrue(rows["JAX"]["differs"])
        self.assertFalse(rows["Geant4"]["differs"])      # series pin
        self.assertFalse(rows["g++ (conda)"]["differs"])
        self.assertTrue(rows["ROOT"]["differs"])         # "6.30" is not "6.32.16"
        self.assertEqual(rows["NumPy"]["pinned"], "")
        self.assertFalse(rows["NumPy"]["differs"])
        self.assertEqual(info["conda_count"], 4)

    def test_run_writes_image_info(self):
        f, rec = self.polled()
        nightly_dir = f.root / "img" / "nightly"
        nightly_dir.mkdir(parents=True)
        (nightly_dir / f"t1-lartpc-cuda12-{rec['tag']}.sif").write_text("x")
        nightly.write_json(nightly_dir / f"t1-base-cuda12-{rec['tag']}.manifest.json",
                           {"pins": {"jax_version": "0.7.0"}})
        path = nightly.build_dir(f.state, rec["key"]) / "image-info.json"
        os.environ["FAKE_APPTAINER_RC"] = "1"  # a failed inspection does not fail the build
        self.assertEqual(f.main("run", rec["key"])[0], 0)
        self.assertEqual(f.only_record()["status"], "success")
        self.assertFalse(path.exists())
        del os.environ["FAKE_APPTAINER_RC"]
        f.main("run", rec["key"])
        info = json.loads(path.read_text())
        self.assertEqual(info["sif"], str(nightly_dir / f"t1-lartpc-cuda12-{rec['tag']}.sif"))
        self.assertTrue({r["name"]: r for r in info["core"]}["JAX"]["differs"])
        self.assertEqual(len(info["pip"]), 6)

    def test_run_failure_stage(self):
        f, rec = self.polled()
        os.environ["FAKE_BUILD_RC"] = "1"
        rc, _, _ = f.main("run", rec["key"])
        self.assertEqual(rc, 1)
        done = f.only_record()
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["failed_stage"], "stage-two")

    def test_run_cache_lock(self):
        f, rec = self.polled()
        os.environ["FAKE_BUILD_RC"] = "1"
        os.environ["FAKE_EXTRA_LINE"] = ("build.sh: another build holds /x/.lock; "
                                         "retry later or drop --cache-dir")
        f.main("run", rec["key"])
        self.assertEqual(f.only_record()["failed_stage"], "cache lock")

    def test_run_checkout_failure(self):
        f, rec = self.polled()
        f.set_status(rec["key"], "queued", t1_software_commit="0" * 40)
        rc, _, err = f.main("run", rec["key"])
        self.assertEqual(rc, 1)
        done = f.only_record()
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["failed_stage"], "checkout")

    def test_run_ignores_checkout_edits(self):
        f = self.fixture()
        work = f.root / "t1work"
        f.git("clone", "--quiet", str(f.t1_git), str(work))
        f.git("-C", str(work), "config", "uploadpack.allowAnySHA1InWant", "true")
        f.site["T1_SOFTWARE_REPO"] = f"file://{work}"
        f.write_site()
        f.poll()
        rec = f.only_record()
        (work / "marker.txt").write_text("dirty\n")

        def snapshot():
            return [f.git("-C", str(work), *args) for args in
                    (["status", "--porcelain"], ["rev-parse", "HEAD"],
                     ["branch", "--show-current"], ["for-each-ref"])]

        before = snapshot()
        os.environ["FAKE_SEEN"] = str(f.root / "seen.txt")
        rc, _, _ = f.main("run", rec["key"])
        self.assertEqual(rc, 0)
        self.assertEqual((f.root / "seen.txt").read_text(), "committed\n")
        self.assertEqual(snapshot(), before)

    # -- prune -------------------------------------------------------------

    def test_prune(self):
        f = self.fixture()
        f.site["NIGHTLY_KEEP"] = "2"
        f.write_site()
        img = f.root / "img" / "nightly"
        img.mkdir(parents=True)

        def record(key, status, day, files):
            tag = f"202611{day:02d}-{key}"
            d = nightly.build_dir(f.state, key)
            d.mkdir(parents=True)
            nightly.write_json(d / "record.json", {
                "key": key, "status": status, "variant": "cuda12", "tag": tag,
                "images_dir": str(img), "queued_at": f"2026-11-{day:02d}T00:00:00Z",
                "finished_at": f"2026-11-{day:02d}T01:00:00Z", "slurm_job_ids": [str(day)]})
            for name in files:
                (img / name.format(tag=tag)).write_text("x")
            return tag

        files = ["t1-lartpc-cuda12-{tag}.sif", "t1-lartpc-cuda12-{tag}.oci.tar",
                 "t1-base-cuda12-{tag}.oci.tar", "t1-lartpc-cuda12-{tag}.manifest.json"]
        old_failed = record("f00000000000", "failed", 1, ["t1-lartpc-cuda12-{tag}.sif"])
        d1 = record("d10000000000", "success", 2, files)
        d2 = record("d20000000000", "success", 3, files)
        d3 = record("d30000000000", "success", 4, files)
        (img / "t1-lartpc-cuda12-latest.sif").symlink_to(f"t1-lartpc-cuda12-{d1}.sif")
        with contextlib.redirect_stdout(io.StringIO()):
            nightly.prune(f.state, nightly.load_site(f.state))
        names = {p.name for p in img.iterdir()}
        self.assertIn(f"t1-lartpc-cuda12-{d1}.sif", names)
        self.assertNotIn(f"t1-lartpc-cuda12-{d1}.oci.tar", names)
        self.assertNotIn(f"t1-base-cuda12-{d1}.oci.tar", names)
        for tag in (d2, d3):
            for name in files:
                self.assertIn(name.format(tag=tag), names)
        for tag in (d1, d2, d3):
            self.assertIn(f"t1-lartpc-cuda12-{tag}.manifest.json", names)
        self.assertIn(f"t1-lartpc-cuda12-{old_failed}.sif", names)
        self.assertIn("t1-lartpc-cuda12-latest.sif", names)

    def test_build_sbatch(self):
        r = subprocess.run(["bash", str(SITE_DIR / "build.sbatch"), "/bin/echo", "a", "b", "c"],
                           stdout=subprocess.PIPE, text=True, check=True)
        self.assertEqual(r.stdout, "run --script a b c\n")

    # -- dashboard ---------------------------------------------------------

    def dashboard_fixture(self, poll_age_hours=49, ok=True, error=None):
        """Five hand-made records and a last-poll.json poll_age_hours old."""
        f = self.fixture()
        img = f.root / "img" / "nightly"
        img.mkdir(parents=True)
        keys = {}
        for i, (name, status, attempts) in enumerate([
                ("ok", "success", 1), ("rev", "failed", 1), ("bad", "failed", 1),
                ("int", "interrupted", 2), ("gone", "success", 1)]):
            key = f"{name:0<12}"[:12]
            keys[name] = key
            tag = f"20261101-{key}"
            d = nightly.build_dir(f.state, key)
            d.mkdir(parents=True)
            rec = {
                "key": key, "variant": "cuda12", "t1_software_commit": "a" * 40,
                "pins": {"PKG_COMMIT": "b" * 40}, "changed": {"PKG_COMMIT": ["c" * 40, "b" * 40]},
                "repos": {"T1_SOFTWARE_COMMIT": "https://github.com/o/t1-software.git",
                          "PKG_COMMIT": "https://github.com/o/pkg"},
                "status": status, "failed_stage": "stage-two" if status == "failed" else None,
                "attempts": attempts, "slurm_job_ids": [str(100 + i)],
                "queued_at": f"2026-11-0{i + 1}T00:00:00Z", "started_at": None,
                "finished_at": None, "tag": tag, "images_dir": str(img)}
            if name == "ok":
                rec.update(started_at="2026-11-01T00:10:00Z", finished_at="2026-11-01T00:44:59Z")
            if name == "int":
                del rec["repos"]  # a record from before repos were stored
            nightly.write_json(d / "record.json", rec)
            (d / "build.log").write_text("== stage-one ==\n<b>log</b>\n")
            if name == "ok":
                (img / f"t1-lartpc-cuda12-{tag}.sif").write_text("x")
                raw = json.loads((f.root / "image_raw.json").read_text())
                nightly.write_json(d / "image-info.json", nightly.compose_image_info(
                    raw, {"jax_version": "0.7.0", "python_version": "3.11"}, 12_586_868_736,
                    "2026-11-01T00:30:00Z", img / f"t1-lartpc-cuda12-{tag}.sif"))
            if name == "rev":
                nightly.write_json(d / "review.json", {"time": "t", "user": "u",
                                                       "note": "upstream bug", "attempt": 1})
        when = datetime.now(timezone.utc).timestamp() - poll_age_hours * 3600
        nightly.write_json(f.state / "last-poll.json", {
            "time": datetime.fromtimestamp(when, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ok": ok, "error": error, "heads": {}})
        return f, keys

    def index_row(self, index_html, key):
        row = [line for line in index_html.splitlines()
               if line.startswith("<tr") and f"builds/{key}.html" in line]
        self.assertEqual(len(row), 1)
        return row[0]

    def test_render_pages(self):
        f, keys = self.dashboard_fixture()
        ok = nightly.load_json(nightly.record_path(f.state, keys["ok"]))
        self.assertTrue(nightly.update_nightly_link(nightly.load_site(f.state), ok))
        out = f.root / "out"
        rc, _, err = f.main("render", "--out", str(out))
        self.assertEqual(rc, 0, err)
        index = (out / "index.html").read_text()
        for key in keys.values():
            self.assertTrue((out / "builds" / f"{key}.html").exists())
        self.assertIn("<title>T1-LArTPC nightly builds</title>", index)
        self.assertIn("Poller stale", index)
        self.assertIn("reviewed", self.index_row(index, keys["rev"]))
        self.assertIn("needs review", self.index_row(index, keys["bad"]))
        self.assertNotIn("pruned", self.index_row(index, keys["ok"]))
        self.assertIn("pruned", self.index_row(index, keys["gone"]))
        self.assertIn("2026-11-01 00:00Z", self.index_row(index, keys["ok"]))
        self.assertIn("t1-lartpc-cuda12-nightly.sif", index)
        self.assertIn(f"&rarr; <code>nightly/t1-lartpc-cuda12-20261101-{keys['ok']}.sif</code>", index)
        self.assertIn("5 builds: 2 succeeded, 2 failed, 1 need review", index)
        self.assertIn("Base image: PyTorch <code>2.13.0+cu129</code>", index)
        self.assertIn(f'href="builds/{keys["ok"]}-image.html"', index)
        image = (out / "builds" / f"{keys['ok']}-image.html").read_text()
        self.assertIn("differs", image)
        self.assertIn("All 6 Python packages", image)
        self.assertFalse((out / "builds" / f"{keys['rev']}-image.html").exists())
        pages = [out / "index.html", *(out / "builds").glob("*.html")]
        for page in pages:
            text = page.read_text()
            self.assertNotIn("<b>log</b>", text)  # log text is escaped
            # One inline script (the theme toggle); no handlers, no absolute paths.
            self.assertEqual(text.count("<script"), 1)
            bare = re.sub(r"<script>.*?</script>", "", text, flags=re.S)
            self.assertNotRegex(bare, r"<script|\son\w+=|href=\"/")
            for href in re.findall(r'href="([^"]+)"', text):
                if href.startswith("https://"):
                    self.assertRegex(href, r"^https://github\.com/o/(t1-software|pkg)/commit/[0-9a-f]{40}$")
                else:
                    self.assertTrue((page.parent / href).exists(), f"{page}: {href}")

    def test_commit_links_and_duration(self):
        f, keys = self.dashboard_fixture()
        out = f.root / "out"
        f.main("render", "--out", str(out))
        page = (out / "builds" / f"{keys['ok']}.html").read_text()
        self.assertIn(f'href="https://github.com/o/pkg/commit/{"b" * 40}"', page)
        self.assertIn(f'href="https://github.com/o/pkg/commit/{"c" * 40}"', page)  # was
        self.assertIn(f'href="https://github.com/o/t1-software/commit/{"a" * 40}"', page)
        self.assertIn("0 h 34 min", page)
        # The log is not published: its location and tail are shown, no copy is made.
        self.assertIn(str(nightly.build_dir(f.state, keys["ok"]) / "build.log"), page)
        self.assertIn("== stage-one ==", page)
        self.assertEqual([p.name for p in (out / "builds").iterdir() if not p.name.endswith(".html")], [])
        index = (out / "index.html").read_text()
        self.assertIn(f'href="https://github.com/o/t1-software/commit/{"a" * 40}"',
                      self.index_row(index, keys["ok"]))
        # A record without stored repos shows plain commits.
        old = (out / "builds" / f"{keys['int']}.html").read_text()
        self.assertNotIn("https://", old)
        self.assertIn("bbbbbbbb", old)

    def test_no_banner_when_fresh(self):
        f, _ = self.dashboard_fixture(poll_age_hours=1)
        f.main("render", "--out", str(f.root / "out"))
        self.assertNotIn("Poller stale", (f.root / "out" / "index.html").read_text())

    def test_error_banner(self):
        f, _ = self.dashboard_fixture(poll_age_hours=1, ok=False, error="boom")
        f.main("render", "--out", str(f.root / "out"))
        self.assertIn("Last poll failed: boom", (f.root / "out" / "index.html").read_text())

    def test_review_flips_row(self):
        f, keys = self.dashboard_fixture(poll_age_hours=1)
        f.main("render")
        index = f.state / "site" / "index.html"
        self.assertIn("needs review", self.index_row(index.read_text(), keys["bad"]))
        rc, _, _ = f.main("review", keys["bad"], "--note", "x")
        self.assertEqual(rc, 0)
        row = self.index_row(index.read_text(), keys["bad"])
        self.assertIn("reviewed", row)
        self.assertNotIn("needs review", row)


if __name__ == "__main__":
    unittest.main()
