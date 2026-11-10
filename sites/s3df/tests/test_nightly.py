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

# Stands in for scripts/build.sh inside the fetched t1-software commit.
FAKE_BUILD = textwrap.dedent("""\
    #!/bin/bash
    [[ -n "${FAKE_ARGV:-}" ]] && printf '%s\\n' "$@" > "$FAKE_ARGV"
    [[ -n "${FAKE_SEEN:-}" ]] && cat marker.txt > "$FAKE_SEEN"
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
        for name, body in [("sbatch", FAKE_SBATCH), ("sacct", FAKE_SACCT)]:
            f = bindir / name
            f.write_text(body)
            f.chmod(0o755)

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
        self._env = {k: os.environ.get(k) for k in [*GIT_ENV, "T1_NIGHTLY_STATE",
                                                     "FAKE_SBATCH_RC"]}
        os.environ.update(GIT_ENV)
        os.environ.pop("T1_NIGHTLY_STATE", None)
        os.environ.pop("FAKE_SBATCH_RC", None)
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
            "finished_at", "tag", "images_dir"})
        self.assertEqual(rec["status"], "queued")
        self.assertEqual(rec["attempts"], 1)
        self.assertEqual(rec["slurm_job_ids"], ["101"])
        self.assertEqual(rec["images_dir"], str(f.root / "img" / "nightly"))
        self.assertEqual(rec["tag"], f"{utc_today()}-{rec['key']}")
        self.assertEqual(rec["changed"], {"T1_SOFTWARE_COMMIT": [None, f.t1_commit],
                                          "PKG_COMMIT": [None, f.A]})
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


if __name__ == "__main__":
    unittest.main()
