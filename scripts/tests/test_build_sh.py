"""Fast tests of scripts/build.sh.

Fake podman, apptainer and nvidia-smi on PATH record their arguments and
write placeholder files, so these tests check option handling, pins,
tags, the output file set and the manifests in seconds, without a real
container build.

    python3 -m unittest discover -s scripts/tests -v
"""
import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILD_SH = REPO / "scripts" / "build.sh"

FAKE_PODMAN = textwrap.dedent("""\
    #!/bin/bash
    # Fake podman: log argv; copy --build-arg-file; write `save -o` targets.
    echo "$*" >> "$FAKE_ROOT/podman.calls"
    args=("$@")
    sub=""
    for a in "${args[@]}"; do
        case "$a" in build|save|run|rm|port) sub="$a"; break ;; esac
    done
    if [[ "$sub" == build ]]; then
        for ((i = 0; i < ${#args[@]}; i++)); do
            if [[ "${args[i]}" == --build-arg-file ]]; then
                n=$(ls "$FAKE_ROOT" | grep -c '^argfile-')
                cp "${args[i+1]}" "$FAKE_ROOT/argfile-$((n + 1))"
            fi
        done
    elif [[ "$sub" == save ]]; then
        for ((i = 0; i < ${#args[@]}; i++)); do
            [[ "${args[i]}" == -o ]] && echo oci > "${args[i+1]}"
        done
    fi
    exit 0
    """)

FAKE_APPTAINER = textwrap.dedent("""\
    #!/bin/bash
    # Fake apptainer: log argv; `build ... <target> <source>` writes <target>.
    echo "$*" >> "$FAKE_ROOT/apptainer.calls"
    if [[ "$1" == build ]]; then
        args=("$@")
        echo sif > "${args[-2]}"
    fi
    exit 0
    """)


def env_lines(path):
    """KEY=VALUE lines of a versions.env, as build.sh's argfile reads them."""
    out = {}
    for line in path.read_text().splitlines():
        if line and (line[0].isalpha() or line[0] == "_") and "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


class BuildShTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        bindir = self.tmp / "bin"
        bindir.mkdir()
        for name, body in [("podman", FAKE_PODMAN), ("apptainer", FAKE_APPTAINER),
                           ("nvidia-smi", "#!/bin/bash\nexit 1\n")]:
            f = bindir / name
            f.write_text(body)
            f.chmod(0o755)
        self.img = self.tmp / "img"

    def tearDown(self):
        self._tmp.cleanup()

    def run_build(self, *args, env_extra=None, images_dir=True):
        env = {k: v for k, v in os.environ.items()
               if k not in ("LSCRATCH", "T1_SOFTWARE_IMAGES_DIR",
                            "T1_SOFTWARE_CACHE_DIR", "T1_SKIP_PIPELINE_TEST")}
        env["PATH"] = f"{self.tmp / 'bin'}:/usr/bin:/bin"
        env["FAKE_ROOT"] = str(self.tmp)
        env.update(env_extra or {})
        cmd = [str(BUILD_SH), "--work-dir", str(self.tmp / "w")]
        if images_dir:
            cmd += ["--images-dir", str(self.img)]
        cmd += list(args)
        return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)

    def calls(self, name):
        path = self.tmp / f"{name}.calls"
        return path.read_text() if path.exists() else None

    def manifest(self, layer, tag):
        return json.loads((self.img / f"t1-{layer}-cuda12-{tag}.manifest.json").read_text())

    def test_help(self):
        r = self.run_build("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        for opt in ["--variant", "--work-dir", "--images-dir", "--cache-dir", "--pin",
                    "--tag", "--no-sif", "--no-pipeline-test", "--gpu",
                    "--squashfs-mem", "--help"]:
            self.assertIn(opt, r.stdout)

    def test_unknown_pin(self):
        r = self.run_build("--pin", "NOPE_COMMIT=x")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--pin NOPE_COMMIT is not in", r.stderr)
        self.assertIsNone(self.calls("podman"))

    def test_empty_repo(self):
        r = self.run_build("--pin", "GOOP_REPO=")
        self.assertEqual(r.returncode, 2)
        self.assertIn("GOOP_REPO is empty", r.stderr)
        self.assertIsNone(self.calls("podman"))

    def test_unknown_option(self):
        r = self.run_build("--bogus")
        self.assertEqual(r.returncode, 2)
        self.assertIsNone(self.calls("podman"))

    def test_pin_and_tag(self):
        git_status = ["git", "-C", str(REPO), "status", "--porcelain"]
        before = subprocess.run(git_status, stdout=subprocess.PIPE,
                                universal_newlines=True, check=True).stdout
        r = self.run_build("--pin", "GOOP_COMMIT=abc123", "--tag", "t1", "--no-sif")
        self.assertEqual(r.returncode, 0, r.stderr)
        lartpc_args = (self.tmp / "argfile-2").read_text().splitlines()
        self.assertIn("GOOP_COMMIT=abc123", lartpc_args)
        self.assertEqual(sum(1 for l in lartpc_args if l.startswith("GOOP_COMMIT=")), 1)
        self.assertEqual(sorted(p.name for p in self.img.iterdir()), sorted([
            "t1-base-cuda12-t1.oci.tar", "t1-lartpc-cuda12-t1.oci.tar",
            "t1-base-cuda12-t1.manifest.json", "t1-lartpc-cuda12-t1.manifest.json",
            "t1-base-cuda12-latest.oci.tar", "t1-lartpc-cuda12-latest.oci.tar",
        ]))
        m = self.manifest("lartpc", "t1")
        self.assertEqual(m["pins"]["goop_commit"], "abc123")
        self.assertEqual(m["pins"]["goop_repo"], "https://github.com/CIDeR-ML/goop.git")
        self.assertEqual(m["tag"], "t1")
        after = subprocess.run(git_status, stdout=subprocess.PIPE,
                               universal_newlines=True, check=True).stdout
        self.assertEqual(before, after)

    def test_default_pins_full_set(self):
        r = self.run_build("--tag", "t2", "--no-pipeline-test")
        self.assertEqual(r.returncode, 0, r.stderr)
        names = {p.name for p in self.img.iterdir()}
        self.assertIn("t1-lartpc-cuda12-t2.sif", names)
        self.assertIn("t1-lartpc-cuda12-latest.sif", names)
        for layer, path in [("lartpc", REPO / "subgroups/lartpc/versions.env"),
                            ("base", REPO / "base/versions.env")]:
            pins = self.manifest(layer, "t2")["pins"]
            expected = {k.lower(): v for k, v in env_lines(path).items()}
            self.assertEqual(pins, expected)
        self.assertIn("== dprod full-chain test: skipped ==", r.stdout)
        self.assertNotIn("--nv", self.calls("apptainer"))

    def test_no_base_archive(self):
        r = self.run_build("--tag", "t5", "--no-base-archive", "--no-pipeline-test")
        self.assertEqual(r.returncode, 0, r.stderr)
        names = {p.name for p in self.img.iterdir()}
        self.assertEqual(names, {
            "t1-lartpc-cuda12-t5.oci.tar", "t1-lartpc-cuda12-t5.sif",
            "t1-lartpc-cuda12-latest.oci.tar", "t1-lartpc-cuda12-latest.sif",
            "t1-lartpc-cuda12-t5.manifest.json", "t1-base-cuda12-t5.manifest.json"})
        self.assertIn("/opt/t1-base-tests/image-test.sh", self.calls("podman"))
        self.assertEqual(self.run_build("--no-base-archive=1").returncode, 2)

    def test_env_fallbacks(self):
        env_img = self.tmp / "envimg"
        r = self.run_build("--tag", "t4", images_dir=False, env_extra={
            "T1_SOFTWARE_IMAGES_DIR": str(env_img), "T1_SKIP_PIPELINE_TEST": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((env_img / "t1-lartpc-cuda12-t4.sif").exists())
        self.assertIn("== dprod full-chain test: skipped ==", r.stdout)

    def test_squashfs_mem(self):
        r = self.run_build("--squashfs-mem", "3G", "--tag", "t3", "--no-pipeline-test")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("-mem 3G", self.calls("apptainer"))


if __name__ == "__main__":
    unittest.main()
