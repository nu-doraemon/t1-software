#!/bin/sh
# Print, as KEY=VALUE lines, every variable that conda's activate.d hooks
# in /opt/conda set, starting from a clean environment. The compiler
# hooks (gcc/gxx/binutils) are skipped: they belong to the opt-in
# /opt/setup-g4root.sh, never to the default environment.
#
# The hooks run under `env -i` because several (geant4-data) keep a value
# that is already set, so running them over the image's ENV would hide
# any drift. Used at build time only, by link-g4data.sh and check-env.sh.
exec env -i CONDA_PREFIX=/opt/conda PATH=/usr/bin:/bin /bin/sh -c '
    env | sort > /tmp/.hook-env.before
    for f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do
        case "$f" in *gcc_linux*|*gxx_linux*|*binutils_linux*) continue ;; esac
        . "$f" > /dev/null 2>&1
    done
    unset f
    env | sort | comm -13 /tmp/.hook-env.before - | grep -v "_CONDA_BACKUP=" | grep -v "^PWD="
    rm -f /tmp/.hook-env.before
'
