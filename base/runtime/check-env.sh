#!/bin/bash
# Build-time guard: the image's static ENV must cover every variable that
# conda's activate.d hooks would set. Run as the last step of every
# Containerfile, because any conda install can add a new hook.
#
# A variable is covered when:
#   - *PATH lists: every entry the hook sets is in the ENV value;
#   - other values: the ENV value resolves (realpath) to the hook's value,
#     so /opt/g4data/<VAR> symlinks match their versioned targets.
# On failure, add or fix the listed ENV lines in the Containerfile.
set -eu
fail=0
while IFS='=' read -r var value; do
    current="${!var-}"
    if [ -z "$current" ]; then
        echo "check-env: $var is set by a conda hook but missing from ENV (hook value: $value)" >&2
        fail=1
        continue
    fi
    case "$var" in
        *PATH)
            IFS=':' read -ra parts <<< "$value"
            for p in "${parts[@]}"; do
                [ -z "$p" ] && continue
                case ":$current:" in
                    *":$p:"*) ;;
                    *) echo "check-env: $var lacks '$p' (ENV value: $current)" >&2; fail=1 ;;
                esac
            done
            ;;
        *)
            if [ "$current" != "$value" ] && \
               [ "$(realpath -q "$current" || true)" != "$(realpath -q "$value" || echo "$value")" ]; then
                echo "check-env: $var: ENV '$current' != hook '$value'" >&2
                fail=1
            fi
            ;;
    esac
done < <(/opt/t1-runtime/hook-env.sh)
[ "$fail" = 0 ] || exit 1
echo "check-env: static ENV covers every conda activate.d variable"
