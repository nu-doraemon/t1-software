#!/bin/sh
# Give every Geant4 dataset variable a version-free path, so the
# Containerfile can set it with a static ENV line. The geant4-data hooks
# point each G4*DATA variable at a versioned directory (e.g.
# G4NEUTRONHPDATA -> .../Geant4/data/NDL4.6) that changes whenever the
# geant4 pin changes; /opt/g4data/<VAR> is a symlink to that directory.
set -eu
mkdir -p /opt/g4data
/opt/t1-runtime/hook-env.sh | grep -E '^G4[A-Z]*DATA=' | while IFS='=' read -r var value; do
    [ -d "$value" ] || { echo "link-g4data: $var=$value is not a directory" >&2; exit 1; }
    ln -sfn "$value" "/opt/g4data/$var"
    echo "link-g4data: /opt/g4data/$var -> $value"
done
