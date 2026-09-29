"""Profile native sparse phases and verify identical implicit/expanded diagrams.

Build first: cargo build --release --locked --example profile_sparse
The native worker reads Linux VmHWM for peak RSS in each fresh process.
"""

import argparse
import json
import random
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--binary", type=Path, default=Path("target/release/examples/profile_sparse")
    )
    parser.add_argument("--compare-binary", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["implicit", "expanded"],
        default=["implicit", "expanded"],
    )
    args = parser.parse_args()
    if args.points < 0 or args.dimensions <= 0 or args.repeats <= 0:
        parser.error("invalid size or repeat count")
    records = []
    with tempfile.TemporaryDirectory(prefix="polars-tda-profile-") as tmp:
        tmp = Path(tmp)
        fixture = tmp / "points.bin"
        rng = random.Random(args.seed)
        with fixture.open("wb") as stream:
            for _ in range(args.points * args.dimensions):
                stream.write(struct.pack("<d", rng.random()))
        variants = [("baseline", args.binary)]
        if args.compare_binary is not None:
            variants.append(("candidate", args.compare_binary))
        tasks = [
            (label, binary, mode) for label, binary in variants for mode in args.modes
        ]
        measurements = {(label, mode): [] for label, _, mode in tasks}
        reference = None
        for repeat in range(args.repeats):
            # Alternate ordering to reduce systematic warmup/load ordering effects.
            for label, binary, mode in tasks if repeat % 2 == 0 else reversed(tasks):
                output = tmp / "diagram.tsv"
                command = [
                    str(binary),
                    str(fixture),
                    str(args.points),
                    str(args.dimensions),
                    str(args.epsilon),
                    mode,
                    str(output),
                ]
                run = json.loads(subprocess.check_output(command, text=True))
                run["total_seconds"] = sum(
                    run[k]
                    for k in (
                        "packing_seconds",
                        "preparation_seconds",
                        "expansion_seconds",
                        "compute_seconds",
                    )
                )
                diagram = output.read_bytes()
                if reference is not None and diagram != reference:
                    raise RuntimeError(
                        "diagrams differ across paths or binaries (bitwise endpoints)"
                    )
                reference = diagram
                measurements[label, mode].append(run)
        for label, binary, mode in tasks:
            runs = measurements[label, mode]
            records.append(
                {
                    "variant": label,
                    "binary": str(binary),
                    "mode": mode,
                    "points": args.points,
                    "dimensions": args.dimensions,
                    "epsilon": args.epsilon,
                    "seed": args.seed,
                    "repeats": args.repeats,
                    "equal_diagrams": True,
                    "medians": {
                        k: statistics.median(run[k] for run in runs) for k in runs[0]
                    },
                    "runs": runs,
                }
            )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
