#!/usr/bin/python3

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import cv2
import numpy as np

DJXL_PATH = (Path(__file__) / "../build/tools/djxl").resolve()

if not DJXL_PATH.is_file():
    print("djxl command is not built", file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: pxentropy.py path/to/image.jxl ...", file=sys.stderr)
    sys.exit(1)

with TemporaryDirectory() as temp_dir:
    for i in range(1, len(sys.argv)):
        src_path = sys.argv[i]
        print("Decoding", os.path.basename(src_path), file=sys.stderr)

        dflif_result = subprocess.run(
            [DJXL_PATH, src_path, os.path.join(temp_dir, f"{i}.png")],
            stdout=subprocess.PIPE,
            text=True,
        )

        if dflif_result.returncode != 0:
            print("Failed to decode", src_path, file=sys.stderr)
            continue

        # 「Decode Entropy: where,p,fr,r,c,entropy」の形式で出力されているので、収集する
        groups = {}
        methods = set()
        for line in dflif_result.stdout.splitlines():
            if not line.startswith("Decode Entropy: "):
                continue

            record = line[len("Decode Entropy: ") :].split(",")
            methods.add(record[0])
            key = tuple(map(int, record[1:3]))
            entry = (int(record[3]), int(record[4]), float(record[5]))
            if key in groups:
                groups[key].append(entry)
            else:
                groups[key] = [entry]

        stem = Path(src_path).stem

        for (key, entries) in groups.items():
            dst_path = f"{stem} p={key[0]} fr={key[1]}.png"
            print("Writing", dst_path, file=sys.stderr)

            max_r = 0
            max_c = 0
            for (r, c, e) in entries:
                max_r = max(max_r, r)
                max_c = max(max_c, c)

            result = np.zeros((max_r + 1, max_c + 1))
            for (r, c, e) in entries:
                if result[r, c] == 0:
                    result[r, c] = e
                else:
                    print(
                        f"duplicate p={key[0]} fr={key[1]} r={r} c={c}", file=sys.stderr
                    )

            RANGE_MAX = 30  # 30 bit を一番濃い色とする
            print(
                f"p={key[0]} fr={key[1]}\tmin={result.min():.2f} max={result.max():.2f}bit avg={np.average(result):.2f}bit {methods}"
            )

            result = (255 / 30 * result).astype(np.uint8)
            cv2.imwrite(dst_path, result)
