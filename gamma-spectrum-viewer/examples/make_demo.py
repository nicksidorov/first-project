"""Create an artificial uint32 big-endian spectrum for the viewer."""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectrum_viewer import demo_counts


def write_demo(filename):
    counts = demo_counts()
    Path(filename).write_bytes(struct.pack(f'>{len(counts)}I', *counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).with_name('demo.bin'))
    args = parser.parse_args()
    write_demo(args.output)
    print(f'Synthetic spectrum written to {args.output}')
