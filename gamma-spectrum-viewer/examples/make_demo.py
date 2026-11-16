"""Create a synthetic spectrum for checking the viewer, not measured data."""
import argparse
import math
from pathlib import Path
import struct


def write_demo(filename):
    counts = []
    peaks = [(650, 16, 1200), (1450, 25, 600), (2400, 32, 250)]
    for channel in range(4096):
        value = 3 + 100 * math.exp(-channel / 700)
        for center, width, amplitude in peaks:
            value += amplitude * math.exp(-0.5 * ((channel - center) / width) ** 2)
        counts.append(round(value))
    Path(filename).write_bytes(struct.pack(f'>{len(counts)}I', *counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path,
                        default=Path(__file__).with_name('demo.bin'))
    args = parser.parse_args()
    write_demo(args.output)
    print(f'Synthetic spectrum written to {args.output}')


if __name__ == '__main__':
    main()
