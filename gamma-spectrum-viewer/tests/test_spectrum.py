import csv
import math
from pathlib import Path
import struct
import tempfile
import unittest
from xml.etree import ElementTree

from spectrum_viewer import (
    MAX_BYTES, compact_path, demo_counts, fit_range, make_geometry,
    read_spectrum, write_csv, write_svg,
)


class ReaderTests(unittest.TestCase):
    def test_unsigned_big_endian(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'спектр.bin'
            file.write_bytes(b'\x00\x00\x00\x01\x01\x02\x03\x04\xff\xff\xff\xff')
            self.assertEqual(read_spectrum(file), [1, 0x01020304, 0xFFFFFFFF])

    def test_invalid_lengths_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'bad.bin'
            for length in [0, 1, 2, 3, 5, MAX_BYTES + 4]:
                with self.subTest(length=length):
                    file.write_bytes(bytes(length))
                    with self.assertRaises(ValueError):
                        read_spectrum(file)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_spectrum(Path(directory) / 'missing.bin')

    def test_zeros_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'zero.bin'
            file.write_bytes(struct.pack('>3I', 0, 12, 0))
            self.assertEqual(read_spectrum(file), [0, 12, 0])

    def test_demo(self):
        counts = demo_counts()
        self.assertEqual(len(counts), 4096)
        self.assertEqual(counts.index(max(counts)), 650)
        for peak in [650, 1450, 2400]:
            self.assertGreater(counts[peak], counts[peak - 100])
            self.assertGreater(counts[peak], counts[peak + 100])


class GeometryTests(unittest.TestCase):
    def test_full_channel_coverage(self):
        geo = make_geometry([2, 10, 4], -0.5, 2.5)
        x0, _, x1, _ = geo.rect
        self.assertEqual(geo.paths[0][0][0], x0)
        self.assertEqual(geo.paths[-1][-1][0], x1)
        self.assertAlmostEqual(geo.channel(x0), -0.5)
        self.assertAlmostEqual(geo.channel(x1), 2.5)

    def test_values_stay_inside_plot(self):
        for counts in ([0], [0, 0], [1], [0, 1, 0xFFFFFFFF, 0], demo_counts()):
            for logarithmic in (False, True):
                with self.subTest(size=len(counts), logarithmic=logarithmic):
                    geo = make_geometry(counts, -0.5, len(counts) - 0.5, logarithmic=logarithmic)
                    x0, y0, x1, y1 = geo.rect
                    for path in geo.paths:
                        for x, y in path:
                            self.assertTrue(math.isfinite(x) and math.isfinite(y))
                            self.assertTrue(x0 - 1e-8 <= x <= x1 + 1e-8)
                            self.assertTrue(y0 - 1e-8 <= y <= y1 + 1e-8)

    def test_logarithm_omits_zero_gaps_without_changing_values(self):
        counts = [0, 1, 10, 0, 100, 0]
        geo = make_geometry(counts, -0.5, 5.5, logarithmic=True)
        self.assertEqual(len(geo.paths), 2)
        self.assertEqual(counts, [0, 1, 10, 0, 100, 0])
        self.assertEqual(make_geometry([0, 0], -0.5, 1.5, logarithmic=True).paths, [])

    def test_zoom_clips_edges(self):
        geo = make_geometry([2, 8, 3, 10], 0.25, 2.2)
        self.assertAlmostEqual(geo.paths[0][0][0], geo.rect[0])
        self.assertAlmostEqual(geo.paths[-1][-1][0], geo.rect[2])

    def test_range_clamps_pan_and_extreme_zoom(self):
        self.assertEqual(fit_range(-100, 3, 10), (-0.5, 9.5))
        self.assertEqual(fit_range(8, 12, 10), (5.5, 9.5))
        self.assertEqual(fit_range(1.5, 1.5, 10), (1.5, 2.5))
        self.assertEqual(fit_range(-10, 10, 1), (-0.5, 0.5))

    def test_pixel_reduction_keeps_narrow_peaks(self):
        points = [(i / 100, 10.0) for i in range(10000)]
        points[5034] = (50.34, -100.0)
        reduced = compact_path(points)
        self.assertIn(points[5034], reduced)
        self.assertEqual(reduced[0], points[0])
        self.assertEqual(reduced[-1], points[-1])
        self.assertLess(len(reduced), len(points) / 5)


class ExportTests(unittest.TestCase):
    def test_csv_exact_values(self):
        counts = [0, 1, 4294967295]
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'test.csv'
            write_csv(file, counts)
            with file.open(encoding='utf-8-sig', newline='') as source:
                rows = list(csv.reader(source, delimiter=';'))
        self.assertEqual(rows, [['channel', 'counts'], ['0', '0'], ['1', '1'], ['2', '4294967295']])

    def test_svg_is_valid_and_escapes_title(self):
        geo = make_geometry([0, 100, 0], -0.5, 2.5)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'test.svg'
            write_svg(file, geo, 'Спектр <a & b>')
            root = ElementTree.parse(file).getroot()
        ns = '{http://www.w3.org/2000/svg}'
        self.assertEqual(root.find(ns + 'title').text, 'Спектр <a & b>')
        self.assertTrue(root.findall(ns + 'polyline'))


if __name__ == '__main__':
    unittest.main()
