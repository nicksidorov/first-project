import struct
import runpy
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from spectrum_viewer import make_figure, read_spectrum


class SpectrumReaderTests(unittest.TestCase):
    def test_unsigned_big_endian(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'spectrum.bin'
            file.write_bytes(b'\x00\x00\x00\x01\x01\x02\x03\x04\xff\xff\xff\xff')
            self.assertEqual(read_spectrum(file), [1, 0x01020304, 0xFFFFFFFF])

    def test_invalid_size(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'spectrum.bin'
            for size in [0, 1, 2, 3, 5, 7]:
                with self.subTest(size=size):
                    file.write_bytes(bytes(size))
                    with self.assertRaises(ValueError):
                        read_spectrum(file)

    def test_zero_counts_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'spectrum.bin'
            file.write_bytes(struct.pack('>3I', 0, 12, 0))
            self.assertEqual(read_spectrum(file), [0, 12, 0])

    def test_synthetic_demo(self):
        script = Path(__file__).resolve().parents[1] / 'examples' / 'make_demo.py'
        demo = runpy.run_path(str(script))
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'demo.bin'
            demo['write_demo'](file)
            counts = read_spectrum(file)
        self.assertEqual(len(counts), 4096)
        self.assertEqual(counts.index(max(counts)), 650)
        for peak in [650, 1450, 2400]:
            self.assertGreater(counts[peak], counts[peak - 100])
            self.assertGreater(counts[peak], counts[peak + 100])


class FigureTests(unittest.TestCase):
    def test_axes_and_data_on_both_scales(self):
        for logarithmic, scale in [(False, 'linear'), (True, 'log')]:
            with self.subTest(scale=scale):
                counts = [0, 10, 100, 0]
                figure = make_figure(counts, 'Test', logarithmic)
                self.assertEqual(list(figure.data[0].x), [0, 1, 2, 3])
                self.assertEqual(list(figure.data[0].y), counts)
                self.assertEqual(figure.layout.yaxis.type, scale)

    def test_html_embeds_plotly_without_external_scripts(self):
        class ScriptSources(HTMLParser):
            def __init__(self):
                super().__init__()
                self.sources = []

            def handle_starttag(self, tag, attrs):
                if tag == 'script' and 'src' in dict(attrs):
                    self.sources.append(dict(attrs)['src'])

        html = make_figure([0, 10], 'Test').to_html(include_plotlyjs=True)
        parser = ScriptSources()
        parser.feed(html)
        self.assertEqual(parser.sources, [])
        self.assertIn('Plotly.newPlot', html)


if __name__ == '__main__':
    unittest.main()
