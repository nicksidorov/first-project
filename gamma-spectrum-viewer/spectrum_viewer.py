#!/usr/bin/env python3
"""Compact, offline gamma spectrum viewer. Uses only Python and Tkinter."""
import argparse
import csv
from dataclasses import dataclass, field
from html import escape
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

VERSION = '0.2.0'
MAX_BYTES = 4 * 1024 * 1024
BLUE = '#2563eb'
TEXT = '#243247'
GRID = '#e5eaf1'


def read_spectrum(filename):
    with open(filename, 'rb') as source:
        data = source.read(MAX_BYTES + 1)
    if not data:
        raise ValueError('Файл пуст.')
    if len(data) > MAX_BYTES:
        raise ValueError('Файл больше 4 МиБ (1 048 576 каналов).')
    if len(data) % 4:
        raise ValueError('Размер файла должен быть кратен 4 байтам (uint32).')
    return [item[0] for item in struct.iter_unpack('>I', data)]


def demo_counts():
    """Artificial data, not a real measurement."""
    peaks = [(650, 16, 1200), (1450, 25, 600), (2400, 32, 250)]
    return [round(3 + 100 * math.exp(-i / 700) + sum(
        a * math.exp(-0.5 * ((i - c) / s) ** 2) for c, s, a in peaks
    )) for i in range(4096)]


def write_csv(filename, counts):
    with open(filename, 'w', encoding='utf-8-sig', newline='') as output:
        writer = csv.writer(output, delimiter=';')
        writer.writerow(['channel', 'counts'])
        writer.writerows(enumerate(counts))


def fit_range(left, right, size):
    """Clamp the horizontal viewport without shrinking it at an edge."""
    span = min(float(size), max(1.0, right - left))
    left = min(max(-0.5, left), size - 0.5 - span)
    return left, left + span


def nice_step(value):
    power = 10 ** math.floor(math.log10(max(value, 1e-12)))
    for factor in (1, 2, 5, 10):
        if factor * power >= value:
            return factor * power


def number(value):
    if abs(value) >= 1_000_000:
        return f'{value:.2g}'
    return f'{value:g}'


@dataclass
class Geometry:
    width: int
    height: int
    rect: tuple
    left: float
    right: float
    lines: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    paths: list = field(default_factory=list)

    def channel(self, pixel):
        x0, _, x1, _ = self.rect
        return self.left + (pixel - x0) / (x1 - x0) * (self.right - self.left)


def compact_path(points):
    """Keep first/last/min/max in each pixel column so narrow peaks survive."""
    if len(points) < 8000:
        return points
    result, group, column = [], [], None

    def flush():
        if group:
            indices = sorted({0, len(group) - 1,
                              min(range(len(group)), key=lambda i: group[i][1]),
                              max(range(len(group)), key=lambda i: group[i][1])})
            result.extend(group[i] for i in indices)

    for point in points:
        current = int(point[0])
        if current != column:
            flush()
            group = []
            column = current
        group.append(point)
    flush()
    return result


def make_geometry(counts, left, right, width=1000, height=560, logarithmic=False):
    width, height = max(360, width), max(240, height)
    x0, y0, x1, y1 = 88, 30, width - 24, height - 62
    geo = Geometry(width, height, (x0, y0, x1, y1), left, right)
    start = max(0, math.floor(left + 0.5))
    stop = min(len(counts), math.ceil(right + 0.5))
    visible = counts[start:stop]
    if logarithmic:
        positive = [v for v in visible if v > 0]
        lo = math.floor(math.log10(min(positive))) if positive else 0
        hi = max(lo + 1, math.log10(max(positive)) + 0.06) if positive else 1
        yticks = [(p, number(10 ** p)) for p in range(lo, math.floor(hi) + 1)]
    else:
        lo = 0
        step = max(1, nice_step(max(visible, default=0) * 1.05 / 5))
        hi = max(step, math.ceil(max(visible, default=0) * 1.05 / step) * step)
        yticks = [(i * step, number(i * step)) for i in range(round(hi / step) + 1)]
    for value, label in yticks:
        y = y1 - (value - lo) / (hi - lo) * (y1 - y0)
        geo.lines.append((x0, y, x1, y, GRID))
        geo.labels.append((x0 - 10, y, label, 'e'))
    step = max(1, nice_step((right - left) / max(2, (x1 - x0) // 100)))
    first = math.ceil(max(0, left) / step)
    last = math.floor(min(len(counts) - 1, right) / step)
    for i in range(first, last + 1):
        value = i * step
        x = x0 + (value - left) / (right - left) * (x1 - x0)
        geo.lines.append((x, y0, x, y1, GRID))
        geo.labels.append((x, y1 + 19, number(value), 'center'))
    geo.lines.extend([(x0, y0, x0, y1, TEXT), (x0, y1, x1, y1, TEXT)])
    geo.labels.extend([
        (x0, 13, 'Число импульсов' + (' · log₁₀' if logarithmic else ''), 'w'),
        ((x0 + x1) / 2, height - 15, 'Номер канала (с 0)', 'center'),
    ])
    path = []
    for index in range(start, stop):
        value = counts[index]
        if logarithmic and value == 0:
            if path:
                geo.paths.append(compact_path(path))
                path = []
            continue
        value = math.log10(value) if logarithmic else value
        y = y1 - (value - lo) / (hi - lo) * (y1 - y0)
        a = x0 + (max(left, index - 0.5) - left) / (right - left) * (x1 - x0)
        b = x0 + (min(right, index + 0.5) - left) / (right - left) * (x1 - x0)
        path.extend([(a, y), (b, y)])
    if path:
        geo.paths.append(compact_path(path))
    if logarithmic and not any(visible):
        geo.labels.append(((x0 + x1) / 2, (y0 + y1) / 2,
                           'В этом диапазоне все значения равны нулю', 'center'))
    return geo


def write_svg(filename, geo, title):
    anchor = {'w': 'start', 'e': 'end', 'center': 'middle'}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{geo.width}" '
             f'height="{geo.height}" viewBox="0 0 {geo.width} {geo.height}">',
             f'<title>{escape(title)}</title>',
             '<rect width="100%" height="100%" fill="white"/>']
    for x0, y0, x1, y1, color in geo.lines:
        parts.append(f'<line x1="{x0:g}" y1="{y0:g}" x2="{x1:g}" y2="{y1:g}" stroke="{color}"/>')
    for path in geo.paths:
        points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in path)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>')
    for x, y, label, align in geo.labels:
        parts.append(f'<text x="{x:g}" y="{y + 4:g}" text-anchor="{anchor[align]}" '
                     f'font-family="sans-serif" font-size="12" fill="{TEXT}">{escape(label)}</text>')
    parts.append('</svg>')
    Path(filename).write_text('\n'.join(parts), encoding='utf-8')


class SpectrumWindow:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.root = root
        root.title('Гамма-спектр')
        root.geometry('1060x700')
        root.minsize(760, 500)
        self.counts, self.name = [], ''
        self.left, self.right = -0.5, 4095.5
        self.geometry = None
        self._pending = None
        self._drag = None
        self._pan = None
        self.logarithmic = tk.BooleanVar(value=False)
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='Гамма-спектр', font=('Segoe UI', 16, 'bold')).pack(anchor='w', pady=(0, 8))
        controls = ttk.Frame(outer)
        controls.pack(fill='x', pady=(0, 10))
        for text, command in [('Открыть файл…', self.choose_file),
                              ('Пример', self.show_demo), ('Сброс масштаба', self.reset)]:
            ttk.Button(controls, text=text, command=command).pack(side='left', padx=(0, 7))
        ttk.Checkbutton(controls, text='Логарифмическая Y', variable=self.logarithmic,
                        command=self.redraw).pack(side='left', padx=(3, 12))
        self.svg_button = ttk.Button(controls, text='SVG…', command=self.save_svg, state='disabled')
        self.svg_button.pack(side='right')
        self.csv_button = ttk.Button(controls, text='CSV…', command=self.save_csv, state='disabled')
        self.csv_button.pack(side='right', padx=(0, 7))
        self.info = ttk.Label(outer, text='Откройте бинарный файл спектра или нажмите «Пример».', wraplength=1000)
        self.info.pack(fill='x', pady=(0, 8))
        self.canvas = tk.Canvas(outer, background='white', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.position = ttk.Label(outer, text='Канал: —    Импульсы: —')
        self.position.pack(anchor='w', pady=(7, 3))
        ttk.Label(outer, text='Колесо — масштаб · Выделение — диапазон · Правая кнопка — сдвиг · Двойной щелчок — сброс').pack(anchor='w')
        ttk.Label(outer, text='На логарифмической шкале нулевые значения скрыты.').pack(anchor='w', pady=(2, 0))
        self.canvas.bind('<Configure>', self.schedule_redraw)
        self.canvas.bind('<Motion>', self.on_motion)
        self.canvas.bind('<Leave>', self.on_leave)
        self.canvas.bind('<ButtonPress-1>', self.start_drag)
        self.canvas.bind('<B1-Motion>', self.move_drag)
        self.canvas.bind('<ButtonRelease-1>', self.end_drag)
        self.canvas.bind('<Double-Button-1>', self.reset)
        self.canvas.bind('<ButtonPress-3>', self.start_pan)
        self.canvas.bind('<B3-Motion>', self.move_pan)
        self.canvas.bind('<ButtonRelease-3>', self.end_pan)
        self.canvas.bind('<MouseWheel>', self.on_wheel)
        self.canvas.bind('<Button-4>', self.on_wheel)
        self.canvas.bind('<Button-5>', self.on_wheel)
        root.bind('<Control-o>', self.choose_file)
        root.bind('<Control-0>', self.reset)
        root.bind('<Escape>', self.cancel_drag)

    def choose_file(self, _event=None):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.root, title='Открыть бинарный спектр',
                                         filetypes=[('Все файлы', '*'), ('Бинарные файлы', '*.bin *.dat *.txt')])
        if path:
            self.load_file(path)

    def load_file(self, filename):
        from tkinter import messagebox
        try:
            counts = read_spectrum(filename)
        except (OSError, ValueError) as error:
            messagebox.showerror('Не удалось открыть спектр', str(error), parent=self.root)
            return False
        self.set_counts(counts, Path(filename).name)
        return True

    def set_counts(self, counts, name):
        self.counts, self.name = counts, name
        self.root.title(f'Гамма-спектр — {name}')
        self.info.configure(text=f'{name}   |   Каналов: {len(counts)}   |   Всего: {sum(counts):,}   |   '
                                 f'Максимум: {max(counts)} (канал {counts.index(max(counts))})')
        self.csv_button.configure(state='normal')
        self.svg_button.configure(state='normal')
        self.position.configure(text='Канал: —    Импульсы: —')
        self.reset()

    def show_demo(self):
        self.set_counts(demo_counts(), 'Искусственный пример')

    def reset(self, _event=None):
        self._drag = self._pan = None
        if self.counts:
            self.left, self.right = -0.5, len(self.counts) - 0.5
        self.redraw()

    def schedule_redraw(self, _event=None):
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(40, self.redraw)

    def redraw(self):
        if self._pending is not None:
            self.root.after_cancel(self._pending)
            self._pending = None
        self.canvas.delete('all')
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.info.configure(wraplength=max(300, width))
        if not self.counts:
            self.geometry = None
            self.canvas.create_text(max(width, 760) / 2, max(height, 300) / 2,
                                    text='Откройте файл спектра', fill='#66758b', font=('Segoe UI', 14))
            return
        geo = make_geometry(self.counts, self.left, self.right, width, height, self.logarithmic.get())
        self.geometry = geo
        for x0, y0, x1, y1, color in geo.lines:
            self.canvas.create_line(x0, y0, x1, y1, fill=color)
        for path in geo.paths:
            self.canvas.create_line(*[v for point in path for v in point], fill=BLUE, width=1.3, tags='spectrum')
        for x, y, label, anchor in geo.labels:
            self.canvas.create_text(x, y, text=label, anchor=anchor, fill=TEXT, font=('Segoe UI', 10))

    def inside(self, event):
        if not self.geometry:
            return False
        x0, y0, x1, y1 = self.geometry.rect
        return x0 <= event.x <= x1 and y0 <= event.y <= y1

    def on_motion(self, event):
        self.canvas.delete('cursor')
        if not self.inside(event):
            return
        index = min(len(self.counts) - 1, max(0, math.floor(self.geometry.channel(event.x) + 0.5)))
        self.position.configure(text=f'Канал: {index}    Импульсы: {self.counts[index]:,}')
        _, y0, _, y1 = self.geometry.rect
        self.canvas.create_line(event.x, y0, event.x, y1, fill='#94a3b8', dash=(3, 4), tags='cursor')

    def on_leave(self, _event):
        self.canvas.delete('cursor')

    def start_drag(self, event):
        if self.inside(event):
            self._drag = event.x

    def move_drag(self, event):
        if self._drag is None:
            return
        x0, y0, x1, y1 = self.geometry.rect
        self.canvas.delete('selection')
        self.canvas.create_rectangle(self._drag, y0, min(x1, max(x0, event.x)), y1,
                                     outline=BLUE, dash=(4, 3), tags='selection')

    def end_drag(self, event):
        if self._drag is None:
            return
        first = self._drag
        self._drag = None
        self.canvas.delete('selection')
        if abs(event.x - first) < 6:
            return
        x0, _, x1, _ = self.geometry.rect
        a, b = sorted([first, min(x1, max(x0, event.x))])
        self.left, self.right = fit_range(self.geometry.channel(a), self.geometry.channel(b), len(self.counts))
        self.redraw()

    def cancel_drag(self, _event=None):
        self._drag = self._pan = None
        self.canvas.delete('selection')

    def start_pan(self, event):
        if self.inside(event):
            self._pan = event.x, self.left, self.right

    def move_pan(self, event):
        if self._pan is None:
            return
        x, left, right = self._pan
        x0, _, x1, _ = self.geometry.rect
        shift = (x - event.x) / (x1 - x0) * (right - left)
        self.left, self.right = fit_range(left + shift, right + shift, len(self.counts))
        self.redraw()

    def end_pan(self, _event):
        self._pan = None

    def on_wheel(self, event):
        if not self.inside(event):
            return
        up = getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0
        factor = 0.75 if up else 1 / 0.75
        center = self.geometry.channel(event.x)
        self.left, self.right = fit_range(center + (self.left - center) * factor,
                                         center + (self.right - center) * factor, len(self.counts))
        self.redraw()

    def export(self, extension, writer):
        from tkinter import filedialog, messagebox
        filename = filedialog.asksaveasfilename(parent=self.root, title='Сохранить',
                                               initialfile='spectrum.' + extension,
                                               defaultextension='.' + extension,
                                               filetypes=[(extension.upper(), '*.' + extension)])
        if filename:
            try:
                writer(filename)
            except OSError as error:
                messagebox.showerror('Ошибка сохранения', str(error), parent=self.root)

    def save_csv(self):
        if self.counts:
            self.export('csv', lambda p: write_csv(p, self.counts))

    def save_svg(self):
        if self.geometry:
            self.export('svg', lambda p: write_svg(p, self.geometry, self.name))


def self_test(root, window, report):
    """Exercise the packaged GUI without any external files or dependencies."""
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory(prefix='gamma-check-') as folder:
        folder = Path(folder)
        file = folder / 'спектр с пробелом.bin'
        counts = demo_counts()
        file.write_bytes(struct.pack(f'>{len(counts)}I', *counts))
        assert window.load_file(file)
        root.update()
        window.redraw()
        assert window.counts == counts and window.canvas.find_withtag('spectrum')
        before = window.right - window.left
        x0, y0, x1, y1 = window.geometry.rect
        event = SimpleNamespace(x=int((x0 + x1) / 2), y=int((y0 + y1) / 2), delta=120)
        window.on_motion(event)
        window.on_wheel(event)
        assert window.right - window.left < before
        window.reset()
        window.logarithmic.set(True)
        window.redraw()
        assert window.canvas.find_withtag('spectrum')
        write_svg(folder / 'chart.svg', window.geometry, window.name)
        write_csv(folder / 'counts.csv', window.counts)
        assert (folder / 'chart.svg').stat().st_size > 1000
        with (folder / 'counts.csv').open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.reader(source, delimiter=';'))
        assert [int(row[1]) for row in rows[1:]] == counts
        window.set_counts([0] * 32, 'Zero spectrum')
        assert not window.canvas.find_withtag('spectrum')
        window.set_counts([1], 'Single channel')
        assert window.canvas.find_withtag('spectrum')
        window.show_demo()
        root.update()
        # The report is explicitly requested by the build/test command.
        Path(report).write_text(json.dumps({'ok': True, 'version': VERSION,
                                           'frozen': bool(getattr(sys, 'frozen', False)),
                                           'channels': len(counts), 'tk': root.tk.call('info', 'patchlevel')},
                                          ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Просмотр гамма-спектра: uint32 big-endian, без заголовка.')
    parser.add_argument('file', nargs='?', help='Бинарный файл спектра')
    parser.add_argument('--version', action='version', version=VERSION)
    parser.add_argument('--self-test', metavar='REPORT', help=argparse.SUPPRESS)
    args = parser.parse_args()
    import tkinter as tk
    root = tk.Tk()
    window = SpectrumWindow(root)
    if args.self_test:
        try:
            self_test(root, window, args.self_test)
        finally:
            root.destroy()
        return 0
    if args.file:
        window.load_file(args.file)
    root.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
