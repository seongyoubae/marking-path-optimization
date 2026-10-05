"""Redraw the supplied marking schematic with readable route annotations."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from src.problem import Problem, validate_solution

# Illustrative sheet coordinates preserve the supplied example's arrangement.
# These are not original production coordinates or a dimensioned fabrication drawing.
ENDPOINTS = [
    (0.5, 3.5),
    (8, 3.5),
    (5, 21),
    (21, 21),
    (3, 39.3),
    (25, 39.3),
    (9.5, 58.5),
    (9.5, 40.5),
    (22, 40.5),
    (37.5, 58.5),
    (41.8, 58.5),
    (46, 57),
    (46.5, 56.2),
    (46.5, 52.5),
    (53.5, 56.2),
    (53.5, 52.5),
    (59.1, 58.5),
    (54.2, 57),
    (62.5, 58.5),
    (78.5, 40.4),
    (90.5, 58.5),
    (90.5, 40.7),
    (78.8, 39.3),
    (96.5, 39.3),
    (79.5, 21),
    (96.5, 21),
    (91.5, 3.2),
    (99, 3.2),
]
SOLUTION = list(range(14)) + [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
LABELS = [
    (5, 8),
    (12, 25),
    (16, 35.5),
    (5, 50),
    (27, 52),
    (38.5, 54),
    (42, 50),
    (58, 50),
    (61.5, 54),
    (75, 50),
    (95, 50),
    (87, 35.5),
    (86, 25),
    (94, 8),
]
# Gray and white contours reproduce the visual context of the research schematic.
# They do not introduce an optimizer constraint or imply one operation per contour.
WHITE_CONTOURS = [
    "M47,55 L43,44 L26,44 L26,40 C26,35 31,35 31,39 L44,39 L40,25 "
    "L23,25 L23,21 C23,16 28,16 28,20 L38,20 L29,0 L71,0 L63,20 "
    "L75,20 C75,16 80,16 80,21 L80,25 L62,25 L58,39 L69,39 "
    "C69,35 74,35 74,40 L74,44 L55,44 L52,55 Z",
    "M46.2,60 L53.8,60 L53.8,58.5 C53.8,53.8 46.2,53.8 46.2,58.5 Z",
    "M14,47.5 C8.5,47.5 8.5,56.5 14,56.5 L24,56.5 C30,56.5 30,47.5 24,47.5 Z",
    "M76,47.5 C70,47.5 70,56.5 76,56.5 L86,56.5 C92,56.5 92,47.5 86,47.5 Z",
]
GRAY_CONTOURS = [
    "M12,49.2 L25.5,49.2 L27.5,54.8 L13.5,54.8 Z",
    "M74.5,49.2 L88,49.2 L86.5,54.8 L72.5,54.8 Z",
    "M32.8,9 L43.8,41.7 L51.9,47.6 L55.3,38.6 L44.7,5.2 L36.4,0 Z",
    "M44.5,10 L48.5,21 L55.5,27 L58.5,20.5 L53.5,4 L47.5,0.5 Z",
]


def schematic_problem():
    coords = np.asarray(ENDPOINTS, dtype=float)
    return Problem([0, 0], coords, coords[np.arange(len(coords)) ^ 1], "reference_schematic")


def render_schematic():
    problem = schematic_problem()
    sequence, direction = validate_solution(problem, SOLUTION)
    navy, muted, coral = "#153746", "#536a78", "#c93f35"
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1310" '
        'viewBox="0 0 1280 1310" role="img" aria-labelledby="title desc">',
        '<title id="title">Marking operations and the travel between their endpoints</title>',
        '<desc id="desc">A readable redraw of the supplied research schematic. Gray and '
        "white shapes retain its visual context. Fourteen numbered red marking segments "
        "follow the supplied arrangement, rather than placing one arbitrary stroke inside "
        "each drawn part. Dashed navy connectors run from the origin to the first entry, "
        "between each exit and the next entry, and back to the origin. Coordinates and "
        "the manually selected route are illustrative, not an industrial dataset or a "
        "reported optimizer result. Contours are not collision constraints.</desc>",
        '<defs><marker id="travel_arrow" markerWidth="15" markerHeight="15" refX="13" '
        'refY="7.5" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M2,2 L13,7.5 L2,13" fill="none" stroke="#153746" stroke-width="2.5"/>'
        '</marker><marker id="mark_arrow" markerWidth="16" markerHeight="16" refX="14" '
        'refY="8" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M1,1 L15,8 L1,15 Z" fill="#c93f35"/></marker></defs>',
        '<rect width="1280" height="1310" fill="#fff"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
    ]

    def text(x, y, value, size=30, color=navy, weight="normal", anchor="start"):
        elements.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
        )

    def line(a, b, color, width=3.8, arrow=None, dashed=False, halo=False, attrs=""):
        extra = f' marker-end="url(#{arrow})"' if arrow else ""
        if dashed:
            extra += ' stroke-dasharray="12 9"'
        coords = f'x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}"'
        if halo:
            elements.append(f'<line {coords} stroke="#fff" stroke-width="{width + 3}"/>')
        elements.append(
            f'<line {coords} stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"{extra}{attrs}/>'
        )

    def point(p):
        return 60 + 11.52 * float(p[0]), 901.2 - 11.52 * float(p[1])

    def contour(path, fill):
        elements.append(
            f'<path d="{path}" transform="translate(60 901.2) scale(11.52 -11.52)" '
            f'fill="{fill}" stroke="#435f6f" stroke-width="0.2" '
            'stroke-linejoin="round"/>'
        )

    text(48, 61, "Marking path optimization", 44, weight="bold")
    text(48, 109, "Choose the order and direction of the marking operations.", 30, muted)
    line((51, 164), (119, 164), coral, 5.5, "mark_arrow")
    text(137, 175, "Marking", 31, weight="bold")
    line((510, 164), (578, 164), navy, 3.8, "travel_arrow", dashed=True)
    text(596, 175, "Travel to minimize", 31, weight="bold")
    elements.append(
        '<rect x="60" y="210" width="1152" height="691.2" '
        'fill="#dce2e6" stroke="#677e8b" stroke-width="2.4"/>'
    )
    for path in WHITE_CONTOURS:
        contour(path, "#fff")
    for path in GRAY_CONTOURS:
        contour(path, "#c7d0d6")
    # Retain the central profiles shown in the supplied schematic.
    for a, b in [((51.9, 47.6), (36.4, 0)), ((55.5, 27), (47.5, 0.5))]:
        line(point(a), point(b), "#435f6f", 2.3)
    for p in [(25.5, 5), (74.5, 5)]:
        x, y = point(p)
        elements.append(
            f'<circle cx="{x}" cy="{y}" r="12" fill="#fff" stroke="#435f6f" stroke-width="2.3"/>'
        )

    selected = 2 * sequence + direction
    entries, exits = problem.coords[selected], problem.coords_paired[selected]
    for leg, (a, b) in enumerate(
        zip(np.vstack([problem.origin, exits]), np.vstack([entries, problem.origin]))
    ):
        pa, pb = np.asarray(point(a)), np.asarray(point(b))
        distance = np.linalg.norm(pb - pa)
        end = pb - (pb - pa) * min(9 / distance, 0.2)
        # Tiny transitions have no arrowhead, avoiding a pile-up at adjacent endpoints.
        marker = "travel_arrow" if distance >= 22 else None
        line(pa, end, navy, 3.8, marker, dashed=True, halo=True, attrs=f' data-travel-leg="{leg}"')
    for visit, operation in enumerate(sequence):
        a, b = point(entries[visit]), point(exits[visit])
        line(
            a,
            b,
            coral,
            5.5,
            "mark_arrow",
            halo=True,
            attrs=f' data-marking-operation="{int(operation)}"',
        )
        elements.append(f'<circle cx="{a[0]}" cy="{a[1]}" r="5" fill="{coral}"/>')
        lx, ly = point(LABELS[int(operation)])
        # Short callouts tie operation IDs to their strokes, including the compact top group.
        label = np.asarray([lx, ly])
        stroke_start, stroke_end = np.asarray(a), np.asarray(b)
        vector = stroke_end - stroke_start
        fraction = np.clip(np.dot(label - stroke_start, vector) / np.dot(vector, vector), 0, 1)
        closest = stroke_start + fraction * vector
        offset = closest - label
        clearance = np.linalg.norm(offset)
        if clearance > 31:
            line(label + offset * (26 / clearance), closest, "#738995", 1.6)
        elements.append(
            f'<circle cx="{lx}" cy="{ly}" r="24" fill="#fff" stroke="#8aa0ad" stroke-width="1.5"/>'
        )
        text(lx, ly + 11, int(operation) + 1, 31, weight="bold", anchor="middle")

    ox, oy = point(problem.origin)
    elements.append(f'<circle cx="{ox}" cy="{oy}" r="9" fill="{navy}"/>')
    text(60, 950, "O  ·  start / return", 30, weight="bold")
    text(1212, 950, "Fixed schematic layout", 29, muted, anchor="end")
    elements.append('<line x1="48" y1="978" x2="1232" y2="978" stroke="#d7e0e5"/>')
    text(48, 1032, "Reversible direction", 32, weight="bold")
    for bit, base in [(0, 490), (1, 876)]:
        text(base, 1032, bit, 31, weight="bold")
        text(base + 47, 1032, "A", 30)
        text(base + 279, 1032, "B", 30, anchor="end")
        a, b = ((base + 86, 1021), (base + 240, 1021))
        if bit:
            a, b = b, a
        line(a, b, coral, 5.5, "mark_arrow")

    text(48, 1103, "Example route · read from left to right", 32, weight="bold")
    text(48, 1157, "Order", 30, weight="bold")
    text(48, 1220, "Direction", 30, weight="bold")
    for visit, (operation, bit) in enumerate(zip(sequence, direction)):
        x = 265 + 69 * visit
        elements.append(
            f'<rect x="{x - 25}" y="1123" width="50" height="47" rx="5" fill="#edf2f5"/>'
        )
        text(x, 1157, int(operation) + 1, 31, weight="bold", anchor="middle")
        text(x, 1220, int(bit), 31, weight="bold", anchor="middle")
    text(48, 1286, "Redrawn schematic · Illustrative coordinates and route", 27, muted)
    elements.extend(["</g>", "</svg>"])
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/problem.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_schematic(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
