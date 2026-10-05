"""Draw an original synthetic layout with multiple marking operations per part."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from src.problem import Problem, validate_solution

# Independently constructed, non-dimensional example geometry.
# Multiple marking operations can belong to a single drawn part.
# Contours are visual context; the solver consumes only paired endpoints.
ENDPOINTS = [
    (10, 9),
    (46, 9),
    (10, 22),
    (10, 49),
    (15, 55),
    (38, 55),
    (63, 56),
    (109, 56),
    (112, 54),
    (112, 40),
    (63, 42),
    (108, 42),
    (65, 9),
    (82, 9),
    (98, 28),
    (114, 28),
]
# Directions belong to visit positions, including the final 8 -> 7 visit order.
SOLUTION = [0, 1, 2, 3, 4, 5, 7, 6] + [0, 0, 0, 0, 0, 1, 0, 1]
LABELS = [(49, 11), (6, 36), (31, 59.5), (78, 61.5), (117, 47), (98, 38), (76, 5.5), (100, 32)]
PARTS = [
    {
        "outline": "M5,6 L53,6 L53,20 L47,20 L47,26 L35,26 L35,30 "
        "Q30,30 30,33 Q30,36 35,36 L35,41 L43,41 L43,58 L10,58 L5,53 Z",
        "holes": [
            "M20,46 L29,46 A3,3 0 0 1 29,52 L20,52 A3,3 0 0 1 20,46 Z",
            "M20,11 L35,11 A3,3 0 0 1 35,17 L20,17 A3,3 0 0 1 20,11 Z",
        ],
        "operations": [0, 1, 2],
    },
    {
        "outline": "M57,39 L62,35 L81,35 L81,39 Q84,43 87,39 L87,35 "
        "L110,35 L116,41 L116,59 L62,59 L57,54 Z",
        "holes": [
            "M68.5,45 L76.5,45 A3.5,3.5 0 0 1 76.5,52 L68.5,52 A3.5,3.5 0 0 1 68.5,45 Z",
            "M94.5,45 L102.5,45 A3.5,3.5 0 0 1 102.5,52 L94.5,52 A3.5,3.5 0 0 1 94.5,45 Z",
        ],
        "operations": [3, 4, 5],
    },
    {
        "outline": "M57,6 L89,6 L89,12 L65,31 L57,31 L57,10 Q62,10 62,6 Z",
        "holes": ["M70,18 A3,3 0 1 0 64,18 A3,3 0 1 0 70,18 Z"],
        "operations": [6],
    },
    {
        "outline": "M94,31 L117,31 L117,6 L111,6 L95,23 Q94,26 94,31 Z",
        "holes": ["M112.8,23 A2.8,2.8 0 1 0 107.2,23 A2.8,2.8 0 1 0 112.8,23 Z"],
        "operations": [7],
    },
]


def schematic_problem():
    coords = np.asarray(ENDPOINTS, dtype=float)
    return Problem([0, 0], coords, coords[np.arange(len(coords)) ^ 1], "synthetic_schematic")


def render_schematic():
    problem = schematic_problem()
    sequence, direction = validate_solution(problem, SOLUTION)
    navy, muted, coral = "#153746", "#536a78", "#c93f35"
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1110" '
        'viewBox="0 0 1280 1110" role="img" aria-labelledby="title desc">',
        '<title id="title">Marking operations and the travel between their endpoints</title>',
        '<desc id="desc">An original synthetic layout with four independently constructed '
        "shipbuilding-style contours. Two larger parts each contain three marking operations; "
        "two smaller brackets each contain one. Eight numbered red strokes have reversible "
        "paired endpoints. Dashed connectors show the origin approach, transfers and return. "
        "The example order ends with operation eight followed by operation seven. Coordinates "
        "and route are illustrative. Contours are not collision constraints.</desc>",
        '<defs><marker id="travel_arrow" markerWidth="15" markerHeight="15" refX="13" '
        'refY="7.5" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M2,2 L13,7.5 L2,13" fill="none" stroke="#153746" stroke-width="2.5"/>'
        '</marker><marker id="mark_arrow" markerWidth="16" markerHeight="16" refX="14" '
        'refY="8" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M1,1 L15,8 L1,15 Z" fill="#c93f35"/></marker></defs>',
        '<rect width="1280" height="1110" fill="#fff"/>',
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
        return 60 + 9.6 * float(p[0]), 815 - 9.6 * float(p[1])

    text(48, 61, "Marking path optimization", 44, weight="bold")
    text(48, 109, "Optimize the travel between marking operations.", 30, muted)
    line((51, 164), (119, 164), coral, 5.5, "mark_arrow")
    text(137, 175, "Marking", 31, weight="bold")
    line((510, 164), (578, 164), navy, 3.8, "travel_arrow", dashed=True)
    text(596, 175, "Travel to minimize", 31, weight="bold")
    elements.append(
        '<rect x="60" y="200.6" width="1152" height="614.4" '
        'fill="#f5f7f8" stroke="#677e8b" stroke-width="2.4"/>'
    )
    for part in PARTS:
        shape = " ".join([part["outline"], *part["holes"]])
        elements.append(
            f'<path d="{shape}" transform="translate(60 815) scale(9.6 -9.6)" '
            'fill="#dce4e9" fill-rule="evenodd" stroke="#435f6f" '
            'stroke-width="0.25" stroke-linejoin="round"/>'
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
        # Short callouts tie operation IDs to their strokes, without covering the strokes.
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
    text(60, 859, "O  ·  start / return", 30, weight="bold")
    text(1212, 859, "Four parts · Eight marking operations", 29, muted, anchor="end")
    elements.append('<line x1="48" y1="886" x2="1232" y2="886" stroke="#d7e0e5"/>')
    text(48, 931, "Reversible direction", 32, weight="bold")
    for row, bit in enumerate([0, 1]):
        y = 982 + row * 58
        text(60, y, bit, 31, weight="bold")
        text(119, y, "A", 30)
        text(354, y, "B", 30, anchor="end")
        a, b = ((157, y - 10), (313, y - 10))
        if bit:
            a, b = b, a
        line(a, b, coral, 5.5, "mark_arrow")
    text(455, 931, "Example route", 32, weight="bold")
    text(455, 984, "Order", 30, weight="bold")
    text(455, 1042, "Dir.", 30, weight="bold")
    for visit, (operation, bit) in enumerate(zip(sequence, direction)):
        x = 629 + 77 * visit
        elements.append(
            f'<rect x="{x - 27}" y="950" width="54" height="47" rx="5" fill="#edf2f5"/>'
        )
        text(x, 984, int(operation) + 1, 31, weight="bold", anchor="middle")
        text(x, 1042, int(bit), 31, weight="bold", anchor="middle")
    text(48, 1092, "Synthetic layout · Illustrative coordinates and route", 27, muted)
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
