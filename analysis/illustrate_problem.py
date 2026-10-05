"""Draw a public, paired-endpoint marking-route schematic as an SVG."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from src.problem import Problem, validate_solution

# Independent illustrative coordinates, not a traced production dataset.
# Endpoints are in an arbitrary 100 x 64 sheet coordinate system.
ENDPOINTS = [
    (1, 3),
    (8, 3),
    (4, 23),
    (21, 23),
    (4, 42),
    (22, 42),
    (10, 61),
    (10, 44),
    (22, 43),
    (39, 61),
    (42, 60),
    (47, 58),
    (47, 57),
    (47, 53),
    (53, 57),
    (53, 53),
    (58, 60),
    (53, 58),
    (62, 61),
    (78, 43),
    (90, 61),
    (90, 44),
    (76, 42),
    (96, 42),
    (80, 23),
    (96, 23),
    (99, 5),
    (92, 5),
]
DIRECTIONS = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
SOLUTION = list(range(14)) + DIRECTIONS
LABELS = [
    (5, 7),
    (13, 27),
    (14, 38),
    (6, 54),
    (27, 54),
    (41, 63),
    (42, 54),
    (58, 54),
    (57, 63),
    (76, 54),
    (96, 54),
    (86, 38),
    (88, 27),
    (93, 10),
]
# Decorative nesting contours: no collision or nesting constraints are modeled.
CONTOURS = [
    "M 12,54 Q 12,59 17,59 L 25,59 Q 29,59 29,54 L 28,51 L 13,51 Z",
    "M 71,54 Q 71,59 76,59 L 84,59 Q 88,59 88,54 L 87,51 L 72,51 Z",
    "M 26,46 L 43,46 L 39,35 L 31,35 L 31,39 L 29,39 Q 26,34 26,39 Z",
    "M 57,46 L 74,46 L 74,39 Q 74,34 71,39 L 66,39 L 63,35 Z",
    "M 21,27 L 35,27 L 31,16 L 26,16 L 26,21 L 24,21 Q 21,16 21,21 Z",
    "M 64,27 L 79,27 L 79,21 Q 79,16 76,21 L 70,21 L 72,16 L 68,16 Z",
    "M 29,3 L 45,43 L 51,49 L 41,15 Z",
    "M 43,3 L 54,29 L 58,33 L 52,10 Z",
    "M 0,0 L 28,0 L 34,16 L 20,16 L 17,3 Z",
    "M 72,0 L 100,0 L 100,3 L 82,3 L 77,16 L 67,16 Z",
]


def schematic_problem():
    coords = np.asarray(ENDPOINTS, dtype=float)
    return Problem([0, 0], coords, coords[np.arange(len(coords)) ^ 1], "synthetic_schematic")


def render_schematic():
    problem = schematic_problem()
    sequence, direction = validate_solution(problem, SOLUTION)
    navy, muted, coral = "#173746", "#536774", "#cb4439"
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1070" '
        'viewBox="0 0 1440 1070" role="img" aria-labelledby="title desc">',
        '<title id="title">Marking path: visit order and reversible direction</title>',
        '<desc id="desc">Fourteen red marking segments on an illustrative nested sheet. '
        "Dashed navy arrows connect the origin to each entry, each exit to the next entry, "
        "and the final exit to the origin. White numbered discs identify marking operations. "
        "The route is manually selected and is not an optimization result.</desc>",
        '<defs><marker id="travel_arrow" markerWidth="12" markerHeight="12" refX="11" '
        'refY="6" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M1,1 L11,6 L1,11" fill="none" stroke="#173746" stroke-width="2"/>'
        '</marker><marker id="mark_arrow" markerWidth="12" markerHeight="12" refX="10" '
        'refY="6" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L12,6 L0,12 Z" fill="#cb4439"/></marker></defs>',
        '<rect width="1440" height="1070" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
    ]

    def text(x, y, value, size=23, color=navy, weight="normal", anchor="start"):
        elements.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
        )

    def line(a, b, color, width=3, arrow=None, dashed=False, halo=False):
        extra = f' marker-end="url(#{arrow})"' if arrow else ""
        if dashed:
            extra += ' stroke-dasharray="9 7"'
        coords = f'x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}"'
        if halo:
            elements.append(f'<line {coords} stroke="#fff" stroke-width="{width + 3}"/>')
        elements.append(
            f'<line {coords} stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"{extra}/>'
        )

    def point(p):
        return 65 + 9.2 * float(p[0]), 775 - 9.2 * float(p[1])

    text(48, 60, "Marking path optimization", 39, weight="bold")
    text(48, 101, "Fixed layout. Variable visit order and processing direction.", 25, muted)
    elements.append('<line x1="48" y1="126" x2="1392" y2="126" stroke="#d8dfe3"/>')
    text(48, 166, "A route through 14 marking operations", 27, weight="bold")
    elements.append(
        '<rect x="65" y="186.2" width="920" height="588.8" '
        'fill="#e1e4e6" stroke="#637680" stroke-width="2"/>'
    )
    for contour in CONTOURS:
        elements.append(
            f'<path d="{contour}" transform="translate(65 775) scale(9.2 -9.2)" '
            'fill="#fff" stroke="#637680" stroke-width="0.2" stroke-linejoin="round"/>'
        )
    for center in [(24, 8), (75, 8)]:
        x, y = point(center)
        elements.append(
            f'<circle cx="{x}" cy="{y}" r="10" fill="#fff" stroke="#637680" stroke-width="2"/>'
        )

    selected = 2 * sequence + direction
    entries, exits = problem.coords[selected], problem.coords_paired[selected]
    for a, b in zip(np.vstack([problem.origin, exits]), np.vstack([entries, problem.origin])):
        # Shorten the connector at its target so the marking entry remains visible.
        pa, pb = np.asarray(point(a)), np.asarray(point(b))
        distance = np.linalg.norm(pb - pa)
        end = pb - (pb - pa) * min(7 / distance, 0.25)
        line(pa, end, navy, 2.7, "travel_arrow", dashed=True, halo=True)

    for part in range(problem.num_parts):
        a, b = point(entries[part]), point(exits[part])
        line(a, b, coral, 4.2, "mark_arrow", halo=True)
        elements.append(f'<circle cx="{a[0]}" cy="{a[1]}" r="4.5" fill="{coral}"/>')
        lx, ly = point(LABELS[part])
        elements.append(
            f'<circle cx="{lx}" cy="{ly}" r="18" fill="#fff" stroke="#a9b7be" stroke-width="1.3"/>'
        )
        text(lx, ly + 7, part + 1, 22, navy, "bold", "middle")

    ox, oy = point(problem.origin)
    elements.append(f'<circle cx="{ox}" cy="{oy}" r="8" fill="{navy}"/>')
    text(65, 812, "O  ·  start / return", 23, weight="bold")
    text(985, 812, "White contours: illustrative part layout", 21, muted, anchor="end")

    elements.append('<line x1="1025" y1="160" x2="1025" y2="814" stroke="#d8dfe3"/>')
    text(1060, 174, "Read the route", 27, weight="bold")
    line((1062, 216), (1122, 216), coral, 4.2, "mark_arrow")
    text(1140, 224, "Marking stroke", 23)
    line((1062, 259), (1122, 259), navy, 2.7, "travel_arrow", dashed=True)
    text(1140, 267, "Travel between strokes", 22)
    elements.append('<circle cx="1079" cy="310" r="18" fill="#fff" stroke="#a9b7be"/>')
    text(1079, 318, "3", 22, weight="bold", anchor="middle")
    text(1111, 318, "Marking operation ID", 22)

    text(1060, 389, "Two endpoints, two directions", 24, weight="bold")
    text(1060, 427, "0", 24, weight="bold")
    text(1110, 427, "A", 23)
    text(1350, 427, "B", 23, anchor="end")
    line((1118, 450), (1337, 450), coral, 4.2, "mark_arrow")
    text(1060, 498, "1", 24, weight="bold")
    text(1110, 498, "A", 23)
    text(1350, 498, "B", 23, anchor="end")
    line((1337, 521), (1118, 521), coral, 4.2, "mark_arrow")
    text(1060, 570, "The selected endpoint is the entry;", 21, muted)
    text(1060, 601, "its paired endpoint is the exit.", 21, muted)

    text(1060, 665, "What the optimizer changes", 24, weight="bold")
    text(1060, 704, "Sequence + direction bits", 24)
    text(1060, 749, "Objective: dashed travel only", 23, weight="bold")
    text(1060, 780, "Marking length stays fixed.", 21, muted)

    elements.append('<line x1="48" y1="840" x2="1392" y2="840" stroke="#d8dfe3"/>')
    text(48, 883, "Example solution", 25, weight="bold")
    text(275, 883, "Each column is one visit: operation ID above, direction bit below.", 23, muted)
    text(48, 935, "Sequence", 22, weight="bold")
    text(48, 988, "Direction", 22, weight="bold")
    for rank, (part, bit) in enumerate(zip(sequence, direction)):
        x = 224 + 83 * rank
        elements.append(
            f'<rect x="{x - 28}" y="906" width="56" height="40" rx="5" fill="#edf1f3"/>'
        )
        text(x, 935, int(part) + 1, 24, weight="bold", anchor="middle")
        text(x, 988, int(bit), 24, weight="bold", anchor="middle")
        if rank < problem.num_parts - 1:
            text(x + 42, 935, "→", 23, muted, anchor="middle")
    text(
        48,
        1040,
        "Synthetic schematic · Manually selected route · No collision constraints",
        21,
        muted,
    )
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
