"""Draw a readable synthetic marking route on shipbuilding-style plate parts."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from src.problem import Problem, validate_solution

# Independently designed example geometry in arbitrary sheet units.
# One reversible marking stroke represents each operation; shapes are visual context.
PARTS = [
    {
        "kind": "bracket",
        "outline": "M4,5 L32,5 Q28,9 25,13 L10,27 L4,27 L4,9 Q8,9 8,5 Z",
        "holes": ["M15,14 A3,3 0 1 0 9,14 A3,3 0 1 0 15,14 Z"],
        "endpoints": [(11, 9), (23, 9)],
        "label": (9, 21),
    },
    {
        "kind": "web_plate",
        "outline": "M4,31 L10,31 L10,34 Q12,36 14,34 L14,31 L23,31 L23,34 "
        "Q25,36 27,34 L27,31 L36,31 L34,48 L9,48 L4,43 Z",
        "holes": ["M15,39 L26,39 A3,3 0 0 1 26,45 L15,45 A3,3 0 0 1 15,39 Z"],
        "endpoints": [(32, 34), (32, 45)],
        "label": (8, 39),
    },
    {
        "kind": "tapered_web_strip",
        "outline": "M4,53 L49,53 L52,56 L52,62 L7,62 L4,59 Z",
        "holes": [],
        "endpoints": [(12, 56), (44, 56)],
        "label": (8, 59),
    },
    {
        "kind": "web_plate",
        "outline": "M55,47 L59,42 L64,42 L64,45 Q66,47 68,45 L68,42 "
        "L77,42 L77,45 Q79,47 81,45 L81,42 L90,42 L90,62 L59,62 L55,56 Z",
        "holes": ["M67,51 L80,51 A3,3 0 0 1 80,57 L67,57 A3,3 0 0 1 67,51 Z"],
        "endpoints": [(60, 49), (86, 49)],
        "label": (59.5, 57),
    },
    {
        "kind": "bracket",
        "outline": "M94,62 L116,62 L116,43 L110,43 L95,56 Q94,58 94,62 Z",
        "holes": ["M112,55 A3,3 0 1 0 106,55 A3,3 0 1 0 112,55 Z"],
        "endpoints": [(98, 60), (114, 60)],
        "label": (108, 49),
    },
    {
        "kind": "bracket",
        "outline": "M88,38 L116,38 L116,18 L110,18 L95,31 Q88,34 88,38 Z",
        "holes": ["M112,30 A3,3 0 1 0 106,30 A3,3 0 1 0 112,30 Z"],
        "endpoints": [(95, 35), (112, 35)],
        "label": (113, 23),
    },
    {
        "kind": "tapered_web_plate",
        "outline": "M39,26 L80,26 L80,38 L69,38 L44,34 L39,30 Z",
        "holes": ["M52,30 L66,30 A1.5,1.5 0 0 1 66,33 L52,33 A1.5,1.5 0 0 1 52,30 Z"],
        "endpoints": [(45, 28), (74, 28)],
        "label": (75, 34.5),
    },
    {
        "kind": "floor_plate",
        "outline": "M38,4 L43,4 L43,7 Q45,9 47,7 L47,4 L58,4 L58,7 "
        "Q60,9 62,7 L62,4 L73,4 L73,7 Q75,9 77,7 L77,4 L83,4 L83,22 L38,22 Z",
        "holes": ["M53,14 L68,14 A2.5,2.5 0 0 1 68,19 L53,19 A2.5,2.5 0 0 1 53,14 Z"],
        "endpoints": [(41, 11), (80, 11)],
        "label": (43, 18),
    },
]
ENDPOINTS = [p for part in PARTS for p in part["endpoints"]]
SOLUTION = list(range(len(PARTS))) + [0, 0, 0, 0, 0, 1, 1, 1]


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
        '<title id="title">Marking route on shipbuilding-style parts</title>',
        '<desc id="desc">Eight independently designed synthetic plate parts: brackets, '
        "web plates, tapered strips and a floor plate, with rounded openings and edge cutouts. "
        "Red arrows represent reversible marking strokes inside each part. Dashed navy "
        "arrows connect the origin, selected entries and exits, and return to the origin. "
        "Operation IDs are numbered from one. This manually selected route illustrates "
        "the paired-endpoint model; part outlines are not collision constraints.</desc>",
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

    def line(a, b, color, width=3.4, arrow=None, dashed=False, halo=False):
        extra = f' marker-end="url(#{arrow})"' if arrow else ""
        if dashed:
            extra += ' stroke-dasharray="12 9"'
        coords = f'x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}"'
        if halo:
            elements.append(f'<line {coords} stroke="#fff" stroke-width="{width + 3}"/>')
        elements.append(
            f'<line {coords} stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"{extra}/>'
        )

    def point(p):
        return 60 + 9.6 * float(p[0]), 815 - 9.6 * float(p[1])

    text(48, 61, "Marking path optimization", 44, weight="bold")
    text(48, 109, "Choose the visit order and direction on a fixed plate layout.", 30, muted)
    line((51, 164), (119, 164), coral, 5, "mark_arrow")
    text(137, 175, "Marking stroke", 31, weight="bold")
    line((510, 164), (578, 164), navy, 3.4, "travel_arrow", dashed=True)
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
    for a, b in zip(np.vstack([problem.origin, exits]), np.vstack([entries, problem.origin])):
        pa, pb = np.asarray(point(a)), np.asarray(point(b))
        distance = np.linalg.norm(pb - pa)
        end = pb - (pb - pa) * min(9 / distance, 0.25)
        line(pa, end, navy, 3.4, "travel_arrow", dashed=True, halo=True)
    for visit, part_id in enumerate(sequence):
        a, b = point(entries[visit]), point(exits[visit])
        line(a, b, coral, 5, "mark_arrow", halo=True)
        elements.append(f'<circle cx="{a[0]}" cy="{a[1]}" r="5" fill="{coral}"/>')
        lx, ly = point(PARTS[int(part_id)]["label"])
        elements.append(
            f'<circle cx="{lx}" cy="{ly}" r="23" fill="#fff" stroke="#8aa0ad" stroke-width="1.5"/>'
        )
        text(lx, ly + 11, int(part_id) + 1, 31, weight="bold", anchor="middle")

    ox, oy = point(problem.origin)
    elements.append(f'<circle cx="{ox}" cy="{oy}" r="9" fill="{navy}"/>')
    text(60, 859, "O  ·  start / return", 30, weight="bold")
    text(1212, 859, "Brackets · Web plates · Floor plate", 29, muted, anchor="end")
    elements.append('<line x1="48" y1="886" x2="1232" y2="886" stroke="#d7e0e5"/>')
    text(48, 931, "Reversible direction", 32, weight="bold")
    for row, bit in enumerate([0, 1]):
        y = 982 + row * 58
        text(60, y, bit, 31, weight="bold")
        text(119, y, "A", 30)
        text(354, y, "B", 30, anchor="end")
        start, end = ((157, y - 10), (313, y - 10))
        if bit:
            start, end = end, start
        line(start, end, coral, 5, "mark_arrow")

    text(455, 931, "Example solution", 32, weight="bold")
    text(455, 984, "Order", 30, weight="bold")
    text(455, 1042, "Dir.", 30, weight="bold")
    for visit, (part_id, bit) in enumerate(zip(sequence, direction)):
        x = 629 + 77 * visit
        elements.append(
            f'<rect x="{x - 27}" y="950" width="54" height="47" rx="5" fill="#edf2f5"/>'
        )
        text(x, 984, int(part_id) + 1, 31, weight="bold", anchor="middle")
        text(x, 1042, int(bit), 31, weight="bold", anchor="middle")
        if visit < problem.num_parts - 1:
            text(x + 39, 984, "→", 28, muted, anchor="middle")
    text(
        48, 1092, "Synthetic geometry · Illustrative route · Marking length stays fixed", 27, muted
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
