"""Render a synthetic nesting-style schematic of the paired-endpoint route model."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from src.problem import Problem, validate_solution

# Outlines are visual context. The optimizer uses the paired endpoints.
OUTLINES = [
    [(8, 8), (34, 8), (34, 26), (27, 26), (27, 20), (8, 20)],
    [(58, 8), (93, 8), (93, 24), (79, 24), (74, 18), (58, 18)],
    [(61, 31), (93, 31), (93, 49), (86, 49), (86, 43), (61, 43)],
    [(62, 54), (90, 54), (95, 62), (90, 67), (62, 67)],
    [(9, 53), (44, 53), (44, 67), (9, 67), (9, 61), (17, 61), (17, 58), (9, 58)],
    [(9, 30), (41, 30), (41, 47), (30, 47), (30, 41), (9, 41)],
]
ENDPOINTS = [
    (11, 12),
    (29, 12),
    (83, 11),
    (88, 20),
    (88, 33),
    (88, 46),
    (66, 61),
    (88, 61),
    (13, 63),
    (40, 63),
    (14, 34),
    (27, 38),
]
LABELS = [(21, 17), (69, 16), (73, 36), (76, 56), (29, 56), (35, 44)]
SOLUTION = [0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 1, 1]


def render_schematic():
    coords = np.asarray(ENDPOINTS, dtype=float)
    problem = Problem([0, 0], coords, coords[np.arange(len(coords)) ^ 1], "synthetic_schematic")
    sequence, direction = validate_solution(problem, SOLUTION)
    visits = {int(part): (rank + 1, int(direction[rank])) for rank, part in enumerate(sequence)}
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="810" viewBox="0 0 1280 810" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">Marking segments and a route through paired endpoints</title>',
        '<desc id="desc">The same synthetic layout before and after selecting a route. '
        "Coral lines are fixed marking segments. Dashed blue arrows connect the origin, "
        "selected entries and exits. Part outlines are illustrative, not collision constraints.</desc>",
        '<defs><marker id="travel_arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8" fill="#244e70"/>'
        '</marker><marker id="mark_arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" '
        'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10" fill="#d45e47"/>'
        "</marker></defs>",
        '<rect width="1280" height="810" rx="20" fill="#f7f9fb"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
    ]

    def text(x, y, value, size=20, color="#17394b", weight="normal", anchor="start"):
        elements.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
        )

    text(32, 54, "Marking segments and travel route", 34, weight="bold")
    text(
        32,
        90,
        "Visit order and direction change the travel between fixed marking segments.",
        21,
        color="#586d7b",
    )

    for panel_x, routed in [(32, False), (664, True)]:
        elements.append(
            f'<rect x="{panel_x}" y="124" width="584" height="534" rx="14" '
            'fill="#ffffff" stroke="#d8e1e7"/>'
        )
        text(
            panel_x + 24,
            164,
            "Sequence and direction" if routed else "Fixed layout",
            27,
            weight="bold",
        )
        text(
            panel_x + 24,
            198,
            "Connect each exit to the next entry"
            if routed
            else "Each segment has two reversible endpoints",
            19,
            color="#586d7b",
        )

        def point(p):
            return (panel_x + 52 + 4.8 * float(p[0]), 590 - 4.8 * float(p[1]))

        def line(a, b, color, width, marker=None, dashed=False):
            ax, ay = point(a)
            bx, by = point(b)
            extra = f' marker-end="url(#{marker})"' if marker else ""
            if dashed:
                extra += ' stroke-dasharray="7 6"'
            elements.append(
                f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" '
                f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"{extra}/>'
            )

        sx, sy = point((0, 70))
        elements.append(
            f'<rect x="{sx}" y="{sy}" width="480" height="336" '
            'fill="#e9eef1" stroke="#80909c" stroke-width="1.5"/>'
        )
        for outline in OUTLINES:
            points = " ".join(f"{point(p)[0]:.2f},{point(p)[1]:.2f}" for p in outline)
            elements.append(
                f'<polygon points="{points}" fill="#cbd5dc" stroke="#667987" '
                'stroke-width="1.6" stroke-linejoin="round"/>'
            )

        if routed:
            selected = 2 * sequence + direction
            entries, exits = problem.coords[selected], problem.coords_paired[selected]
            for a, b in zip(
                np.vstack([problem.origin, exits]), np.vstack([entries, problem.origin])
            ):
                line(a, b, "#244e70", 2.6, "travel_arrow", dashed=True)

        for i in range(problem.num_parts):
            a, b = problem.coords[2 * i], problem.coords[2 * i + 1]
            rank, bit = visits[i]
            if routed and bit == 1:
                a, b = b, a
            line(a, b, "#d45e47", 4.2)
            if routed:
                line(a + (b - a) * 0.3, a + (b - a) * 0.68, "#d45e47", 4.2, "mark_arrow")
            for p in [a, b]:
                px, py = point(p)
                elements.append(
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.6" '
                    'fill="#ffffff" stroke="#d45e47" stroke-width="2.2"/>'
                )
            lx, ly = point(LABELS[i])
            if routed:
                elements.append(
                    f'<rect x="{lx - 40:.2f}" y="{ly - 15:.2f}" width="80" height="28" '
                    'rx="7" fill="#ffffff" stroke="#c4d1db"/>'
                )
                text(lx, ly + 6, f"{rank} · M{i + 1}", 18, weight="bold", anchor="middle")
            else:
                text(lx, ly + 6, f"M{i + 1}", 20, weight="bold", anchor="middle")

        if not routed:
            for label, endpoint in [("A", ENDPOINTS[0]), ("B", ENDPOINTS[1])]:
                px, py = point(endpoint)
                text(px, py + 24, label, 16, color="#a54a3a", anchor="middle")
        ox, oy = point(problem.origin)
        elements.append(f'<circle cx="{ox}" cy="{oy}" r="9" fill="#244e70"/>')
        text(ox + 16, oy + 26, "Origin", 18, color="#244e70")

    text(32, 703, "Visit order", 20, weight="bold")
    text(32, 751, "Directions", 20, weight="bold")
    for i, bit in enumerate(direction):
        x = 205 + 105 * i
        elements.append(f'<rect x="{x}" y="681" width="63" height="31" rx="7" fill="#e5edf3"/>')
        text(x + 31.5, 704, f"M{sequence[i] + 1}", 20, weight="bold", anchor="middle")
        if i < problem.num_parts - 1:
            text(x + 84, 704, "→", 24, color="#667987", anchor="middle")
        text(x + 31.5, 751, bit, 22, weight="bold", anchor="middle")
    text(32, 790, "Direction 0: A → B    ·    Direction 1: B → A", 18, color="#586d7b")
    elements.append('<line x1="915" y1="693" x2="960" y2="693" stroke="#d45e47" stroke-width="4"/>')
    text(975, 700, "Marking (fixed)", 20)
    elements.append(
        '<line x1="915" y1="735" x2="960" y2="735" stroke="#244e70" stroke-width="2.6" '
        'stroke-dasharray="7 6" marker-end="url(#travel_arrow)"/>'
    )
    text(975, 742, "Travel (objective)", 20)
    text(1248, 790, "Synthetic layout · Illustrative route", 17, color="#586d7b", anchor="end")
    elements.extend(["</g>", "</svg>"])
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/problem.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_schematic())
    print(args.output)


if __name__ == "__main__":
    main()
