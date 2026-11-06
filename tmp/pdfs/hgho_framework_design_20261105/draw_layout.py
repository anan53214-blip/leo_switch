from pathlib import Path
from html import escape


OUT = Path(r'D:\program\LEO_switch\output\figures\hgho_framework_design_20261105')
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1800, 1330
INK = '#26313D'
LINE = '#485461'
GREEN, GREEN_EDGE = '#E1EFD9', '#7E9F68'
BLUE, BLUE_EDGE = '#D9E7FB', '#6B8EBF'
YELLOW, YELLOW_EDGE = '#FFF1CC', '#B79B4C'
PURPLE, PURPLE_EDGE = '#E7DFFC', '#9D84B5'
GRAY, GRAY_EDGE = '#F2F3F5', '#8B929C'
ORANGE = '#EC7C30'
bg, edges, nodes, labels = [], [], [], []


def text(x, y, s, size=27, anchor='middle', bold=False, color=INK, raw=False):
    value = s if raw else escape(s)
    weight = '700' if bold else '400'
    labels.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}">{value}</text>')


def lines(x, cy, values, size=27, gap=31, bold=False, color=INK):
    start = cy - (len(values) - 1) * gap / 2 + size * .34
    for i, value in enumerate(values):
        text(x, start + i * gap, value, size=size, bold=bold, color=color)


def rect(x, y, w, h, fill='white', stroke=GRAY_EDGE, radius=10, dash=False, background=False, width=2.2):
    d = ' stroke-dasharray="9 7"' if dash else ''
    value = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>'
    (bg if background else nodes).append(value)


def box(x, y, w, h, values, fill, stroke, size=27, bold=False):
    rect(x, y, w, h, fill, stroke)
    lines(x+w/2, y+h/2, values if isinstance(values, list) else [values], size=size, bold=bold)


def edge(points, color=LINE, dash=False, arrow=True, width=2.6):
    d = 'M ' + ' L '.join(f'{x},{y}' for x, y in points)
    marker = ' marker-end="url(#arrow-gray)"' if arrow else ''
    if color == ORANGE:
        marker = ' marker-end="url(#arrow-orange)"' if arrow else ''
    dash_attr = ' stroke-dasharray="8 6"' if dash else ''
    edges.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash_attr}{marker}/>')


def graph_circle(x, y, s):
    nodes.append(f'<circle cx="{x}" cy="{y}" r="17" fill="{GREEN}" stroke="{GREEN_EDGE}" stroke-width="2.5"/>')
    text(x, y+7, s, size=21)


def graph_square(x, y, s):
    rect(x-19, y-19, 38, 38, BLUE, BLUE_EDGE, radius=2)
    text(x, y+7, s, size=21)


def save_scene(filename, viewbox, title, description):
    x, y, width, height = viewbox
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{x} {y} {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title>
<desc id="desc">{escape(description)}</desc>
<defs>
<marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker>
<marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker>
</defs>
<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="white"/>
<g font-family="Times New Roman, Times, serif">
{''.join(bg)}
{''.join(edges)}
{''.join(nodes)}
{''.join(labels)}
</g>
</svg>'''
    target = OUT / filename
    target.write_text(svg, encoding='utf-8')
    print(target)
    bg.clear()
    edges.clear()
    nodes.clear()
    labels.clear()


# Three independent figures, each with its own canvas and SVG document.
rect(30, 25, 840, 645, 'white', GREEN_EDGE, radius=0, background=True, width=1.4)
text(52, 68, 'HAN encoder: contention representation', size=32, anchor='start', bold=True)
text(842, 68, 'Frozen', size=25, anchor='end', color=GRAY_EDGE)

# (a) A small typed graph and one repeated HAN layer.
text(150, 125, 'Typed graph', size=29, bold=True)
text(150, 158, 'Gₜ', size=29)
edge([(100, 230), (200, 230)], color=GRAY_EDGE, arrow=False)
edge([(90, 340), (100, 230)], color=GREEN_EDGE, arrow=False)
edge([(205, 340), (200, 230)], color=GREEN_EDGE, arrow=False)
edge([(145, 410), (100, 230)], color=PURPLE_EDGE, dash=True, arrow=False)
edge([(145, 410), (200, 230)], color=PURPLE_EDGE, dash=True, arrow=False)
edge([(90, 340), (205, 340)], color=GRAY_EDGE, dash=True, arrow=False)
graph_square(100, 230, 'S')
graph_square(200, 230, 'S')
graph_circle(90, 340, 'U')
graph_circle(205, 340, 'U')
graph_circle(145, 410, 'U')
text(150, 462, 'Node / edge', size=24)
text(150, 490, 'features', size=24)

rect(288, 100, 550, 416, '#FBFCFA', GREEN_EDGE, radius=12, dash=True, background=True)
text(563, 132, 'HAN layer  × 2', size=29, bold=True)
box(312, 155, 260, 88, ['U–S–U  (serving)', 'Established contention'], GREEN, GREEN_EDGE, size=25)
box(312, 263, 260, 88, ['U–S–U  (visibility)', 'Potential contention'], GREEN, GREEN_EDGE, size=25)
box(312, 371, 260, 88, ['U–U; S–U–S; S–S–S', 'Other relational context'], '#F4F8F0', GREEN_EDGE, size=23)
box(624, 173, 184, 108, ['Per-path', 'node attention α'], GREEN, GREEN_EDGE, size=25)
box(624, 345, 184, 126, ['Type-wise', 'semantic', 'attention β'], GREEN, GREEN_EDGE, size=26)
edge([(232, 308), (280, 308), (280, 199), (312, 199)])
edge([(280, 308), (312, 308)])
edge([(280, 308), (280, 415), (312, 415)])
edge([(572, 199), (594, 199), (594, 227), (624, 227)])
edge([(572, 307), (594, 307), (594, 227)], arrow=False)
edge([(572, 415), (594, 415), (594, 307)], arrow=False)
edge([(716, 281), (716, 345)])
box(608, 542, 216, 86, ['User embedding', 'zᵢ(t)'], GREEN, GREEN_EDGE, size=26, bold=True)
edge([(716, 471), (716, 542)])
text(73, 555, 'User and satellite representations', size=24, anchor='start')
text(73, 585, 'are propagated across layers.', size=24, anchor='start')
text(73, 631, 'Serving: solid     Visibility: dashed', size=22, anchor='start', color=LINE)
save_scene('han_encoder_draft.svg', (20, 15, 860, 665), 'HAN encoder layout draft', 'A standalone HAN module figure: typed graph, established and potential contention paths, per-path node attention, type-wise semantic attention, two layers, and user embeddings. The encoder is frozen.')

# (b) Every candidate receives its own fraction. A shared Q-network
# evaluates (fused observation, one-hot index, its fraction).
rect(900, 25, 870, 645, 'white', PURPLE_EDGE, radius=0, background=True, width=1.4)
text(925, 68, 'PDQN controller: parameterized joint control', size=32, anchor='start', bold=True)
box(930, 150, 140, 82, ['Fused obs.', 'õᵢ(t)'], GRAY, GRAY_EDGE, size=26)
box(1120, 118, 260, 108, ['Parameter branches μψ', 'One fraction per action'], BLUE, BLUE_EDGE, size=26)
box(1450, 133, 250, 78, ['ρ̂ᵢ,₀  …  ρ̂ᵢ,K'], BLUE, BLUE_EDGE, size=30)
box(930, 344, 140, 83, ['Index j', '(one-hot)'], GRAY, GRAY_EDGE, size=26)
box(1120, 316, 260, 109, ['Shared Q-network Qθ', 'Input: (õᵢ, j, ρ̂ᵢ,j)'], YELLOW, YELLOW_EDGE, size=26)
box(1450, 332, 250, 78, ['qᵢ,₀  …  qᵢ,K'], YELLOW, YELLOW_EDGE, size=30)
box(930, 504, 140, 78, ['Mask mᵢ'], GRAY, GRAY_EDGE, size=26)
box(1120, 494, 260, 100, ['Masked argmax', '+ matching fraction'], PURPLE, PURPLE_EDGE, size=27)
box(1450, 508, 250, 76, ['aᵢ = (dᵢ*, ρ̂ᵢ)'], '#FCEDDF', ORANGE, size=30, bold=True)
edge([(1070, 191), (1093, 191), (1093, 172), (1120, 172)])
edge([(1093, 191), (1093, 340), (1120, 340)])
edge([(1380, 172), (1450, 172)])
edge([(1575, 211), (1575, 271), (1250, 271), (1250, 316)])
text(1430, 258, 'Pair (j, ρ̂ᵢ,j)', size=25)
edge([(1070, 386), (1120, 386)])
edge([(1380, 371), (1450, 371)])
edge([(1575, 410), (1575, 448), (1200, 448), (1200, 494)])
edge([(1700, 172), (1736, 172), (1736, 472), (1320, 472), (1320, 494)])
edge([(1070, 543), (1120, 543)])
edge([(1380, 546), (1450, 546)], color=ORANGE)
text(935, 633, 'Shared parameters across users i = 1, …, N', size=26, anchor='start')
save_scene('pdqn_controller_draft.svg', (890, 15, 890, 665), 'PDQN controller layout draft', 'A standalone PDQN module figure: fused observations enter both the parameter branches and a shared Q-network. Candidate index-fraction pairs are evaluated, masked argmax selects an index, and the corresponding fraction is gathered. Parameters are shared across users.')

# (c) Compressed modules; bypasses retain raw observations and masks.
rect(30, 700, 1740, 605, 'white', '#C8CED5', radius=0, background=True, width=1.4)
text(52, 744, 'HGHO: overall interaction and learning flow', size=32, anchor='start', bold=True)
box(65, 860, 220, 125, ['LEO-MEC', 'environment'], GRAY, GRAY_EDGE, size=31, bold=True)
box(350, 860, 220, 90, ['Graph construction', 'Gₜ'], GRAY, GRAY_EDGE, size=27)
box(625, 860, 170, 90, ['HAN encoder', 'Frozen'], GREEN, GREEN_EDGE, size=25, bold=True)
box(850, 860, 210, 90, ['Concat', 'õᵢ = [oᵢ ∥ zᵢ]'], PURPLE, PURPLE_EDGE, size=28)
box(1115, 860, 210, 90, ['PDQN controller', 'Shared θ, ψ'], PURPLE, PURPLE_EDGE, size=25, bold=True)
box(1390, 860, 300, 90, ['Joint requests Aₜ', 'Handover + fraction'], '#FCEDDF', ORANGE, size=29, bold=True)
edge([(285, 905), (350, 905)])
edge([(570, 905), (625, 905)])
edge([(795, 905), (850, 905)])
edge([(1060, 905), (1115, 905)])
edge([(1325, 905), (1390, 905)])
text(821, 886, 'zᵢ', size=25)
edge([(1690, 905), (1731, 905), (1731, 802), (175, 802), (175, 860)])
text(950, 790, 'Execute joint requests and advance the environment', size=26)
edge([(175, 985), (175, 1020), (955, 1020), (955, 950)])
text(610, 1008, 'Raw observations oᵢ: user, task, candidates', size=26)
edge([(285, 945), (310, 945), (310, 1065), (1220, 1065), (1220, 950)])
text(930, 1053, 'Feasible-action mask mᵢ', size=25)

# Off-policy training has its own lane; no update arrow enters HAN.
rect(350, 1120, 1360, 167, '#FAFBFC', GRAY_EDGE, radius=10, dash=True, background=True)
text(372, 1155, 'Training only', size=26, anchor='start', bold=True, color=LINE)
box(400, 1180, 360, 70, ['Encoded replay D'], GRAY, GRAY_EDGE, size=29)
box(850, 1180, 450, 70, ['PDQN learner: L_Q and L_μ'], GRAY, GRAY_EDGE, size=29)
box(1410, 1180, 270, 70, ['Target networks'], GRAY, GRAY_EDGE, size=28)
edge([(175, 1020), (175, 1215), (400, 1215)], dash=True, color=GRAY_EDGE)
text(260, 1127, 'Collect', size=25, color=LINE)
text(260, 1158, 'transitions', size=25, color=LINE)
edge([(760, 1215), (850, 1215)], dash=True, color=GRAY_EDGE)
text(806, 1201, 'sample', size=23, color=LINE)
edge([(1300, 1215), (1410, 1215)], dash=True, color=GRAY_EDGE)
text(1355, 1200, 'copy', size=23, color=LINE)
edge([(1545, 1250), (1545, 1269), (1075, 1269), (1075, 1250)], dash=True, color=GRAY_EDGE)
text(1355, 1260, 'TD targets', size=21, color=LINE)
edge([(1180, 1180), (1180, 1096), (1295, 1096), (1295, 950)], dash=True, color=GRAY_EDGE)
text(1430, 1105, 'Update θ and ψ only', size=25, color=LINE)

save_scene('hgho_overall_draft.svg', (20, 690, 1760, 625), 'HGHO overall framework layout draft', 'A standalone HGHO overall framework figure. HAN encoder and PDQN controller are abbreviated modules whose internals appear in two separate figures. Raw observations bypass HAN and are fused with its user embeddings; masks enter PDQN selection; encoded replay and target networks support PDQN-only learning.')
