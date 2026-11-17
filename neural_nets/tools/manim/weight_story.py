import textwrap
import numpy as np
from manim import *
from matplotlib import colormaps

# Weight snapshots (epochs 0, 100, 200, 300) of the notebook run: seed 42, full-batch SGD, lr 0.1.
DATA_W = {'W1': {'0': [[0.44141000509262085, 0.4792099893093109, -0.13526000082492828], [0.5303599834442139, -0.1264999955892563, 0.11649999767541885], [-0.28108999133110046, 0.33906999230384827, 0.5089600086212158]], '100': [[0.33976998925209045, 0.5298900008201599, -0.37514999508857727], [0.5319399833679199, -0.12410999834537506, 0.10785999894142151], [-0.30344000458717346, 0.3512200117111206, 0.4387199878692627]], '200': [[0.1979299932718277, 0.5440300107002258, -0.7293800115585327], [0.5141599774360657, -0.12140999734401703, 0.1890600025653839], [-0.2902100086212158, 0.35308000445365906, 0.48249000310897827]], '300': [[0.07468000054359436, 0.4501599967479706, -0.9612500071525574], [0.48701000213623047, -0.09624999761581421, 0.2954300045967102], [-0.23806999623775482, 0.4077500104904175, 0.5938100218772888]]}, 'W2': {'0': [[0.4265500009059906, 0.07818999886512756, 0.27838999032974243], [-0.08151999861001968, 0.4450699985027313, 0.0853400006890297]], '100': [[0.5638300180435181, 0.006310000084340572, 0.06605999916791916], [-0.087459996342659, 0.44196999073028564, 0.09212999790906906]], '200': [[0.8255199790000916, -0.016860000789165497, -0.17107999324798584], [-0.16176000237464905, 0.4360800087451935, 0.17851999402046204]], '300': [[1.0072200298309326, -0.009469999931752682, -0.4045700132846832], [-0.25870999693870544, 0.4361099898815155, 0.30667001008987427]]}, 'W3': {'0': [[-0.3257899880409241, -0.08291999995708466]], '100': [[-0.4334999918937683, 0.07163000106811523]], '200': [[-0.79721999168396, 0.22421999275684357]], '300': [[-1.1987700462341309, 0.4336099922657013]]}}
W = {l: {int(ep): np.array(m) for ep, m in DATA_W[f"W{l}"].items()} for l in (1, 2, 3)}
EPOCHS = 300

# Numbers produced by the checks in the notebook's "story" cell for the same run.
CHECK = dict(
    base_acc=0.6906, shuffle_acc=[0.6981, 0.6885, 0.5030],
    corr_x3=0.929, corr_x1sq=-0.043, corr_x2sq=-0.018, corr_curve=-0.016, rule_x3_acc=0.7078,
    probe={300: dict(val=0.675, cx3=0.93, ccurve=-0.02), 1000: dict(val=0.762, cx3=0.80, ccurve=0.33),
           3000: dict(val=0.913, cx3=0.62, ccurve=0.65)},
)

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
CMAP = colormaps["RdBu_r"]
SEE_C, WHY_C, SO_C = BLUE_C, YELLOW, GREEN_C
POS_C, NEG_C = RED_C, BLUE_C


def mt(s, size=28, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def txt(s, size=20, color=WHITE, width=None):
    if width:
        s = textwrap.fill(s, width=width)
    return Text(s, font_size=size, color=color, line_spacing=0.9)


def heat_color(t):
    r, g, b, _ = CMAP((float(np.clip(t, -1, 1)) + 1) / 2)
    return rgb_to_color([r, g, b])


class Heat:
    pass


def heatmap(mat, vmax, cell=0.95, fs=18):
    h = Heat()
    rows, cols = mat.shape
    h.cells, h.labels = {}, {}
    parts = VGroup()
    for j in range(rows):
        for i in range(cols):
            t = mat[j, i] / vmax
            sq = Square(side_length=cell, stroke_width=1.5, stroke_color=GREY_D, fill_color=heat_color(t), fill_opacity=1)
            sq.move_to([i * cell, -j * cell, 0])
            lab = Text(f"{mat[j, i]:+.2f}", font_size=fs, color=BLACK if abs(t) < 0.55 else WHITE).move_to(sq)
            h.cells[(j, i)], h.labels[(j, i)] = sq, lab
            parts.add(sq, lab)
    parts.move_to(ORIGIN)
    h.group = parts
    return h


def sign_color(v):
    return POS_C if v > 0 else NEG_C


class WeightStory(Scene):
    PACE = 0.32

    def hold(self, text, minimum=1.5, extra=0.6):
        self.wait(max(minimum, self.PACE * len(text.split()) + extra))

    def say(self, s, color=WHITE, y=-2.75, width=86, size=21, minimum=2.0, x=0.0):
        t = txt(s, size, color, width).move_to([x, y, 0])
        self.play(FadeIn(t, shift=UP * 0.1), run_time=0.8)
        self.hold(t.text, minimum)
        return t

    # ------------------------------------------------------------------ intro
    def part_intro(self):
        title = Text("Reading the weight heatmaps", font_size=40).move_to([0, 3.0, 0])
        sub = Text("what training did to the 17 weights", font_size=26, color=GREY_B).move_to([0, 2.3, 0])
        self.play(Write(title), run_time=1.6)
        self.play(FadeIn(sub))
        self.wait(1.0)
        demo = heatmap(W[1][0], 0.6, cell=1.0, fs=19)
        demo.group.move_to([-2.6, -0.2, 0])
        rl = VGroup(*[Text(f"neuron {j + 1}", font_size=18, color=GREY_B).next_to(demo.cells[(j, 0)], LEFT, buff=0.2) for j in range(3)])
        cl = VGroup(*[Text(f"x{i + 1}", font_size=18, color=GREY_B).next_to(demo.cells[(0, i)], UP, buff=0.2) for i in range(3)])
        self.play(FadeIn(demo.group), FadeIn(rl), FadeIn(cl), run_time=1.4)
        items = [
            "Each layer is a table of weights.",
            "Row j is neuron j of this layer. Column i is the input coming into it.",
            "The cell is the weight connecting input i to neuron j.",
            "Red is positive, blue is negative, white is near zero.",
        ]
        right = VGroup(*[txt(s, 21, WHITE, 34) for s in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        right.move_to([0, 0, 0]).align_to(np.array([1.5, 0, 0]), LEFT).align_to(np.array([0, 1.3, 0]), UP)
        for k, t in enumerate(right):
            anims = [FadeIn(t, shift=RIGHT * 0.15)]
            if k == 1:
                anims.append(Create(SurroundingRectangle(VGroup(*[demo.cells[(1, i)] for i in range(3)]), color=YELLOW, buff=0.03, stroke_width=4)))
            if k == 2:
                anims.append(Indicate(demo.cells[(1, 2)], color=YELLOW, scale_factor=1.15))
            self.play(*anims, run_time=1.0)
            self.hold(t.text, 2.0)
        self.wait(1.0)
        note = txt("For every layer the plot shows three tables: before training (epoch 0), after (epoch 300), and the change.", 21,
                   GREY_B, 80).move_to([0, -3.0, 0])
        self.play(FadeIn(note))
        self.hold(note.text, 3.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ layers
    def part_layer(self, l, in_names, row_names, steps, rings, headline):
        W0, W3 = W[l][0], W[l][EPOCHS]
        D = W3 - W0
        vmax = max(np.abs(W0).max(), np.abs(W3).max(), np.abs(D).max()) + 1e-6
        head = Text(headline, font_size=30).move_to([0, 3.45, 0])
        self.play(FadeIn(head), run_time=0.8)
        xs = [-4.6, 0.0, 4.6]
        heats = [heatmap(m, vmax) for m in (W0, W3, D)]
        titles = ["epoch 0", f"epoch {EPOCHS}", "change (after − before)"]
        grids = VGroup()
        for hmap, x, t in zip(heats, xs, titles):
            hmap.group.move_to([x, 0.75, 0])
            tt = Text(t, font_size=20, color=YELLOW if "change" in t else WHITE).next_to(hmap.group, UP, buff=0.75)
            cl = VGroup(*[Text(n, font_size=15, color=GREY_B).next_to(hmap.cells[(0, i)], UP, buff=0.1) for i, n in enumerate(in_names)])
            grids.add(VGroup(hmap.group, tt, cl))
        rl = VGroup(*[Text(n, font_size=15, color=GREY_B).next_to(heats[0].cells[(j, 0)], LEFT, buff=0.12) for j, n in enumerate(row_names)])
        self.play(FadeIn(grids[0]), FadeIn(rl), run_time=1.2)
        self.wait(1.0)
        self.play(FadeIn(grids[1]), run_time=1.2)
        self.wait(1.0)
        self.play(FadeIn(grids[2]), run_time=1.2)
        self.wait(1.2)
        prev = None
        ring_objs = []
        for text, color, ring_ids in steps:
            t = txt(text, 21, color, 86).move_to([0, -2.75, 0])
            anims = [FadeIn(t, shift=UP * 0.1)]
            if prev is not None:
                anims.append(FadeOut(prev))
            for rid in ring_ids:
                j, i = rings[rid]
                for hmap in heats:
                    r = SurroundingRectangle(hmap.cells[(j, i)], color=YELLOW, buff=0.0, stroke_width=4)
                    ring_objs.append(r)
                    anims.append(Create(r))
            self.play(*anims, run_time=1.2)
            self.hold(t.text, 2.5)
            prev = t
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in [head, grids, rl, prev, *ring_objs]], run_time=1.0)

    # ------------------------------------------------------------------ the x3 path
    def part_path(self):
        head = Text("Follow the signs: the path from x3 to the output", font_size=32).move_to([0, 3.45, 0])
        self.play(FadeIn(head), run_time=0.8)
        xs, sizes = [-6.1, -4.5, -2.9, -1.3], [3, 3, 2, 1]
        circ = []
        for l, (x, n) in enumerate(zip(xs, sizes)):
            row = []
            for i in range(n):
                c = Circle(radius=0.34, color=GREY_B, stroke_width=3).move_to([x, (n - 1) / 2 * 1.25 - i * 1.25 + 0.3, 0])
                row.append(c)
            circ.append(row)
        labels = VGroup()
        for l in range(4):
            for i, c in enumerate(circ[l]):
                name = f"x{i + 1}" if l == 0 else ("out" if l == 3 else f"n{i + 1}")
                labels.add(Text(name, font_size=16).move_to(c))
        wires = {}
        for l in range(3):
            for j in range(sizes[l + 1]):
                for i in range(sizes[l]):
                    wires[(l, j, i)] = Line(circ[l][i].get_right(), circ[l + 1][j].get_left(), stroke_width=1.5, color=GREY_D)
        tags = VGroup(*[Text(t, font_size=15, color=GREY_B).move_to([x, -2.45, 0])
                        for x, t in zip(xs, ["inputs", "layer 1", "layer 2", "layer 3"])])
        net = VGroup(*wires.values(), *[c for r in circ for c in r], labels, tags)
        self.play(FadeIn(net), run_time=1.4)
        w1, w2, w3 = W[1][EPOCHS], W[2][EPOCHS], W[3][EPOCHS]
        path = [((0, 0, 2), w1[0, 2]), ((1, 0, 0), w2[0, 0]), ((2, 0, 0), w3[0, 0])]
        right = RIGHT_X = 1.5

        def panel(lines, color=WHITE):
            g = VGroup(*[txt(s, 21, c if isinstance(c, str) is False else color, 34) for s, c in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            g.move_to([0, 0, 0]).align_to(np.array([RIGHT_X, 0, 0]), LEFT).align_to(np.array([0, 2.6, 0]), UP)
            return g

        intro = panel([("The three weights that changed most form one path.", WHITE),
                       ("Start at input x3 and follow the weights.", WHITE)])
        self.play(FadeIn(intro[0], shift=RIGHT * 0.15)); self.hold(intro[0].text)
        self.play(FadeIn(intro[1], shift=RIGHT * 0.15)); self.hold(intro[1].text)
        wl = VGroup()
        arrows = []
        steps = [
            (circ[0][2], circ[1][0], path[0], "x3 → neuron 1 of layer 1", 0.0),
            (circ[1][0], circ[2][0], path[1], "that neuron → neuron 1 of layer 2", 0.0),
            (circ[2][0], circ[3][0], path[2], "that neuron → the output", 0.0),
        ]
        for (a, b, ((l, j, i), val), name, _), k in zip(steps, range(3)):
            col = sign_color(val)
            lab = Text(f"{val:+.2f}", font_size=17, color=col).move_to(wires[(l, j, i)].get_center() + UP * 0.28)
            self.play(wires[(l, j, i)].animate.set_color(col).set_stroke(width=7), FadeIn(lab), run_time=1.2)
            wl.add(lab)
            self.wait(1.0)
        self.play(FadeOut(intro))
        signs = panel([("The signs are  −,  +,  −.", WHITE), ("Two minuses make a plus.", WHITE)])
        eq = mt(r"(-)\times(+)\times(-) = (+)", 30, YELLOW).next_to(signs, DOWN, aligned_edge=LEFT, buff=0.35)
        self.play(FadeIn(signs[0], shift=RIGHT * 0.15)); self.hold(signs[0].text)
        self.play(FadeIn(signs[1], shift=RIGHT * 0.15), Write(eq), run_time=1.4); self.hold(signs[1].text)
        # signal arrows: x3 up -> n1 down -> n1' down -> output up
        dirs = [(circ[0][2], UP, "x3 goes up"), (circ[1][0], DOWN, "n1 goes down"), (circ[2][0], DOWN, "n1 goes down"), (circ[3][0], UP, "output goes up")]
        marks = VGroup()
        for c, d, _ in dirs:
            a = Arrow(c.get_top() + UP * 0.05 if d is UP else c.get_bottom() + DOWN * 0.05,
                      c.get_top() + UP * 0.55 if d is UP else c.get_bottom() + DOWN * 0.55,
                      buff=0, stroke_width=6, color=YELLOW, max_tip_length_to_length_ratio=0.5)
            marks.add(a)
        for m in marks:
            self.play(FadeIn(m, scale=0.8), run_time=0.8)
            self.wait(0.6)
        conc = txt("When x3 increases, the output increases: the network has a path that carries x3.", 21, SO_C, 34)
        conc.next_to(eq, DOWN, aligned_edge=LEFT, buff=0.35)
        self.play(FadeIn(conc, shift=RIGHT * 0.15)); self.hold(conc.text, 3.0)
        self.play(FadeOut(VGroup(signs, eq, conc)))
        # all six paths
        paths = {(j, k): w1[j, 2] * w2[k, j] * w3[0, k] for j in range(3) for k in range(2)}
        total = sum(paths.values())
        q = panel([("There are six paths from x3 to the output.", WHITE),
                   ("Multiply the three weights along each path:", WHITE)])
        self.play(FadeIn(q[0], shift=RIGHT * 0.15)); self.hold(q[0].text)
        self.play(FadeIn(q[1], shift=RIGHT * 0.15)); self.hold(q[1].text, 1.5)
        order = sorted(paths.items(), key=lambda kv: -kv[1])
        bars = VGroup()
        x0 = 3.1
        for r, ((j, k), v) in enumerate(order):
            y = 0.3 - 0.4 * r
            lab = Text(f"via n{j + 1} then n{k + 1}", font_size=15, color=GREY_B).move_to([x0 - 1.0, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=max(v * 1.8, 0.03), height=0.28, stroke_width=0, fill_color=POS_C, fill_opacity=0.9).move_to([x0 - 0.9, y, 0], aligned_edge=LEFT)
            val = Text(f"{v:+.2f}", font_size=15, color=POS_C).next_to(bar, RIGHT, buff=0.1)
            bars.add(VGroup(lab, bar, val))
        for b in bars:
            self.play(FadeIn(b, shift=RIGHT * 0.1), run_time=0.7)
        self.wait(1.0)
        best = order[0][1] / total
        c1 = txt(f"All six are positive: every path agrees. The top path carries about {best:.0%} of the total.", 21, SO_C, 34)
        c1.move_to([0, 0, 0]).align_to(np.array([RIGHT_X, 0, 0]), LEFT).align_to(np.array([0, -2.0, 0]), UP)
        self.play(FadeIn(c1, shift=RIGHT * 0.15)); self.hold(c1.text, 3.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ the test
    def bar_chart(self, center, title, cats, vals, colors, y0, y1, base, fmt, line=None):
        w, h = 5.0, 2.6
        ax = Axes(x_range=[0, len(cats), 1], y_range=[y0, y1, (y1 - y0) / 4], x_length=w, y_length=h, tips=False,
                  axis_config={"include_ticks": False, "stroke_width": 2, "color": GREY_B}).move_to(center)
        parts = VGroup(ax)
        zero_y = ax.c2p(0, base)[1]
        parts.add(Line([ax.c2p(0, 0)[0], zero_y, 0], [ax.c2p(len(cats), 0)[0], zero_y, 0], stroke_width=2, color=GREY_B))
        bw = w / len(cats) * 0.55
        for k, (c, v, col) in enumerate(zip(cats, vals, colors)):
            cx = ax.c2p(k + 0.5, 0)[0]
            top = ax.c2p(0, v)[1]
            height = abs(top - zero_y)
            r = Rectangle(width=bw, height=max(height, 0.02), stroke_width=0, fill_color=col, fill_opacity=0.95)
            r.move_to([cx, (top + zero_y) / 2, 0])
            parts.add(r)
            lab = Text(c, font_size=16, color=WHITE).move_to([cx, ax.c2p(0, y0)[1] - 0.25, 0])
            vl = Text(fmt(v), font_size=16, color=col).move_to([cx, top + (0.2 if v >= base else -0.2), 0])
            parts.add(lab, vl)
        if line:
            ly, name = line
            parts.add(DashedLine([ax.c2p(0, 0)[0], ax.c2p(0, ly)[1], 0], [ax.c2p(len(cats), 0)[0], ax.c2p(0, ly)[1], 0], color=GREY_B, stroke_width=2),
                      Text(name, font_size=14, color=GREY_B).move_to([ax.c2p(len(cats), 0)[0] + 0.45, ax.c2p(0, ly)[1], 0]))
        parts.add(Text(title, font_size=18, color=YELLOW).move_to([center[0], center[1] + h / 2 + 0.55, 0]))
        return parts

    def part_test(self):
        head = Text("Is it really only x3? Test it", font_size=32).move_to([0, 3.45, 0])
        self.play(FadeIn(head), run_time=0.8)
        c = CHECK
        intro = self.say("Two checks on the trained network, using the training data.", WHITE, y=2.6, minimum=2.0)
        A = self.bar_chart([-3.4, 0.2, 0], "how the output moves with each part of the rule",
                           ["x3", "x1²", "x2²", "x1²−x2²"], [c["corr_x3"], c["corr_x1sq"], c["corr_x2sq"], c["corr_curve"]],
                           [YELLOW, GREY_B, GREY_B, GREY_B], -0.25, 1.0, 0.0, lambda v: f"{v:+.2f}")
        self.play(FadeIn(A), run_time=1.4)
        n1 = self.say("Check 1: correlation of the network's output with each part of the rule.", WHITE, y=-2.3, width=40, size=20, minimum=2.0, x=-3.4)
        self.say("It follows x3 (+0.93). The squares x1² and x2² barely register.", SO_C, y=-3.2, width=40, size=20, minimum=2.5, x=-3.4)
        B = self.bar_chart([3.6, 0.2, 0], "accuracy when one input is shuffled",
                           ["none", "x1", "x2", "x3"], [c["base_acc"]] + c["shuffle_acc"],
                           [GREY_B, GREY_B, GREY_B, RED_C], 0.4, 0.8, 0.4, lambda v: f"{v:.2f}", line=(0.5, "chance"))
        self.play(FadeIn(B), run_time=1.4)
        self.hold("Check 2: scramble one input at a time and see how much accuracy drops.", 2.0)
        n2 = txt("Check 2: shuffle one input at a time.\nShuffling x1 or x2 changes nothing.\nShuffling x3 drops accuracy to chance.", 20, WHITE).move_to([3.6, -2.6, 0])
        self.play(FadeIn(n2))
        self.hold(n2.text, 3.5)
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ what that means
    def part_meaning(self):
        head = Text("What this means", font_size=32).move_to([0, 3.45, 0])
        self.play(FadeIn(head), run_time=0.8)
        rule = mt(r"\text{label} = \big[\,\underbrace{x_1^2 - x_2^2}_{\text{curved part}} \;+\; \underbrace{x_3}_{\text{straight part}} > 0\,\big]", 34).move_to([0, 2.2, 0])
        self.play(Write(rule), run_time=2.0)
        self.wait(1.0)
        box1 = VGroup(RoundedRectangle(width=5.7, height=1.7, corner_radius=0.15, color=RED_C, stroke_width=3),
                      txt("the straight part  x3\nLEARNED: the path we traced", 20, WHITE)).move_to([3.25, 0.2, 0])
        box2 = VGroup(RoundedRectangle(width=5.7, height=1.7, corner_radius=0.15, color=GREY_B, stroke_width=3),
                      txt("the curved part  x1² − x2²\nNOT YET: no weight builds it", 20, WHITE)).move_to([-3.25, 0.2, 0])
        self.play(FadeIn(box2), run_time=1.0)
        self.hold("the curved part not yet", 2.5)
        self.play(FadeIn(box1), run_time=1.0)
        self.hold("the straight part learned", 2.5)
        t1 = self.say("The simplest rule on the same labels, \"x3 > 0\", scores 70.8 %. The network at epoch 300 scores 69.1 %.", YELLOW,
                      y=-1.7, width=80, size=21, minimum=3.5)
        t2 = self.say("So the network has found the straight part and nothing more. That is what \"under-trained\" means here.", WHITE,
                      y=-2.6, width=80, size=21, minimum=3.5)
        t3 = self.say("Our reading: with small weights tanh is almost a straight line, so a straight-line feature is the first thing a network can learn. "
                      "Squares need the neurons to bend.", GREY_B, y=-3.5, width=84, size=18, minimum=4.0)
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ keep training
    def part_more(self):
        head = Text("What happens if we keep training?", font_size=32).move_to([0, 3.45, 0])
        self.play(FadeIn(head), run_time=0.8)
        sub = Text("same network, same learning rate, same starting weights", font_size=20, color=GREY_B).move_to([0, 2.95, 0])
        self.play(FadeIn(sub))
        P = CHECK["probe"]
        eps = [300, 1000, 3000]
        ax = Axes(x_range=[0, 2, 1], y_range=[-0.2, 1.0, 0.2], x_length=6.2, y_length=3.6, tips=False,
                  axis_config={"include_ticks": False, "stroke_width": 2, "color": GREY_B}).move_to([-2.9, -0.2, 0])
        xl = VGroup(*[Text(str(e), font_size=17, color=GREY_B).next_to(ax.c2p(k, -0.2), DOWN, buff=0.15) for k, e in enumerate(eps)],
                    Text("epoch", font_size=16, color=GREY_B).next_to(ax, DOWN, buff=0.55))
        yl = VGroup(*[Text(s, font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v, s in
                      [(0, "0"), (0.5, "0.5"), (1.0, "1.0")]])
        zero = Line(ax.c2p(0, 0), ax.c2p(2, 0), stroke_width=1, color=GREY_D)
        self.play(FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(zero), run_time=1.2)
        series = [("output follows x3", [P[e]["cx3"] for e in eps], RED_C),
                  ("output follows x1² − x2² (the curve)", [P[e]["ccurve"] for e in eps], GREEN_C),
                  ("validation accuracy", [P[e]["val"] for e in eps], WHITE)]
        leg = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.5, color=c, stroke_width=5), Text(n, font_size=16, color=c)).arrange(RIGHT, buff=0.15)
                       for n, _, c in series]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.3, -2.55, 0])
        self.play(FadeIn(leg))
        for (name, vals, col), pos in zip(series, ["x3", "curve", "acc"]):
            line = ax.plot_line_graph([0, 1, 2], vals, line_color=col, stroke_width=5, add_vertex_dots=True, vertex_dot_radius=0.07,
                                      vertex_dot_style={"stroke_width": 0, "fill_color": col})
            self.play(Create(line["line_graph"]), run_time=2.0)
            self.play(FadeIn(line["vertex_dots"]))
            dirs = {"x3": [UR, UP, DR], "curve": [DOWN, DOWN, UR], "acc": [DOWN, DOWN, UP]}[pos]
            labs = VGroup(*[Text(f"{v:+.2f}" if pos != "acc" else f"{v:.2f}", font_size=15, color=col).next_to(ax.c2p(k, v), dirs[k], buff=0.1)
                            for k, v in enumerate(vals)])
            self.play(FadeIn(labs))
            self.wait(1.0)
        pts = [txt("At epoch 300 the network follows x3 and not the curve: the picture we just read.", 20, WHITE, 36),
               txt("By epoch 1000 it has started on the curve. By epoch 3000 it follows the curve more than x3.", 20, WHITE, 36),
               txt("Validation accuracy goes from 0.675 to 0.913. (The check is in the notebook cell above.)", 20, SO_C, 36)]
        g = VGroup(*pts).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        g.move_to([0, 0, 0]).align_to(np.array([1.6, 0, 0]), LEFT).align_to(np.array([0, 1.9, 0]), UP)
        for t in g:
            self.play(FadeIn(t, shift=RIGHT * 0.15), run_time=0.9)
            self.hold(t.text, 2.5)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ verdict
    def part_verdict(self):
        title = Text("What the weight heatmaps tell us", font_size=36).move_to([0, 3.3, 0])
        self.play(FadeIn(title))
        points = [
            ("Few weights change much: 3 of 9 in layer 1, 3 of 6 in layer 2, both in layer 3.", SO_C),
            ("They form one path from x3 to the output, and the signs along it agree.", SO_C),
            ("The network has learned the straight part of the rule (+x3) and not yet the curved part.", YELLOW),
            ("More training starts on the curve: validation accuracy 0.675 → 0.913 by epoch 3000.", SO_C),
        ]
        items = []
        for s, c in points:
            items.append(VGroup(Dot(color=c, radius=0.09), txt(s, 23, WHITE, 72)).arrange(RIGHT, buff=0.3))
        g = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, 0.7, 0])
        for it in g:
            self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.9)
            self.hold(it[1].text, 2.5)
        v = Text("Healthy, but only stage one.", font_size=32, color=YELLOW).move_to([0, -2.2, 0])
        self.play(Write(v), run_time=1.4)
        self.wait(3.0)

    def construct(self):
        self.part_intro()
        self.part_layer(1, ["x1", "x2", "x3"], ["n1", "n2", "n3"], [
            ("Most of the nine weights barely change: the x2 column changes by less than 0.07.", WHITE, []),
            ("Neuron 1 ← x3 goes from −0.14 to −0.96, the biggest change of all.", YELLOW, [0]),
            ("Neuron 1 ← x1 falls from 0.44 to 0.08, and neuron 2 ← x3 rises from 0.12 to 0.30.", YELLOW, [1, 2]),
            ("These three weights carry 98 % of the change in this layer.", SO_C, []),
        ], rings=[(0, 2), (0, 0), (1, 2)], headline="Layer 1: three of nine weights change much")
        self.part_layer(2, ["z1", "z2", "z3"], ["n1", "n2"], [
            ("Again a few weights do the moving, and two of them change sign or size a lot.", WHITE, []),
            ("Neuron 1 ← z1 grows from 0.43 to 1.01.", YELLOW, [0]),
            ("Neuron 1 ← z3 falls from +0.28 to −0.41: it changes sign.", YELLOW, [1]),
            ("Three weights carry 96 % of the change in this layer.", SO_C, []),
        ], rings=[(0, 0), (0, 2)], headline="Layer 2: the same pattern")
        self.part_layer(3, ["z1", "z2"], ["out"], [
            ("The output neuron has only two weights, and both move a lot.", WHITE, []),
            ("The weight on z1 goes from −0.33 to −1.20.", YELLOW, [0]),
            ("The weight on z2 goes from −0.08 to +0.43: it changes sign.", YELLOW, [1]),
            ("Relative to its starting size this layer changed the most: its weights grew about three times over.", SO_C, []),
        ], rings=[(0, 0), (0, 1)], headline="Layer 3: the output layer changes most")
        self.part_path()
        self.part_test()
        self.part_meaning()
        self.part_more()
        self.part_verdict()
