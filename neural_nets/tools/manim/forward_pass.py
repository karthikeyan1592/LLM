import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{xcolor}")
FWD = BLUE_C
NEURON_COLORS = [BLUE_C, GREEN_C, ORANGE]
SIZES = [3, 3, 2, 1]
XS = [-6.4, -4.8, -3.2, -1.6]
R = 0.36
PANEL_X, PANEL_W = 3.5, 6.3


def mt(s, size=34, color=WHITE, **kw):
    return MathTex(s, tex_template=TEX, font_size=size, color=color, **kw)


def bmat(rows):
    return r"\begin{bmatrix}" + r"\\".join(" & ".join(r) for r in rows) + r"\end{bmatrix}"


def vec(name, n):
    return bmat([[f"{name}_{i}"] for i in range(1, n + 1)])


def place(items, buff=0.28, y=-0.1):
    for m in items:
        if m.width > PANEL_W:
            m.scale_to_fit_width(PANEL_W)
    g = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=buff)
    if g.height > 6.3:
        g.scale_to_fit_height(6.3)
    g.move_to([PANEL_X, y, 0])
    return g


class Net:
    pass


def build_network():
    net = Net()
    net.circ, net.top, net.bot, net.biases = [], [], [], []
    parts = []
    for l, (x, n) in enumerate(zip(XS, SIZES)):
        ys = [(n - 1) / 2 * 1.35 - i * 1.35 for i in range(n)]
        cs, ts, bs, bias = [], [], [], []
        for i, y in enumerate(ys):
            c = np.array([x, y, 0])
            circle = Circle(radius=R, color=GREY_B, stroke_width=3).move_to(c)
            cs.append(circle)
            if l == 0:
                ts.append(mt(f"x_{i + 1}", 22).move_to(c))
                bs.append(None)
                parts += [circle, ts[-1]]
            else:
                top = mt(f"a_{i + 1}" if l < 3 else "a", 16).move_to(c + UP * 0.17)
                bot = mt(f"z_{i + 1}" if l < 3 else "y", 16).move_to(c + DOWN * 0.17)
                div = Line(c + LEFT * R, c + RIGHT * R, stroke_width=2, color=GREY_B)
                ts.append(top)
                bs.append(bot)
                b = mt(rf"+b_{i + 1}", 14, GREY_B).next_to(circle, UP, buff=0.04)
                bias.append(b)
                parts += [circle, div, top, bot, b]
        net.circ.append(cs)
        net.top.append(ts)
        net.bot.append(bs)
        net.biases.append(bias)
    net.edge = []
    edge_parts = []
    for l in range(3):
        d = {}
        for j in range(SIZES[l + 1]):
            for i in range(SIZES[l]):
                d[(j, i)] = Line(net.circ[l][i].get_right(), net.circ[l + 1][j].get_left(),
                                 stroke_width=1.5, color=GREY_D)
                edge_parts.append(d[(j, i)])
        net.edge.append(d)
    names = ["inputs", "layer 1\ntanh", "layer 2\ntanh", "layer 3\nsigmoid"]
    net.tags = VGroup(*[Text(t, font_size=17, color=GREY_B, line_spacing=0.6).move_to([x, -2.5, 0])
                        for x, t in zip(XS, names)])
    net.all = VGroup(*edge_parts, *parts, net.tags)
    return net


def flash(net, l, color=FWD, pairs=None):
    keys = pairs if pairs is not None else list(net.edge[l].keys())
    return [ShowPassingFlash(net.edge[l][k].copy().set_color(color).set_stroke(width=6), time_width=0.7)
            for k in keys]


def hl(net, l, pairs, color, width=5):
    return [net.edge[l][k].animate.set_color(color).set_stroke(width=width) for k in pairs]


def unhl(net, l, pairs):
    return [net.edge[l][k].animate.set_color(GREY_D).set_stroke(width=1.5) for k in pairs]


class ForwardPass(Scene):
    def construct(self):
        title = Text("Forward pass — formulas, layer by layer", font_size=32).to_edge(UP, buff=0.25)
        net = build_network()
        self.play(Write(title))
        self.play(FadeIn(net.all), run_time=1.5)

        # --- reading the picture -------------------------------------------------
        head = Text("Reading the picture", font_size=26, color=FWD)
        r1 = mt(r"\text{top half: } a_j = \text{(weighted sum of inputs)} + b_j", 26)
        r2 = mt(r"\text{bottom half: } z_j = h(a_j)", 26)
        r3 = Text("each grey line is one weight  w, a connection between two neurons", font_size=22)
        count = mt(r"\underbrace{9+6+2}_{17\ \text{weights}}\ +\ \underbrace{3+2+1}_{6\ \text{biases}}\ =\ 23\ \text{numbers to learn}", 26, YELLOW)
        p = place([head, r1, r2, r3, count])
        self.play(Write(head))
        self.play(Write(r1), Indicate(VGroup(*net.top[1]), color=YELLOW), run_time=1.6)
        self.play(Write(r2), Indicate(VGroup(*net.bot[1]), color=YELLOW), run_time=1.6)
        self.play(FadeIn(r3), *flash(net, 0, YELLOW), run_time=1.4)
        self.play(Write(count), run_time=1.8)
        self.wait(1.2)
        self.play(FadeOut(p))

        # --- numbering rule ------------------------------------------------------
        head = Text("The numbering rule", font_size=26, color=FWD)
        sym = mt(r"w_{23}^{(1)}", 64, tex_to_color_map={"2": ORANGE, "3": GREEN_C, "1": BLUE_C})
        l1 = Text("bracket (1): which layer the weight belongs to", font_size=21, color=BLUE_C)
        l2 = Text("first index 2: destination, the neuron it connects to", font_size=21, color=ORANGE)
        l3 = Text("second index 3: source, where it comes from (x₃)", font_size=21, color=GREEN_C)
        l4 = Text("destination first, source second: that is what makes\nrow j of W = the weights of neuron j",
                  font_size=20, color=GREY_B, line_spacing=0.8)
        p = place([head, sym, l1, l2, l3, l4])
        wire = net.edge[0][(1, 2)]
        wire_lab = mt(r"w^{(1)}_{23}", 20, YELLOW).move_to(wire.get_center() + UP * 0.22)
        self.play(Write(head), Write(sym), *hl(net, 0, [(1, 2)], YELLOW, 6), FadeIn(wire_lab), run_time=1.6)
        for m in (l1, l2, l3):
            self.play(FadeIn(m, shift=RIGHT * 0.2), run_time=0.9)
        self.play(FadeIn(l4))
        self.wait(1.5)
        self.play(FadeOut(p), FadeOut(wire_lab), *unhl(net, 0, [(1, 2)]))

        # --- layer 1 -------------------------------------------------------------
        head = Text("Layer 1: one equation per neuron", font_size=26, color=FWD)
        eqs = []
        for j in (1, 2, 3):
            terms = " + ".join(rf"w^{{(1)}}_{{{j}{i}}}x_{i}" for i in (1, 2, 3))
            eqs.append(mt(rf"a^{{(1)}}_{j} = {terms} + b^{{(1)}}_{j}", 27, NEURON_COLORS[j - 1]))
        pattern = Text("same shape every time; only the numbers differ", font_size=21, color=GREY_B)
        p = place([head, *eqs, pattern], buff=0.34)
        self.play(Write(head))
        for j in (0, 1, 2):
            pairs = [(j, i) for i in range(3)]
            self.play(*hl(net, 0, pairs, NEURON_COLORS[j]), Write(eqs[j]), run_time=1.5)
            self.wait(0.3)
        self.play(FadeIn(pattern))
        self.wait(1.2)
        self.play(FadeOut(VGroup(*eqs, pattern)), *unhl(net, 0, list(net.edge[0].keys())))

        w1 = bmat([[f"w_{{{j}{i}}}" for i in (1, 2, 3)] for j in (1, 2, 3)])
        big = mt(rf"{vec('a', 3)} = {w1}{vec('x', 3)} + {vec('b', 3)}", 28)
        short = mt(r"a^{(1)} = W^{(1)}x + b^{(1)}", 38, YELLOW)
        shape = mt(r"(3\times1) = (3\times3)(3\times1) + (3\times1)", 26, GREY_B)
        rowrule = Text("row j of W = the weights of neuron j", font_size=21, color=GREY_B)
        act = mt(r"z^{(1)} = h\!\left(a^{(1)}\right) = h\!\left(W^{(1)}x + b^{(1)}\right)", 32)
        tag = Text("a layer = multiply by a matrix, add a vector, bend with h", font_size=21, color=YELLOW)
        p2 = place([head, big, short, shape, rowrule, act, tag], buff=0.3)
        self.play(Write(big), run_time=2.0)
        self.play(Write(short), FadeIn(shape), run_time=1.4)
        self.play(FadeIn(rowrule))
        self.play(Write(act), *flash(net, 0), *[c.animate.set_color(FWD) for c in net.circ[1]], run_time=1.8)
        self.play(FadeIn(tag))
        self.wait(1.4)
        self.play(FadeOut(p2))

        # --- layer 2 -------------------------------------------------------------
        head = Text("Layer 2: same recipe, new inputs", font_size=26, color=FWD)
        note = Text("it reads the outputs of layer 1, the vector z⁽¹⁾, not x", font_size=21, color=GREY_B)
        eqs = []
        for j in (1, 2):
            terms = " + ".join(rf"w^{{(2)}}_{{{j}{i}}}z^{{(1)}}_{i}" for i in (1, 2, 3))
            eqs.append(mt(rf"a^{{(2)}}_{j} = {terms} + b^{{(2)}}_{j}", 27, NEURON_COLORS[j - 1]))
        short = mt(r"a^{(2)} = W^{(2)}z^{(1)} + b^{(2)},\quad z^{(2)} = h\!\left(a^{(2)}\right)", 32, YELLOW)
        w2 = bmat([[f"w_{{{j}{i}}}" for i in (1, 2, 3)] for j in (1, 2)])
        wshape = mt(rf"W^{{(2)}} = {w2}\quad (2\times3)", 28)
        rule = Text("W has one row per neuron in this layer and\none column per neuron in the previous layer",
                    font_size=21, color=YELLOW, line_spacing=0.8)
        inner = mt(r"(2\times3)(3\times1) = (2\times1)\ \checkmark\quad\text{the inner 3s count the same neurons}", 24, GREY_B)
        p = place([head, note, *eqs, short, wshape, rule, inner], buff=0.26)
        self.play(Write(head), FadeIn(note))
        for j in (0, 1):
            pairs = [(j, i) for i in range(3)]
            self.play(*hl(net, 1, pairs, NEURON_COLORS[j]), Write(eqs[j]), run_time=1.3)
        self.play(Write(short), run_time=1.4)
        self.play(Write(wshape), run_time=1.4)
        self.play(FadeIn(rule), FadeIn(inner), *flash(net, 1),
                  *[c.animate.set_color(FWD) for c in net.circ[2]])
        self.wait(1.4)
        self.play(FadeOut(p), *unhl(net, 1, list(net.edge[1].keys())))

        # --- layer 3 -------------------------------------------------------------
        head = Text("Layer 3: sigmoid output", font_size=26, color=FWD)
        a3 = mt(r"a^{(3)} = w^{(3)}_{11}z^{(2)}_1 + w^{(3)}_{12}z^{(2)}_2 + b^{(3)} = W^{(3)}z^{(2)} + b^{(3)}", 27)
        w3 = mt(r"W^{(3)} = \begin{bmatrix} w_{11} & w_{12}\end{bmatrix}\ (1\times2),\quad a^{(3)}\ \text{is one number}", 26, GREY_B)
        sig = mt(r"y = \sigma\!\left(a^{(3)}\right) = \frac{1}{1+e^{-a^{(3)}}}", 36, YELLOW)
        sn = Text("squeezes any number into (0, 1): readable as a probability", font_size=21, color=GREY_B)
        p = place([head, a3, w3, sig, sn], buff=0.32)
        self.play(Write(head), *flash(net, 2), run_time=1.0)
        self.play(Write(a3), run_time=1.6)
        self.play(FadeIn(w3))
        self.play(Write(sig), *[c.animate.set_color(FWD) for c in net.circ[3]], run_time=1.6)
        self.play(FadeIn(sn))
        self.wait(1.4)
        self.play(FadeOut(p))

        # --- everything in one formula ------------------------------------------
        head = Text("The whole network in one formula", font_size=26, color=FWD)
        nested = mt(r"y = \sigma\!\Big(W^{(3)}\,h^{(2)}\!\big(W^{(2)}\,h^{(1)}(W^{(1)}x + b^{(1)}) + b^{(2)}\big) + b^{(3)}\Big)", 30, YELLOW)
        l1 = mt(r"z^{(1)} = h^{(1)}\!\left(W^{(1)}x + b^{(1)}\right)\qquad (3\times1)", 26)
        l2 = mt(r"z^{(2)} = h^{(2)}\!\left(W^{(2)}z^{(1)} + b^{(2)}\right)\qquad (2\times1)", 26)
        l3 = mt(r"y = \sigma\!\left(W^{(3)}z^{(2)} + b^{(3)}\right)\qquad (1\times1)", 26)
        keep = Text("every a and z is kept: the backward pass will need them", font_size=21, color=GREY_B)
        p = place([head, nested, l1, l2, l3, keep], buff=0.32)
        self.play(Write(head))
        self.play(Write(nested), run_time=2.2)
        for m in (l1, l2, l3):
            self.play(Write(m), run_time=1.1)
        self.play(FadeIn(keep))
        self.wait(3.0)
