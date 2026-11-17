import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{xcolor}")
BWD, UPD = RED_C, YELLOW
ROUTE1, ROUTE2 = ORANGE, TEAL_C
SIZES = [3, 3, 2, 1]
XS = [-6.4, -4.8, -3.2, -1.6]
R = 0.36
PANEL_X, PANEL_W = 3.5, 6.3


def hexc(c):
    return c.to_hex().lstrip("#")


def mt(s, size=34, color=WHITE, **kw):
    return MathTex(s, tex_template=TEX, font_size=size, color=color, **kw)


def code(s, size=24, color=WHITE):
    return Tex(r"\texttt{" + s + "}", tex_template=TEX, font_size=size, color=color)


def bmat(rows):
    return r"\begin{bmatrix}" + r"\\".join(" & ".join(r) for r in rows) + r"\end{bmatrix}"


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
    net.circ, net.top, net.bot = [], [], []
    parts = []
    for l, (x, n) in enumerate(zip(XS, SIZES)):
        ys = [(n - 1) / 2 * 1.35 - i * 1.35 for i in range(n)]
        cs, ts, bs = [], [], []
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
                parts += [circle, div, top, bot]
        net.circ.append(cs)
        net.top.append(ts)
        net.bot.append(bs)
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
    out_c = net.circ[3][0].get_center()
    net.E = Square(side_length=0.5, color=GREY_B, stroke_width=3).move_to([-0.2, out_c[1], 0])
    net.Elab = mt("E", 22).move_to(net.E)
    net.Elink = Line(net.circ[3][0].get_right(), net.E.get_left(), stroke_width=1.5, color=GREY_D)
    names = ["inputs", "layer 1\ntanh", "layer 2\ntanh", "layer 3\nsigmoid"]
    net.tags = VGroup(*[Text(t, font_size=17, color=GREY_B, line_spacing=0.6).move_to([x, -2.5, 0])
                        for x, t in zip(XS, names)])
    net.all = VGroup(*edge_parts, net.Elink, *parts, net.E, net.Elab, net.tags)
    return net


def back_flash(net, l):
    return [ShowPassingFlash(Line(e.get_end(), e.get_start()).set_color(BWD).set_stroke(width=6), time_width=0.7)
            for e in net.edge[l - 1].values()]


def hl(net, l, pairs, color, width=6):
    return [net.edge[l][k].animate.set_color(color).set_stroke(width=width) for k in pairs]


def delta_tags(net, l):
    if l == 3:
        return VGroup(mt(r"\delta^{(3)}", 18, BWD).next_to(net.circ[3][0], UP, buff=0.05))
    return VGroup(*[mt(rf"\delta_{i + 1}", 18, BWD).next_to(c, UP, buff=0.05) for i, c in enumerate(net.circ[l])])


def chain_diagram(names, slopes, color=BWD):
    nodes = VGroup(*[VGroup(Circle(radius=0.3, color=GREY_B, stroke_width=3), mt(n, 26)) for n in names])
    nodes.arrange(RIGHT, buff=1.25)
    arrows, labels = VGroup(), VGroup()
    for i, s in enumerate(slopes):
        a = Arrow(nodes[i].get_right(), nodes[i + 1].get_left(), buff=0.05, stroke_width=3, color=color,
                  max_tip_length_to_length_ratio=0.25)
        arrows.add(a)
        labels.add(mt(s, 20, color).next_to(a, UP, buff=0.05))
    return VGroup(nodes, arrows, labels)


def routes_diagram():
    def node(name, pos):
        return VGroup(Circle(radius=0.3, color=GREY_B, stroke_width=3).move_to(pos), mt(name, 26).move_to(pos))
    u, p, q, e = node("u", [-2.4, 0, 0]), node("p", [0, 0.9, 0]), node("q", [0, -0.9, 0]), node("E", [2.4, 0, 0])
    specs = [(u, p, r"\frac{\partial p}{\partial u}", UL, ROUTE1), (u, q, r"\frac{\partial q}{\partial u}", DL, ROUTE2),
             (p, e, r"\frac{\partial E}{\partial p}", UR, ROUTE1), (q, e, r"\frac{\partial E}{\partial q}", DR, ROUTE2)]
    out = VGroup(u, p, q, e)
    for a, b, s, d, col in specs:
        arr = Arrow(a.get_center(), b.get_center(), buff=0.32, stroke_width=3, color=col, max_tip_length_to_length_ratio=0.2)
        out.add(arr, mt(s, 20, col).next_to(arr.get_center(), d, buff=0.12))
    return out


class BackwardPass(Scene):
    def construct(self):
        title = Text("Backpropagation — computing every gradient", font_size=32).to_edge(UP, buff=0.25)
        net = build_network()
        self.play(Write(title))
        self.play(FadeIn(net.all), run_time=1.5)
        self.play(*[n.animate.set_color(BLUE_C) for c in net.circ[1:] for n in c], run_time=0.5)

        # --- goal -----------------------------------------------------------------
        head = Text("The goal", font_size=26, color=BWD)
        g1 = mt(r"E = \tfrac12\,(y-t)^2", 34)
        g1b = Text("(our notebook uses BCE instead; Move A shows the difference)", font_size=20, color=GREY_B)
        g2 = Text("how much does E change if we nudge each of the 23 numbers?", font_size=21)
        g3 = mt(r"\frac{\partial E}{\partial w}\ \text{for every weight},\qquad \frac{\partial E}{\partial b}\ \text{for every bias}", 28, YELLOW)
        p = place([head, g1, g1b, g2, g3], buff=0.3)
        self.play(Write(head), Write(g1), FadeIn(g1b), run_time=1.6)
        self.play(FadeIn(g2))
        self.play(Write(g3), run_time=1.4)
        self.wait(0.8)
        self.play(FadeOut(p))

        # --- three tools ---------------------------------------------------------
        head = Text("Tool 1 — slope", font_size=26, color=BWD)
        s1 = mt(r"\frac{\partial E}{\partial w} > 0:\ \text{raising } w \text{ raises } E", 28)
        s2 = mt(r"\frac{\partial E}{\partial w} < 0:\ \text{raising } w \text{ lowers } E", 28)
        s3 = mt(r"\frac{\partial E}{\partial w} \approx 0:\ E \text{ hardly depends on } w", 28)
        s4 = Text("training moves every weight opposite to its slope", font_size=21, color=YELLOW)
        p = place([head, s1, s2, s3, s4], buff=0.32)
        self.play(Write(head), Write(s1), Write(s2), Write(s3), run_time=2.4)
        self.play(FadeIn(s4))
        self.wait(0.8)
        self.play(FadeOut(p))

        head = Text("Tool 2 — chain rule: slopes multiply", font_size=26, color=BWD)
        dia = chain_diagram(["w", "a", "z", "E"], [r"\frac{\partial a}{\partial w}", r"\frac{\partial z}{\partial a}", r"\frac{\partial E}{\partial z}"])
        f = mt(r"\frac{\partial E}{\partial w}=\frac{\partial E}{\partial z}\cdot\frac{\partial z}{\partial a}\cdot\frac{\partial a}{\partial w}", 34, YELLOW)
        n = Text("computed from the output end backwards", font_size=21, color=GREY_B)
        p = place([head, dia, f, n], buff=0.4)
        self.play(Write(head), FadeIn(dia), run_time=1.4)
        self.play(Write(f), run_time=1.6)
        self.play(FadeIn(n))
        self.wait(0.8)
        self.play(FadeOut(p))

        head = Text("Tool 3 — one value, two paths: add", font_size=26, color=BWD)
        dia = routes_diagram()
        f = mt(r"\frac{\partial E}{\partial u} = {\color[HTML]{" + hexc(ROUTE1) + r"}\frac{\partial E}{\partial p}\frac{\partial p}{\partial u}}"
               r" + {\color[HTML]{" + hexc(ROUTE2) + r"}\frac{\partial E}{\partial q}\frac{\partial q}{\partial u}}", 30)
        n = Text("multiply along each path, then add the paths", font_size=21, color=GREY_B)
        p = place([head, dia, f, n], buff=0.34)
        self.play(Write(head), FadeIn(dia), run_time=1.4)
        self.play(Write(f), run_time=1.6)
        self.play(FadeIn(n))
        self.wait(0.8)
        self.play(FadeOut(p))

        # --- the path from one weight to the error --------------------------------
        head = Text("The path from one weight to the error", font_size=26, color=BWD)
        path = mt(r"w^{(2)}_{jk}\to a^{(2)}_j \to z^{(2)}_j \to a^{(3)} \to y \to E", 30)
        prod = mt(r"\frac{\partial E}{\partial w^{(2)}_{jk}} &= \frac{\partial E}{\partial y}\cdot\frac{\partial y}{\partial a^{(3)}}\cdot\frac{\partial a^{(3)}}{\partial z^{(2)}_j}"
                  r" \\ &\quad\cdot\frac{\partial z^{(2)}_j}{\partial a^{(2)}_j}\cdot\frac{\partial a^{(2)}_j}{\partial w^{(2)}_{jk}}", 26)
        note = Text("five links, one per stage; shown for j = 1, k = 1", font_size=21, color=GREY_B)
        p = place([head, path, prod, note], buff=0.4)
        self.play(Write(head), Write(path), run_time=1.8)
        self.play(Write(prod), run_time=2.2)
        self.play(FadeIn(note))
        self.wait(0.8)
        self.play(FadeOut(p))

        stage_defs = [
            ("Stage 1 — E depends on y", "E", [
                (r"E = \tfrac12\,(y-t)^2",
                 "t is a fixed target, so E depends on y only", False),
                (r"\frac{\partial E}{\partial y} = \tfrac12\cdot 2\,(y-t)\cdot 1",
                 "power rule: the slope of (·)² is 2·(·),\nand the inner slope ∂(y − t)/∂y is 1", False),
                (r"\frac{\partial E}{\partial y} = y - t",
                 "the ½ was put there to cancel the 2.\ny below the target: negative slope,\nso raising y lowers E", True),
            ]),
            ("Stage 2 — y depends on a⁽³⁾ (sigmoid)", "y", [
                (r"y = \sigma(a) = \frac{1}{1+e^{-a}} = \left(1+e^{-a}\right)^{-1}",
                 "the output neuron squeezes its sum a\nthrough the sigmoid", False),
                (r"\frac{dy}{da} = -\left(1+e^{-a}\right)^{-2}\cdot\left(-e^{-a}\right) = \frac{e^{-a}}{\left(1+e^{-a}\right)^2}",
                 "chain rule on (1 + e⁻ᵃ)⁻¹:\nouter slope × inner slope", False),
                (r"= \underbrace{\frac{1}{1+e^{-a}}}_{y}\cdot\underbrace{\frac{e^{-a}}{1+e^{-a}}}_{1-y}",
                 "split the fraction in two. The first factor\nis y itself; the second is 1 − y, because\n1 − y = e⁻ᵃ / (1 + e⁻ᵃ)", False),
                (r"\frac{\partial y}{\partial a^{(3)}} = y\,(1-y)",
                 "y is known from the forward pass: one\nmultiplication. Steepest at y = 0.5 (0.25),\nflat near 0 or 1", True),
            ]),
            ("Stage 3 — a⁽³⁾ depends on z⁽²⁾ (the weight)", "w3", [
                (r"a^{(3)} = w^{(3)}_{11}\,z^{(2)}_1 + w^{(3)}_{12}\,z^{(2)}_2 + b^{(3)}",
                 "the output neuron's weighted sum", False),
                (r"\frac{\partial a^{(3)}}{\partial z^{(2)}_1} = w^{(3)}_{11}\cdot 1 + 0 + 0",
                 "treat z₂ and b as constants: only the term\nthat contains z₁ changes, by w₁₁ per unit z₁", False),
                (r"\frac{\partial a^{(3)}}{\partial z^{(2)}_j} = w^{(3)}_{1j}",
                 "δ is propagated backward by\nmultiplying with the weight", True),
            ]),
            ("Stage 4 — z⁽²⁾ depends on a⁽²⁾ (tanh)", "z2", [
                (r"z=\tanh(a)=\frac{d}{s},\qquad s=e^{a}+e^{-a},\quad d=e^{a}-e^{-a}",
                 "tanh written with exponentials.\nNotice d′ = s and s′ = d", False),
                (r"\frac{dz}{da} = \frac{d'\,s - d\,s'}{s^{2}} = \frac{s\cdot s - d\cdot d}{s^{2}}",
                 "quotient rule:\n(top′·bottom − top·bottom′) / bottom²", False),
                (r"= \frac{s^2}{s^2} - \frac{d^2}{s^2} = 1 - \left(\frac{d}{s}\right)^{2}",
                 "split the numerator over s²;\nd / s is exactly z", False),
                (r"\frac{\partial z}{\partial a} = 1 - z^{2}",
                 "no extra work: z was stored by the forward\npass. A saturated neuron (z near ±1)\nhas slope near 0", True),
            ]),
            ("Stage 5 — a⁽²⁾ depends on its weight", "w2", [
                (r"a^{(2)}_j = w^{(2)}_{j1}z^{(1)}_1 + w^{(2)}_{j2}z^{(1)}_2 + w^{(2)}_{j3}z^{(1)}_3 + b^{(2)}_j",
                 "the layer-2 neuron's weighted sum", False),
                (r"\frac{\partial a^{(2)}_j}{\partial w^{(2)}_{jk}} = z^{(1)}_k,\qquad \frac{\partial a^{(2)}_j}{\partial b^{(2)}_j}=1",
                 "same trick: only one term contains w_jk,\nand it multiplies z_k. A bias behaves like\na weight whose input is always 1", True),
            ]),
        ]
        for name, what, lines in stage_defs:
            head = Text(name, font_size=24, color=BWD)
            forms = [mt(tex, 36 if final else 28, YELLOW if final else WHITE) for tex, _, final in lines]
            fg = place([head, *forms], buff=0.26, y=0.7)
            cap_y = fg.get_bottom()[1] - 0.65
            caps = [Text(cap, font_size=20, color=GREY_B, line_spacing=0.8) for _, cap, _ in lines]
            for c in caps:
                if c.width > PANEL_W:
                    c.scale_to_fit_width(PANEL_W)
                c.move_to([PANEL_X, cap_y, 0])
            if what == "E":
                anims = [net.E.animate.set_color(BWD), net.Elink.animate.set_color(BWD).set_stroke(width=6)]
            elif what == "y":
                anims = [net.circ[3][0].animate.set_color(BWD)]
            elif what == "w3":
                anims = hl(net, 2, [(0, 0)], BWD)
            elif what == "z2":
                anims = [net.circ[2][0].animate.set_color(BWD)]
            else:
                anims = hl(net, 1, [(0, 0)], BWD)
            self.play(Write(head), *anims, run_time=1.4)
            prev = None
            for f, c in zip(forms, caps):
                self.play(Write(f), run_time=1.7)
                if prev is None:
                    self.play(FadeIn(c))
                else:
                    self.play(FadeOut(prev), FadeIn(c))
                prev = c
                self.wait(1.6)
            self.wait(0.6)
            self.play(FadeOut(VGroup(fg, prev)))

        head = Text("Multiply the five stages", font_size=26, color=BWD)
        full = mt(r"\frac{\partial E}{\partial w^{(2)}_{jk}} = \underbrace{(y-t)\,y(1-y)}_{\delta^{(3)}}\ \cdot\ w^{(3)}_{1j}\bigl(1-{z^{(2)}_j}^{2}\bigr)\ \cdot\ z^{(1)}_k", 28)
        d3 = mt(r"\delta^{(3)} = (y-t)\,y(1-y)", 30, YELLOW)
        d2 = mt(r"\delta^{(2)}_j = \delta^{(3)}\,w^{(3)}_{1j}\bigl(1-{z^{(2)}_j}^{2}\bigr)", 30, YELLOW)
        gw = mt(r"\frac{\partial E}{\partial w^{(2)}_{jk}} = \delta^{(2)}_j\, z^{(1)}_k", 30, YELLOW)
        p = place([head, full, d3, d2, gw], buff=0.3)
        self.play(Write(head), Write(full), run_time=2.2)
        for m in (d3, d2, gw):
            self.play(Write(m), run_time=1.2)
        self.wait(1.0)
        self.play(FadeOut(p))

        # --- routes ---------------------------------------------------------------
        self.play(net.E.animate.set_color(GREY_B), net.Elink.animate.set_color(GREY_D).set_stroke(width=1.5),
                  *[c.animate.set_color(BLUE_C) for c in (net.circ[3][0], net.circ[2][0])],
                  *[net.edge[2][(0, 0)].animate.set_color(GREY_D).set_stroke(width=1.5),
                    net.edge[1][(0, 0)].animate.set_color(GREY_D).set_stroke(width=1.5)], run_time=0.8)
        head = Text("Stage 6 — one layer back: two paths", font_size=24, color=BWD)
        why = Text("z₁⁽¹⁾ is an input to both layer-2 sums,\nso it reaches E by two paths", font_size=20, color=GREY_B, line_spacing=0.8)
        route = mt(r"\frac{\partial E}{\partial z^{(1)}_1} = {\color[HTML]{" + hexc(ROUTE1) + r"}\delta^{(2)}_1 w^{(2)}_{11}}"
                   r" + {\color[HTML]{" + hexc(ROUTE2) + r"}\delta^{(2)}_2 w^{(2)}_{21}}", 32)
        gen = mt(r"\frac{\partial E}{\partial z^{(1)}_k} = \sum_j \delta^{(2)}_j\, w^{(2)}_{jk}", 32, YELLOW)
        warn = Text("forget one path and the sign can come out wrong", font_size=20, color=RED_C)
        p = place([head, why, route, gen, warn], buff=0.34)
        self.play(Write(head), FadeIn(why),
                  net.circ[1][0].animate.set_color(YELLOW),
                  *hl(net, 1, [(0, 0)], ROUTE1), *hl(net, 1, [(1, 0)], ROUTE2), run_time=1.6)
        self.play(Write(route), run_time=2.0)
        self.play(Write(gen), run_time=1.4)
        self.play(FadeIn(warn))
        self.wait(1.0)
        self.play(FadeOut(p))

        head = Text("Stage 7 — layer 1: sum and weights", font_size=24, color=BWD)
        d1 = mt(r"\delta^{(1)}_k = \Big(\sum_j \delta^{(2)}_j\, w^{(2)}_{jk}\Big)\bigl(1-{z^{(1)}_k}^{2}\bigr)", 30, YELLOW)
        g1w = mt(r"\frac{\partial E}{\partial w^{(1)}_{km}} = \delta^{(1)}_k\, x_m", 32, YELLOW)
        s = Text("the layer-1 weights multiply the inputs x", font_size=20, color=GREY_B)
        p = place([head, d1, g1w, s], buff=0.4)
        self.play(Write(head), Write(d1), run_time=2.0)
        self.play(Write(g1w), FadeIn(s), run_time=1.4)
        self.wait(1.0)
        self.play(FadeOut(p),
                  *[net.edge[1][k].animate.set_color(GREY_D).set_stroke(width=1.5) for k in ((0, 0), (1, 0))],
                  net.circ[1][0].animate.set_color(BLUE_C))

        # --- one rule for every layer --------------------------------------------
        head = Text("One rule for every layer", font_size=26, color=BWD)
        ma = mt(r"\textbf{A}\ \ \delta^{(L)} = \frac{\partial E}{\partial y}\cdot h'\!\left(a^{(L)}\right)", 28)
        ma1 = mt(r"\text{MSE + sigmoid: } (y-t)\,y(1-y)", 22, GREY_B)
        ma2 = mt(r"\text{BCE + sigmoid (our notebook): } y-t", 22, YELLOW)
        mb = mt(r"\textbf{B}\ \ \frac{\partial E}{\partial w^{(l)}_{kj}} = \delta^{(l)}_k\, z^{(l-1)}_j,\quad \frac{\partial E}{\partial b^{(l)}_k} = \delta^{(l)}_k", 26)
        mc = mt(r"\textbf{C}\ \ \delta^{(l)}_j = \Big(\sum_k w^{(l+1)}_{kj}\,\delta^{(l+1)}_k\Big)\, h'\!\left(a^{(l)}_j\right)", 26)
        p = place([head, ma, ma1, ma2, mb, mc], buff=0.3)
        self.play(Write(head))
        self.play(Write(ma), run_time=1.5)
        self.play(FadeIn(ma1), FadeIn(ma2))
        t3 = delta_tags(net, 3)
        self.play(FadeIn(t3), net.circ[3][0].animate.set_color(BWD))
        self.play(Write(mb), *back_flash(net, 3), run_time=1.5)
        self.play(Write(mc), *back_flash(net, 2), run_time=1.5)
        t2 = delta_tags(net, 2)
        self.play(FadeIn(t2), *[c.animate.set_color(BWD) for c in net.circ[2]])
        self.play(*back_flash(net, 1), run_time=1.0)
        t1 = delta_tags(net, 1)
        self.play(FadeIn(t1), *[c.animate.set_color(BWD) for c in net.circ[1]])
        self.wait(1.0)
        self.play(FadeOut(p))

        # --- matrix form ----------------------------------------------------------
        head = Text("Matrix form: a whole layer at a time", font_size=26, color=BWD)
        mm = mt(r"\delta^{(l)} = \Big({W^{(l+1)}}^{\top}\delta^{(l+1)}\Big)\odot h'\!\left(a^{(l)}\right)", 30, YELLOW)
        c1 = r"\color[HTML]{" + hexc(ROUTE1) + "}"
        w = r"\begin{bmatrix} {" + c1 + r" w_{11}} & w_{12} & w_{13}\\ {" + c1 + r" w_{21}} & w_{22} & w_{23}\end{bmatrix}"
        wt = r"\begin{bmatrix} {" + c1 + r" w_{11}} & {" + c1 + r" w_{21}}\\ w_{12} & w_{22}\\ w_{13} & w_{23}\end{bmatrix}"
        wl = mt(r"W^{(2)} = " + w + r"\quad(2\times3)", 26)
        wtl = mt(r"\left(W^{(2)}\right)^{\top} = " + wt + r"\quad(3\times2)", 26)
        note = Text("column 1 of W (the weights leaving neuron 1) becomes row 1 of Wᵀ:\nthat is the sum over paths", font_size=20,
                    color=GREY_B, line_spacing=0.8)
        fw = Text("forward uses W · backward uses Wᵀ", font_size=21, color=YELLOW)
        p = place([head, mm, wl, wtl, note, fw], buff=0.28)
        self.play(Write(head))
        self.play(Write(mm), run_time=1.8)
        self.play(Write(wl), run_time=1.4)
        self.play(Write(wtl), run_time=1.4)
        self.play(FadeIn(note), FadeIn(fw))
        self.wait(1.2)
        self.play(FadeOut(p))

        head = Text("Gradients: one outer product per layer", font_size=26, color=BWD)
        og = mt(r"\frac{\partial E}{\partial W^{(l)}} = \delta^{(l)}\left(z^{(l-1)}\right)^{\top},\qquad \frac{\partial E}{\partial b^{(l)}} = \delta^{(l)}", 28, YELLOW)
        entries = bmat([[rf"\delta_{i}z_{j}" for j in (1, 2, 3)] for i in (1, 2)])
        ex = mt(r"\delta^{(2)}\left(z^{(1)}\right)^{\top} = " + entries, 28)
        sh = mt(r"(2\times1)(1\times3) = (2\times3)\ \text{— the same shape as } W^{(2)}", 24, GREY_B)
        nn = Text("weight (k, j): gradient = δₖ × zⱼ, the error signal of neuron k times the input zⱼ", font_size=20, color=GREY_B)
        p = place([head, og, ex, sh, nn], buff=0.34)
        self.play(Write(head), Write(og), run_time=1.8)
        self.play(Write(ex), run_time=1.6)
        self.play(FadeIn(sh), FadeIn(nn))
        self.wait(1.2)
        self.play(FadeOut(p))

        # --- the six lines --------------------------------------------------------
        head = Text("The whole backward sweep in six lines", font_size=26, color=BWD)
        lines = [
            code(r"delta3 = (y - t) * y * (1 - y)"),
            code(r"gradW3 = delta3 @ z2.T"),
            code(r"delta2 = (W3.T @ delta3) * (1 - z2**2)"),
            code(r"gradW2 = delta2 @ z1.T"),
            code(r"delta1 = (W2.T @ delta2) * (1 - z1**2)"),
            code(r"gradW1 = delta1 @ x.T"),
        ]
        bce = Text("with BCE the first line is just  delta3 = y - t", font_size=20, color=YELLOW)
        p = place([head, *lines, bce], buff=0.2)
        self.play(Write(head))
        for m in lines:
            self.play(FadeIn(m, shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeIn(bce))
        self.wait(1.2)
        self.play(FadeOut(p))

        # --- update ---------------------------------------------------------------
        head = Text("Update — the one step that changes the weights", font_size=26, color=UPD)
        uw = mt(r"W^{(l)} \leftarrow W^{(l)} - \eta\,\frac{\partial E}{\partial W^{(l)}}", 36, UPD)
        ub = mt(r"b^{(l)} \leftarrow b^{(l)} - \eta\,\delta^{(l)}", 36, UPD)
        un = Text("η = LR: training is many such steps", font_size=21, color=GREY_B)
        p = place([head, uw, ub, un], buff=0.4)
        widths = [3.5, 1.0, 4.5, 2.0, 0.8, 3.0, 1.5, 4.0, 2.5]
        anims = [e.animate.set_color(UPD).set_stroke(width=widths[(k + l) % len(widths)])
                 for l in range(3) for k, e in enumerate(net.edge[l].values())]
        self.play(Write(head), Write(uw), Write(ub), FadeIn(un), *anims, run_time=2.2)
        self.wait(3.0)
