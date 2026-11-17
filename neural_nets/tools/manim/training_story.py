import textwrap
import numpy as np
from manim import *

# Logged diagnostics of the notebook run (seed 42, full-batch SGD, lr 0.1, 300 epochs, logged every 30 epochs).
DATA = {
    'E': [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    'train_loss': [0.7092, 0.69414, 0.68832, 0.68289, 0.67512, 0.66281, 0.64446, 0.62208, 0.60179, 0.5879, 0.57973],
    'val_loss': [0.70371, 0.6889, 0.68335, 0.67842, 0.67167, 0.66174, 0.64835, 0.63405, 0.62341, 0.61811, 0.61633],
    'train_acc': [0.4766, 0.5016, 0.5828, 0.5859, 0.6141, 0.6531, 0.6766, 0.6969, 0.6891, 0.6891, 0.6906],
    'val_acc': [0.475, 0.5563, 0.6375, 0.6313, 0.6313, 0.6062, 0.6375, 0.625, 0.6562, 0.6625, 0.675],
    'grad_norm': [[0.02785, 0.026676, 0.027746, 0.030505, 0.034914, 0.040322, 0.044498, 0.044089, 0.037288, 0.027881, 0.020188], [0.033854, 0.026419, 0.024863, 0.026707, 0.031205, 0.037575, 0.043188, 0.043677, 0.037608, 0.028829, 0.021548], [0.036536, 0.0186, 0.015105, 0.019847, 0.028665, 0.039658, 0.050017, 0.05448, 0.050061, 0.040234, 0.030252]],
    'weight_norm': [[1.098474, 1.093179, 1.09768, 1.112157, 1.138066, 1.177698, 1.231303, 1.292925, 1.351185, 1.396767, 1.42798], [0.69107, 0.683203, 0.692761, 0.717407, 0.759462, 0.822969, 0.90934, 1.010167, 1.106821, 1.183134, 1.235332], [0.336178, 0.347283, 0.373704, 0.417461, 0.486141, 0.586053, 0.719404, 0.876556, 1.033832, 1.168271, 1.271829]],
    'update_ratio': [[0.002535, 0.00244, 0.002528, 0.002743, 0.003068, 0.003424, 0.003614, 0.00341, 0.00276, 0.001996, 0.001414], [0.004899, 0.003867, 0.003589, 0.003723, 0.004109, 0.004566, 0.004749, 0.004324, 0.003398, 0.002437, 0.001744], [0.010868, 0.005356, 0.004042, 0.004754, 0.005896, 0.006767, 0.006953, 0.006215, 0.004842, 0.003444, 0.002379]],
    'act_std': [[0.540844, 0.540476, 0.540804, 0.541851, 0.543737, 0.546658, 0.550921, 0.556578, 0.562769, 0.568116, 0.571942], [0.371008, 0.395924, 0.410917, 0.418886, 0.422868, 0.425794, 0.43259, 0.448267, 0.4708, 0.492378, 0.507862], [0.018073, 0.017124, 0.018527, 0.022763, 0.031154, 0.046043, 0.070041, 0.102773, 0.138125, 0.168609, 0.191028]],
    'sat_frac': [[0.0109, 0.0104, 0.0094, 0.0099, 0.0125, 0.0146, 0.0214, 0.026, 0.0312, 0.0385, 0.0406], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [None, None, None, None, None, None, None, None, None, None, None]],
}

E = DATA["E"]
TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
LC = ["#1f77b4", "#ff7f0e", "#2ca02c"]
LNAME = ["L1  (tanh)", "L2  (tanh)", "L3  (sigmoid output)"]
PHASES = [(1, 100, "plateau", GREY_B), (100, 210, "fast decrease", GREEN_C), (210, 300, "slowing decrease", ORANGE)]
PLOT_C = np.array([-2.75, -0.3, 0.0])
RIGHT_X, RIGHT_TOP = 1.55, 2.4
SEE_C, WHY_C, SO_C = BLUE_C, YELLOW, GREEN_C


def mt(s, size=28, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def txt(s, size=19, color=WHITE, width=None):
    if width:
        s = textwrap.fill(s, width=width)
    return Text(s, font_size=size, color=color, line_spacing=0.9)


class Chart:
    pass


def build_chart(y_range, y_ticks, band=None, hline=None):
    c = Chart()
    ymin, ymax = y_range
    ax = Axes(x_range=[0, 300, 100], y_range=[ymin, ymax, (ymax - ymin) / 4], x_length=6.9, y_length=3.6, tips=False,
              axis_config={"include_ticks": False, "stroke_width": 2, "color": GREY_B}).move_to(PLOT_C)
    c.ax = ax
    bands = VGroup()
    labels = VGroup()
    for lo, hi, name, col in PHASES:
        p0, p1 = ax.c2p(lo, ymin), ax.c2p(hi, ymax)
        r = Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1], stroke_width=0, fill_color=col, fill_opacity=0.11)
        r.move_to((p0 + p1) / 2)
        bands.add(r)
        labels.add(Text(name, font_size=17, color=col).move_to([(p0[0] + p1[0]) / 2, p1[1] + 0.22, 0]))
    xt = VGroup(*[Text(str(x), font_size=16, color=GREY_B).next_to(ax.c2p(x, ymin), DOWN, buff=0.12) for x in (0, 100, 200, 300)])
    xl = Text("epoch", font_size=17, color=GREY_B).next_to(ax, DOWN, buff=0.48)
    yt = VGroup(*[Text(s, font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v, s in y_ticks])
    grid = VGroup(*[Line(ax.c2p(0, v), ax.c2p(300, v), stroke_width=1, color=GREY_D) for v, _ in y_ticks])
    extra = VGroup()
    if band:
        lo, hi, col, name = band
        p0, p1 = ax.c2p(0, lo), ax.c2p(300, hi)
        r = Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1], stroke_width=0, fill_color=col, fill_opacity=0.22)
        r.move_to((p0 + p1) / 2)
        t = Text(name, font_size=16, color=col).move_to([p1[0] - 1.2, p1[1] - 0.2, 0])
        extra.add(r, t)
    if hline:
        v, name, col = hline
        extra.add(DashedLine(ax.c2p(0, v), ax.c2p(300, v), color=col, stroke_width=2),
                  Text(name, font_size=16, color=col).next_to(ax.c2p(300, v), UP, buff=0.08).shift(LEFT * 0.9))
    c.static = VGroup(bands, labels, grid, ax, xt, xl, yt, extra)
    return c


def line_for(chart, vals, color, r=0.05):
    return chart.ax.plot_line_graph(E[:len(vals)], vals, line_color=color, stroke_width=4, add_vertex_dots=True,
                                    vertex_dot_radius=r, vertex_dot_style={"stroke_width": 0, "fill_color": color})


def legend_row(names=LNAME, colors=LC, y=2.55):
    items = [VGroup(Line(ORIGIN, RIGHT * 0.5, color=c, stroke_width=5), Text(n, font_size=17, color=c)
                    ).arrange(RIGHT, buff=0.15) for n, c in zip(names, colors)]
    return VGroup(*items).arrange(RIGHT, buff=0.55).move_to([PLOT_C[0], y, 0])


def bullets(items, tag, tag_color, max_h=5.6, width=36, size=19, top=RIGHT_TOP):
    head = Text(tag, font_size=17, color=tag_color, weight=BOLD)
    texts = []
    for it in items:
        color = WHITE
        if isinstance(it, tuple):
            it, color = it
        texts.append(txt(it, size, color, width))
    g = VGroup(head, *texts).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
    if g.height > max_h:
        g.scale_to_fit_height(max_h)
    g.move_to([0, 0, 0])
    g.align_to(np.array([RIGHT_X, 0, 0]), LEFT).align_to(np.array([0, top, 0]), UP)
    return g


class TrainingStory(Scene):
    PACE = 0.32

    def hold(self, text, minimum=1.5, extra=0.6):
        self.wait(max(minimum, self.PACE * len(text.split()) + extra))

    def reveal(self, g, acts=None):
        self.play(FadeIn(g[0], shift=RIGHT * 0.15), run_time=0.6)
        for i, t in enumerate(g[1:]):
            self.play(FadeIn(t, shift=RIGHT * 0.15), run_time=0.9)
            if acts and i < len(acts) and acts[i]:
                self.play(*acts[i](), run_time=0.9)
            self.hold(t.text)

    # ------------------------------------------------------------------ helpers
    def ring(self, chart, l, i, label, color=WHITE, direction=UP, series=None):
        v = series[l][i]
        p = chart.ax.c2p(E[i], v)
        ring = Circle(radius=0.14, color=color, stroke_width=3).move_to(p)
        lab = Text(label, font_size=16, color=color).next_to(ring, direction, buff=0.1)
        return VGroup(ring, lab)

    def draw_curves(self, chart, series, layers=(0, 1, 2), fy=lambda v: v, time=2.4):
        out = []
        for l in layers:
            vals = [fy(v) for v in series[l]]
            vd = line_for(chart, vals, LC[l])
            self.play(Create(vd["line_graph"]), run_time=time)
            self.play(FadeIn(vd["vertex_dots"]), run_time=0.5)
            out.append(vd)
            self.wait(0.6)
        return out

    def open_panel(self, title, subtitle, y_range, y_ticks, band=None, hline=None, legend_names=None):
        t = Text(title, font_size=28).move_to([0, 3.45, 0])
        s = Text(subtitle, font_size=19, color=GREY_B).move_to([0, 2.98, 0])
        chart = build_chart(y_range, y_ticks, band, hline)
        leg = legend_row() if legend_names is None else legend_row(*legend_names)
        self.play(FadeIn(t), FadeIn(s), run_time=0.8)
        self.play(FadeIn(chart.static), FadeIn(leg), run_time=1.2)
        return chart, VGroup(t, s, leg)

    def close_panel(self, *mobs):
        self.play(*[FadeOut(m) for m in mobs], run_time=1.0)

    # ------------------------------------------------------------------ parts
    def part_intro(self):
        title = Text("Reading the five diagnostic plots", font_size=40).move_to([0, 3.0, 0])
        sub = Text("the story of one training run", font_size=26, color=GREY_B).move_to([0, 2.3, 0])
        self.play(Write(title), run_time=1.6)
        self.play(FadeIn(sub))
        self.wait(1.2)
        setup = VGroup(
            txt("The network: 3 → 3 → 2 → 1 neurons, tanh / tanh / sigmoid.", 24),
            txt("Training: full-batch SGD, learning rate 0.1, 300 epochs.", 24),
            txt("Every 30 epochs we logged five numbers for each layer.", 24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 0.5, 0])
        for t in setup:
            self.play(FadeIn(t, shift=RIGHT * 0.2), run_time=0.9)
            self.hold(t.text, 2.0)
        self.play(FadeOut(setup))
        five = [
            ("gradient norm", "size of the gradient of the loss w.r.t. a layer's weights"),
            ("weight norm", "how big a layer's weights are"),
            ("update ratio", "step size compared with the weights"),
            ("activation std", "how spread out a layer's outputs are"),
            ("saturated fraction", "fraction of tanh neurons that are saturated (|z| > 0.95)"),
        ]
        rows = VGroup(*[VGroup(Text(a, font_size=24, color=YELLOW), Text(b, font_size=22)).arrange(RIGHT, buff=0.4)
                        for a, b in five])
        for r in rows:
            if r[0].width > 2.9:
                r[0].scale_to_fit_width(2.9)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, 0.5, 0])
        # align the descriptions in a column
        for r in rows:
            r[1].next_to(np.array([-2.3, r[0].get_center()[1], 0]), RIGHT, buff=0.0)
            r[0].next_to(np.array([-2.5, r[0].get_center()[1], 0]), LEFT, buff=0.0)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.8)
            self.hold(r[1].text, 1.8)
        self.wait(0.8)
        self.play(FadeOut(rows))
        key = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.6, color=c, stroke_width=6), Text(n, font_size=22, color=c)
                              ).arrange(RIGHT, buff=0.2) for n, c in zip(LNAME, LC)]).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        key.move_to([0, 0.3, 0])
        n1 = txt("Each plot has one line per layer, in these colours.", 24).move_to([0, 1.5, 0])
        n2 = txt("The shaded bands mark three stages of the run: plateau, fast decrease and slowing decrease of the loss. We will see why.", 22,
                 GREY_B, 70).move_to([0, -1.5, 0])
        self.play(FadeIn(n1))
        self.play(FadeIn(key, shift=UP * 0.1))
        self.hold(n1.text, 2.5)
        self.play(FadeIn(n2))
        self.hold(n2.text, 3.0)
        self.play(FadeOut(VGroup(title, sub, key, n1, n2)))

    def part_loss(self):
        chart, tops = self.open_panel(
            "The anchor: the loss", "everything in the next five plots explains this one curve",
            [0.55, 0.72], [(0.6, "0.60"), (0.65, "0.65"), (0.7, "0.70")], legend_names=None)
        self.play(FadeOut(tops[2]))
        leg = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.5, color=YELLOW, stroke_width=5), Text("train loss", font_size=17, color=YELLOW)
                            ).arrange(RIGHT, buff=0.15),
                     VGroup(DashedLine(ORIGIN, RIGHT * 0.5, color=WHITE, stroke_width=4), Text("validation loss", font_size=17)
                            ).arrange(RIGHT, buff=0.15)).arrange(RIGHT, buff=0.6).move_to([PLOT_C[0], 2.55, 0])
        self.play(FadeIn(leg))
        tr = line_for(chart, DATA["train_loss"], YELLOW)
        self.play(Create(tr["line_graph"]), run_time=3.0)
        self.play(FadeIn(tr["vertex_dots"]))
        va = chart.ax.plot_line_graph(E, DATA["val_loss"], line_color=WHITE, stroke_width=3, add_vertex_dots=False)
        dashed = DashedVMobject(va["line_graph"], num_dashes=40)
        self.play(Create(dashed), run_time=2.0)
        self.wait(0.8)
        tl = DATA["train_loss"]
        stages = [
            ([f"Epochs 1–30: a quick first drop ({tl[0]:.3f} → {tl[1]:.3f}).",
              "Probably the easy part: learning the class balance."], 0, 1, "0.709 → 0.694"),
            ([f"Epochs 30–90: almost flat ({tl[1]:.3f} → {tl[3]:.3f}).", "This is the plateau."], 1, 3, "0.694 → 0.683"),
            ([f"Epochs 90–210: the loss decreases faster ({tl[3]:.3f} → {tl[7]:.3f}).", "This is the fast-decrease phase."], 3, 7, "0.683 → 0.622"),
            ([f"Epochs 210–300: still falling ({tl[7]:.3f} → {tl[10]:.3f}),", "but more slowly: the slowing-decrease phase."], 7, 10, "0.622 → 0.580"),
        ]
        prev = None
        for lines, a, b, lab in stages:
            g = bullets([lines[0], lines[1]], "THE LOSS", SEE_C)
            seg = Line(chart.ax.c2p(E[a], tl[a]), chart.ax.c2p(E[b], tl[b]), color=RED_C, stroke_width=6)
            tag = Text(lab, font_size=16, color=RED_C).next_to(seg, UP if a < 7 else RIGHT, buff=0.12)
            if a >= 7:
                tag.next_to(seg.get_center(), DOWN, buff=0.2)
            anims = [Create(seg), FadeIn(tag)]
            if prev:
                anims += [FadeOut(prev[0]), FadeOut(prev[1]), FadeOut(prev[2])]
            self.play(FadeIn(g[0]), *anims, run_time=1.0)
            for t in g[1:]:
                self.play(FadeIn(t, shift=RIGHT * 0.15), run_time=0.9)
                self.hold(t.text)
            prev = (g, seg, tag)
        self.wait(1.0)
        end = txt("At the end the train and validation losses start to separate (0.580 vs 0.616). We come back to this at the end.",
                  19, YELLOW, 36).move_to([RIGHT_X + 2.6, -2.3, 0])
        self.play(FadeIn(end))
        self.hold(end.text, 3.0)
        self.close_panel(chart.static, tops, leg, tr, dashed, *prev, end)

    # --------------------------------------------------------------- gradient norm
    def part_grad(self):
        key = "grad_norm"
        S = DATA[key]
        chart, tops = self.open_panel(
            "1 · Gradient norm  ‖∇W‖", "the size of the gradient of the loss with respect to each layer's weights",
            [0.0, 0.06], [(0, "0"), (0.02, "0.02"), (0.04, "0.04"), (0.06, "0.06")])
        curves = self.draw_curves(chart, S)
        c_start = lambda: [FadeIn(self.ring(chart, 2, 0, "0.037", LC[2], UP, S))]
        c_dip = lambda: [FadeIn(self.ring(chart, 2, 2, "0.015", LC[2], DOWN, S))]
        c_peak = lambda: [FadeIn(self.ring(chart, 2, 7, "0.054", LC[2], UP, S))]
        see = bullets([
            "All three layers start around 0.03.",
            "They dip first: the output layer falls from 0.037 to 0.015 by epoch 60.",
            "Then they climb to a peak near epochs 180–210 (the output layer reaches 0.054).",
            "And fall back to 0.02–0.03 by the end.",
        ], "WHAT WE SEE", SEE_C)
        marks = [c_start(), c_dip(), c_peak(), []]
        self.play(FadeIn(see[0]), run_time=0.6)
        shown = []
        for i, t in enumerate(see[1:]):
            self.play(FadeIn(t, shift=RIGHT * 0.15), run_time=0.9)
            if marks[i]:
                m = marks[i][0]
                shown.append(m.mobject if hasattr(m, "mobject") else m)
                self.play(*marks[i], run_time=0.9)
            self.hold(t.text)
        self.play(FadeOut(see), *[FadeOut(m) for m in self.mobjects if isinstance(m, VGroup) and len(m) == 2 and isinstance(m[0], Circle)], run_time=0.9)

        why = bullets([
            "Backprop multiplies by the weights on its way back.",
            "At the start the weights are small, so the error signal δ is small in the earlier layers: slow learning.",
            "As the output weights grow, δ in the earlier layers grows and the gradients rise.",
        ], "WHY", WHY_C, max_h=3.1)
        # mini visual: the backward chain
        names = [r"\delta^{(3)}", r"\delta^{(2)}", r"\delta^{(1)}"]
        cols = [LC[2], LC[1], LC[0]]
        boxes = VGroup(*[VGroup(RoundedRectangle(width=1.0, height=0.62, corner_radius=0.1, color=c, stroke_width=3), mt(n, 24, c))
                         for n, c in zip(names, cols)]).arrange(RIGHT, buff=1.25).move_to([4.2, -2.35, 0])

        def arrows(w, color):
            g = VGroup()
            for i, lab in enumerate([r"\times W^{(3)\top}", r"\times W^{(2)\top}"]):
                a = Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.05, stroke_width=w, color=color,
                          max_tip_length_to_length_ratio=0.45)
                g.add(a)
            return g

        weak, strong = arrows(2, GREY_B), arrows(9, RED_C)
        labs = VGroup(mt(r"\times W^{(3)\top}", 18, GREY_B).next_to(weak[0], UP, buff=0.12),
                      mt(r"\times W^{(2)\top}", 18, GREY_B).next_to(weak[1], UP, buff=0.12))
        cap_w = txt("plateau: small weights, small δ", 17, GREY_B).move_to([4.2, -3.3, 0])
        cap_s = txt("fast decrease: larger weights, larger δ", 17, RED_C).move_to([4.2, -3.3, 0])
        self.play(FadeIn(why[0]), run_time=0.6)
        self.play(FadeIn(why[1], shift=RIGHT * 0.15), run_time=0.9)
        self.play(FadeIn(boxes), FadeIn(weak), FadeIn(labs), FadeIn(cap_w), run_time=1.0)
        self.hold(why[1].text)
        self.play(FadeIn(why[2], shift=RIGHT * 0.15), run_time=0.9)
        self.play(Transform(weak, strong), Transform(cap_w, cap_s), run_time=1.5)
        self.hold(why[2].text)
        last = txt("Near a minimum of the loss the gradient flattens, so they decrease again.", 19, WHITE, 36)
        last.next_to(why, DOWN, aligned_edge=LEFT, buff=0.26)
        self.play(FadeIn(last, shift=RIGHT * 0.15))
        self.hold(last.text)
        self.play(FadeOut(VGroup(why, last, boxes, weak, labs, cap_w)), run_time=0.9)

        so = bullets([
            "A rise and fall, not the textbook steady decrease.",
            "It never reaches 0 and never blows up.",
            ("Layer 1 gets about as much gradient as the output layer: nothing shrinks on the way back.", SO_C),
        ], "SO WHAT", SO_C)
        self.reveal(so)
        self.wait(1.0)
        self.close_panel(chart.static, tops, so, *curves)

    # ---------------------------------------------------------------- weight norm
    def part_weight(self):
        S = DATA["weight_norm"]
        chart, tops = self.open_panel(
            "2 · Weight norm  ‖W‖", "how big each layer's weights have grown",
            [0.0, 1.6], [(0, "0"), (0.5, "0.5"), (1.0, "1.0"), (1.5, "1.5")])
        curves = self.draw_curves(chart, S)
        see = bullets([
            "Flat for roughly the first 90 epochs.",
            "Then all three layers grow.",
            "L1: 1.10 → 1.43 (+30 %)",
            "L2: 0.69 → 1.24 (+80 %)",
            "L3: 0.34 → 1.27 (about ×4)",
        ], "WHAT WE SEE", SEE_C)
        self.reveal(see)
        self.play(FadeOut(see), run_time=0.9)

        why = bullets([
            "The output layer starts smallest.",
            "To move a prediction away from 0.5, its weights must grow, so it grows the most.",
            "The hidden layers change less.",
        ], "WHY", WHY_C, max_h=2.9)
        base = -3.35
        scale = 1.35
        first = [S[l][0] for l in range(3)]
        last = [S[l][-1] for l in range(3)]
        bars = VGroup()
        tags = VGroup()
        for l in range(3):
            cx = 2.55 + 1.65 * l
            b1 = Rectangle(width=0.5, height=first[l] * scale, stroke_width=0, fill_color=GREY_B, fill_opacity=0.9)
            b2 = Rectangle(width=0.5, height=last[l] * scale, stroke_width=0, fill_color=LC[l], fill_opacity=0.95)
            b1.move_to([cx - 0.3, base, 0], aligned_edge=DOWN)
            b2.move_to([cx + 0.3, base, 0], aligned_edge=DOWN)
            bars.add(b1, b2)
            tags.add(Text(f"L{l + 1}", font_size=17, color=LC[l]).move_to([cx, base - 0.25, 0]),
                     Text(f"×{last[l] / first[l]:.1f}", font_size=18, color=YELLOW).move_to([cx + 0.3, base + last[l] * scale + 0.22, 0]))
        key = VGroup(Rectangle(width=0.3, height=0.16, stroke_width=0, fill_color=GREY_B, fill_opacity=0.9),
                     Text("epoch 1", font_size=15, color=GREY_B),
                     Rectangle(width=0.3, height=0.16, stroke_width=0, fill_color=WHITE, fill_opacity=0.9),
                     Text("epoch 300 (layer colour)", font_size=15)).arrange(RIGHT, buff=0.12).move_to([4.2, -0.95, 0])
        self.play(FadeIn(why[0]), run_time=0.6)
        self.play(FadeIn(why[1], shift=RIGHT * 0.15), FadeIn(key), FadeIn(bars[0::2]), FadeIn(tags[0::2]), run_time=1.0)
        self.hold(why[1].text)
        self.play(FadeIn(why[2], shift=RIGHT * 0.15), *[GrowFromEdge(b, DOWN) for b in bars[1::2]], FadeIn(tags[1::2]), run_time=1.5)
        self.hold(why[2].text)
        self.wait(1.0)
        self.play(FadeOut(VGroup(why, bars, tags, key)), run_time=0.9)

        so = bullets([
            "Weights growing is what learning looks like here.",
            "No layer is frozen and none is running away.",
            ("The three norms end up at a similar size, 1.2–1.4.", SO_C),
        ], "SO WHAT", SO_C)
        self.reveal(so)
        self.wait(1.0)
        self.close_panel(chart.static, tops, so, *curves)

    # --------------------------------------------------------------- update ratio
    def part_ratio(self):
        S = DATA["update_ratio"]
        fy = lambda v: float(np.log10(v) + 4)
        chart, tops = self.open_panel(
            "3 · Update ratio  lr·‖∇W‖ / ‖W‖", "the size of each step, compared with the size of the weights (log scale)",
            [0.6, 3.4], [(1, "0.001"), (2, "0.01"), (3, "0.1")],
            band=(2, 3, GREEN_C, "rule-of-thumb band"))
        curves = self.draw_curves(chart, S, fy=fy)
        see = bullets([
            "Between 0.0014 and 0.011.",
            "Almost everything sits below the 0.01–0.1 band.",
            "It follows the gradient norm: a dip, a maximum near epoch 180, then a decrease.",
            "L3 is the highest line: its weights are the smallest.",
        ], "WHAT WE SEE", SEE_C)
        self.reveal(see)
        self.play(FadeOut(see), run_time=0.9)

        why = bullets([
            "The ratio is the gradient divided by the weights, times lr.",
            "A small gradient or big weights make it small.",
        ], "WHY", WHY_C, max_h=2.2)
        f = mt(r"\text{ratio} = \frac{\text{lr}\cdot\|\nabla W\|}{\|W\|}", 30).move_to([4.2, -0.6, 0])
        bar_w = 4.4
        x0 = 2.0
        rows = []
        for k, (ep_i, lab) in enumerate([(0, "epoch 1"), (10, "epoch 300")]):
            ratio = S[2][ep_i]
            y = -1.8 - 1.1 * k
            wb = Rectangle(width=bar_w, height=0.28, stroke_width=0, fill_color=LC[2], fill_opacity=0.85).move_to([x0, y, 0], aligned_edge=LEFT)
            sb = Rectangle(width=max(bar_w * ratio, 0.03), height=0.28, stroke_width=0, fill_color=RED_C, fill_opacity=1).move_to([x0, y - 0.38, 0], aligned_edge=LEFT)
            t1 = Text(f"L3 weights, {lab}", font_size=15, color=GREY_B).move_to([x0, y + 0.3, 0], aligned_edge=LEFT)
            t2 = Text(f"step = {ratio * 100:.2f} % of the weights", font_size=16, color=RED_C).next_to(sb, RIGHT, buff=0.15)
            rows.append(VGroup(wb, sb, t1, t2))
        self.play(FadeIn(why[0]), run_time=0.6)
        self.play(FadeIn(why[1], shift=RIGHT * 0.15), FadeIn(f), run_time=0.9)
        self.hold(why[1].text)
        for r in rows:
            self.play(FadeIn(r), run_time=1.0)
            self.wait(1.8)
        self.wait(1.0)
        self.play(FadeOut(VGroup(why, f, *rows)), run_time=0.9)

        so = bullets([
            "Each step moves the weights by a fraction of a percent.",
            "Safe: nowhere near 1.",
            "But cautious: each step is only 0.1–1 % of the weights, so learning is slow.",
            ("Below the band means slow here, not stalled: the loss was still falling at epoch 300.", SO_C),
            ("(The 0.01–0.1 band is a rule of thumb from larger networks and Adam.)", GREY_B),
        ], "SO WHAT", SO_C)
        self.reveal(so)
        self.wait(1.0)
        self.close_panel(chart.static, tops, so, *curves)

    # ------------------------------------------------------------ activation std
    def part_std(self):
        S = DATA["act_std"]
        chart, tops = self.open_panel(
            "4 · Activation std  σ(z)", "how spread out each layer's outputs are",
            [0.0, 0.65], [(0, "0"), (0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6")],
            hline=(0.1, "0.1", RED_C))
        curves = self.draw_curves(chart, S)
        cross = lambda: [FadeIn(self.ring(chart, 2, 7, "≈ 0.1 at epoch 205", LC[2], UP, S))]
        see = bullets([
            "L1 stays near 0.54–0.57: steady.",
            "L2 rises from 0.37 to 0.51.",
            "L3 starts at 0.018 and rises to 0.19, crossing 0.1 around epoch 205.",
        ], "WHAT WE SEE", SEE_C)
        self.play(FadeIn(see[0]), run_time=0.6)
        for i, t in enumerate(see[1:]):
            self.play(FadeIn(t, shift=RIGHT * 0.15), run_time=0.9)
            if i == 1:
                self.play(*cross(), run_time=0.9)
            self.hold(t.text)
        self.play(FadeOut(see), *[FadeOut(m) for m in self.mobjects if isinstance(m, VGroup) and len(m) == 2 and isinstance(m[0], Circle)], run_time=0.9)

        why = bullets([
            "L3 is the sigmoid output, so its σ is the spread of the predictions.",
            "Early on all predictions are about 0.5.",
            "Later the predictions differ between samples.",
        ], "WHY", WHY_C, max_h=2.5)
        rng = np.random.default_rng(3)
        z = rng.standard_normal(24)
        z = (z - z.mean()) / z.std(ddof=1)
        strips = VGroup()
        for k, (ep_i, lab) in enumerate([(0, "epoch 1"), (10, "epoch 300")]):
            sd = S[2][ep_i]
            y = -2.1 - 1.1 * k
            nl = NumberLine(x_range=[0, 1, 0.5], length=4.4, include_numbers=False, stroke_width=2, color=GREY_B).move_to([4.2, y, 0])
            ticks = VGroup(*[Text(s, font_size=14, color=GREY_B).next_to(nl.n2p(v), DOWN, buff=0.1) for v, s in ((0, "0"), (0.5, "0.5"), (1, "1"))])
            pos = np.clip(0.5 + sd * z, 0.02, 0.98)
            dots = VGroup(*[Dot(nl.n2p(p) + UP * 0.2 * ((j % 3) - 1), radius=0.055, color=LC[2]) for j, p in enumerate(pos)])
            cap = Text(f"{lab}:  σ = {sd:.3f}", font_size=16, color=LC[2]).next_to(nl, UP, buff=0.42).align_to(nl, LEFT)
            strips.add(VGroup(nl, ticks, dots, cap))
        note = txt("predictions y for 24 samples (positions illustrative, spread = measured σ)", 14, GREY_B, 52).move_to([4.2, -3.75, 0])
        self.play(FadeIn(why[0]), run_time=0.6)
        self.play(FadeIn(why[1], shift=RIGHT * 0.15), FadeIn(strips[0]), FadeIn(note), run_time=1.0)
        self.hold(why[1].text)
        self.play(FadeIn(why[2], shift=RIGHT * 0.15), FadeIn(strips[1]), run_time=1.0)
        self.hold(why[2].text)
        self.wait(1.0)
        self.play(FadeOut(VGroup(why, strips, note)), run_time=0.9)

        so = bullets([
            "The rise through 0.1 comes just as the loss starts to fall faster.",
            ("The one plot where a low σ is not a warning: the output layer should start narrow and widen.", SO_C),
            "The hidden layers never come near 0, so nothing collapsed.",
        ], "SO WHAT", SO_C)
        self.reveal(so)
        self.wait(1.0)
        self.close_panel(chart.static, tops, so, *curves)

    # ------------------------------------------------------------ saturated fraction
    def part_sat(self):
        S = DATA["sat_frac"]
        chart, tops = self.open_panel(
            "5 · Saturated fraction", "fraction of tanh neurons that are saturated (|z| > 0.95)",
            [0.0, 0.05], [(0, "0 %"), (0.01, "1 %"), (0.02, "2 %"), (0.03, "3 %"), (0.04, "4 %")])
        curves = self.draw_curves(chart, S, layers=(0, 1))
        note = txt("L3 has no line: it is a sigmoid, and we only measure tanh neurons.", 15, LC[2], 60).move_to([PLOT_C[0], -3.4, 0])
        see = bullets([
            "L1 rises slowly from 1.1 % to 4.1 %.",
            "L2 stays at exactly 0.",
        ], "WHAT WE SEE", SEE_C)
        self.reveal(see)
        self.play(FadeIn(note))
        self.hold(note.text, 3.0)
        self.play(FadeOut(see), FadeOut(note), run_time=0.9)

        why = bullets([
            "Layer 1 sees the standardized inputs, whose outliers reach ±3, so some of its sums are large.",
            "Layer 2 only sees z(1), bounded in (−1, 1), so its sums stay small.",
        ], "WHY", WHY_C, max_h=2.6)
        ax = Axes(x_range=[-3, 3, 1], y_range=[-1.3, 1.3, 1], x_length=4.4, y_length=1.5, tips=False,
                  axis_config={"include_ticks": False, "stroke_width": 2, "color": GREY_B}).move_to([4.2, -1.65, 0])
        th = ax.plot(lambda a: np.tanh(a), x_range=[-3, 3], color=WHITE, stroke_width=3)
        zl = ax.c2p(-3, -1.3), ax.c2p(-1.83, 1.3)
        zr = ax.c2p(1.83, -1.3), ax.c2p(3, 1.3)
        zones = VGroup(*[Rectangle(width=b[0] - a[0], height=b[1] - a[1], stroke_width=0, fill_color=ORANGE, fill_opacity=0.25
                                   ).move_to((a + b) / 2) for a, b in (zl, zr)])
        zt = Text("flat ends: |z| > 0.95", font_size=14, color=ORANGE).next_to(ax, UP, buff=0.08)
        def span(lo, hi, y, col, label):
            xs = ax.c2p(lo, 0)[0], ax.c2p(hi, 0)[0]
            bar = Line([xs[0], y, 0], [xs[1], y, 0], color=col, stroke_width=8)
            lab = Text(label, font_size=14, color=col).next_to(bar, DOWN, buff=0.08)
            return VGroup(bar, lab)
        b1 = span(-2.6, 2.6, -2.85, LC[0], "layer 1 sums: wide, the tails reach the flat ends")
        b2 = span(-1.0, 1.0, -3.45, LC[1], "layer 2 sums: narrow, stay in the steep middle")
        self.play(FadeIn(why[0]), run_time=0.6)
        self.play(FadeIn(why[1], shift=RIGHT * 0.15), FadeIn(ax), Create(th), FadeIn(zones), FadeIn(zt), run_time=1.2)
        self.play(FadeIn(b1), run_time=1.0)
        self.hold(why[1].text)
        self.play(FadeIn(b2), run_time=1.0)
        self.wait(1.5)
        self.wait(1.0)
        self.play(FadeOut(VGroup(why, ax, th, zones, zt, b1, b2)), run_time=0.9)

        so = bullets([
            "The rise of L1 follows its weights growing: larger weights give larger pre-activations, so more outputs move toward ±1.",
            "Tiny and harmless here: at most 4 %.",
            ("A slow rise like this is what to watch if we trained much longer or with a larger learning rate.", SO_C),
        ], "SO WHAT", SO_C)
        self.reveal(so)
        self.wait(1.0)
        self.close_panel(chart.static, tops, so, *curves)

    # ------------------------------------------------------------------ together
    def part_together(self):
        title = Text("How the five plots connect", font_size=34).move_to([0, 3.4, 0])
        self.play(FadeIn(title))
        cols = ["gradient\nnorm", "weight\nnorm", "update\nratio", "output\nσ", "L1 sat\nfraction"]
        rows = [
            ("plateau", GREY_B, ["small, dipping", "flat", "low", "≈ 0.02", "≈ 1 %"]),
            ("fast decrease", GREEN_C, ["rising to a peak", "growing", "maximum", "crosses 0.1", "rising"]),
            ("slowing decrease", ORANGE, ["falling", "still growing", "falling", "0.19 and rising", "4 %"]),
        ]
        x0, dx = -3.4, 2.3
        head = VGroup(*[Text(c, font_size=18, color=YELLOW, line_spacing=0.8).move_to([x0 + dx * j, 2.3, 0]) for j, c in enumerate(cols)])
        self.play(FadeIn(head))
        cells = []
        for i, (name, col, vals) in enumerate(rows):
            y = 1.2 - 0.95 * i
            label = Text(name, font_size=18, color=col).move_to([-5.6, y, 0])
            band = Rectangle(width=12.6, height=0.8, stroke_width=0, fill_color=col, fill_opacity=0.10).move_to([0.25, y, 0])
            cs = VGroup(*[Text(v, font_size=17).move_to([x0 + dx * j, y, 0]) for j, v in enumerate(vals)])
            cells.append(VGroup(label, band, cs))
        for g in cells:
            self.play(FadeIn(g[1]), FadeIn(g[0]), run_time=0.7)
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in g[2]], lag_ratio=0.25), run_time=1.6)
            self.wait(1.2)
        chain = [
            ("plateau", GREY_B, "weights start small, so backprop gives weak gradients, so the steps are tiny, so the weights barely move and the output stays near 0.5."),
            ("fast decrease", GREEN_C, "the output weights grow, δ in the earlier layers grows, the gradients rise, the steps get bigger, the predictions spread out and the loss decreases faster."),
            ("slowing decrease", ORANGE, "the slope flattens, so gradients and steps fall; the weights are still growing and the saturated fraction of L1 is still rising: not converged."),
        ]
        prev = None
        for i, (name, col, s) in enumerate(chain):
            t = txt(f"{name}: {s}", 19, WHITE, 78).move_to([0, -2.3, 0])
            hl = SurroundingRectangle(cells[i][1], color=col, buff=0.0, stroke_width=3)
            if prev is None:
                self.play(FadeIn(t), Create(hl), run_time=1.0)
            else:
                self.play(FadeOut(prev[0]), FadeOut(prev[1]), FadeIn(t), Create(hl), run_time=1.0)
            self.hold(t.text, 3.0)
            prev = (t, hl)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, head, *[g for g in cells], prev[0], prev[1])))

    def part_verdict(self):
        title = Text("Verdict and what to do next", font_size=36).move_to([0, 3.3, 0])
        self.play(FadeIn(title))
        checks = [
            "No vanishing or exploding gradients: the gradient norm never leaves 0.015–0.055.",
            "No layer with unchanged weights, none growing without bound: every weight norm grows steadily.",
            "No collapsed layer: the hidden σ never drops near 0.",
            "No saturation to speak of: at most 4 % of the tanh outputs are saturated.",
            "Update ratio far below the band: the run is safe but slow.",
        ]
        items = []
        for i, c in enumerate(checks):
            dot = Dot(color=SO_C if i < 4 else YELLOW, radius=0.09)
            t = txt(c, 22, WHITE, 70)
            items.append(VGroup(dot, t).arrange(RIGHT, buff=0.3))
        g = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 1.2, 0])
        for it in g:
            self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.9)
            self.hold(it[1].text, 2.2)
        v = Text("Healthy, but under-trained.", font_size=32, color=YELLOW).move_to([0, -1.5, 0])
        self.play(Write(v), run_time=1.4)
        self.wait(1.5)
        nxt = txt("Next: more epochs or a larger learning rate would speed this up. Expect diminishing returns on validation, "
                  "because the train and validation losses are starting to separate (0.580 vs 0.616). Keep an eye on L1's saturated fraction: it should stay well below 50 %.",
                  21, GREY_B, 78).move_to([0, -2.9, 0])
        self.play(FadeIn(nxt))
        self.hold(nxt.text, 4.0)
        self.wait(2.0)

    def construct(self):
        self.part_intro()
        self.part_loss()
        self.part_grad()
        self.part_weight()
        self.part_ratio()
        self.part_std()
        self.part_sat()
        self.part_together()
        self.part_verdict()
