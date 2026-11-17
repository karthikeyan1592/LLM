import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
BASE = np.array([-0.85, -0.7, -0.55, -0.4, -0.25, -0.1, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8])


def mt(s, size=34, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def epoch_slide(scene, title, subtitle, y_range, y_ticks, y_note, cases, band=None, hline=None):
    for m in scene.mobjects:
        m.clear_updaters()
    head = Text(title, font_size=34).to_edge(UP, buff=0.3)
    sub = Text(subtitle, font_size=20, color=GREY_B).next_to(head, DOWN, buff=0.2)
    axes = Axes(x_range=[0, 2.4, 0.4], y_range=y_range, x_length=8.0, y_length=3.8, tips=False,
                y_axis_config={"include_ticks": False}).shift(LEFT * 1.7 + DOWN * 0.7)
    x_lab = Text("epoch", font_size=20).next_to(axes, DOWN, buff=0.15)
    grid = VGroup(*[Line(axes.c2p(0, v), axes.c2p(2.4, v), stroke_width=1, color=GREY_D) for v, _ in y_ticks])
    ticks = VGroup(*[Text(s, font_size=18, color=GREY_B).next_to(axes.c2p(0, v), LEFT, buff=0.12) for v, s in y_ticks])
    y_lab = Text(y_note, font_size=18, color=GREY_B).next_to(axes, UP, buff=0.1).align_to(axes, LEFT)
    header = Text("Three possible runs\nfor this layer:", font_size=20, color=GREY_B)
    fixed, extra_entries = [], []
    if band:
        lo, hi, bcolor, blabel = band
        p0, p1 = axes.c2p(0, lo), axes.c2p(2.4, hi)
        rect = Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1], stroke_width=0,
                         fill_color=bcolor, fill_opacity=0.15)
        rect.move_to((p0 + p1) / 2)
        fixed.append(rect)
        extra_entries.append(VGroup(Rectangle(width=0.7, height=0.22, stroke_width=0, fill_color=bcolor, fill_opacity=0.35),
                                    Text(blabel, font_size=18, color=bcolor)).arrange(RIGHT, buff=0.2))
    if hline:
        v, hlabel = hline
        fixed.append(DashedLine(axes.c2p(0, v), axes.c2p(2.4, v), color=GREY_B, stroke_width=2))
        extra_entries.append(VGroup(DashedLine(ORIGIN, RIGHT * 0.7, color=GREY_B, stroke_width=3),
                                    Text(hlabel, font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.2))
    entries = [VGroup(Line(ORIGIN, RIGHT * 0.7, color=c, stroke_width=6), Text(n, font_size=22, color=c)
                      ).arrange(RIGHT, buff=0.2) for _, c, n, _ in cases]
    legend = VGroup(header, *extra_entries, *entries).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([4.7, 0.4, 0])
    scene.play(*[FadeOut(m) for m in scene.mobjects if not isinstance(m, ValueTracker)])
    scene.play(Write(head), FadeIn(sub))
    scene.play(Create(axes), FadeIn(x_lab), FadeIn(grid), FadeIn(ticks), FadeIn(y_lab), FadeIn(header),
               *[FadeIn(m) for m in fixed], *[FadeIn(e) for e in extra_entries])
    caption = Text("", font_size=22).to_edge(DOWN, buff=0.35)
    scene.add(caption)
    for (f, color, _, text), entry in zip(cases, entries):
        curve = axes.plot(f, x_range=[0, 2.4], color=color, stroke_width=5)
        scene.play(Transform(caption, Text(text, font_size=22, color=color).to_edge(DOWN, buff=0.35)),
                   Create(curve), FadeIn(entry), run_time=2.0)
        scene.wait(1.6)
    scene.wait(1.5)


class ActivationStd(Scene):
    def construct(self):
        title = Text("Activation std  σ(z)", font_size=34).to_edge(UP, buff=0.3)
        formula = mt(r"\sigma = \sqrt{\frac{1}{n-1}\sum_{k}\left(z_k-\mu\right)^2}", 36).move_to([0, 2.55, 0])
        intuition = VGroup(
            Text("Do the outputs of this layer differ from each other?", font_size=22, color=GREY_B),
            Text("σ is the standard deviation of the layer's outputs z(l).", font_size=20, color=GREY_B),
            Text("If σ ≈ 0, all outputs are equal and the layer carries no information about the input.", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.1).move_to([0, 1.3, 0])
        line = NumberLine(x_range=[-1, 1, 0.5], length=10, include_numbers=False).move_to([0, -0.3, 0])
        ticks = VGroup(*[Text(t, font_size=20, color=GREY_B).next_to(line.n2p(v), DOWN, buff=0.2)
                         for t, v in (("-1", -1), ("0", 0), ("1", 1))])
        legend = VGroup(
            VGroup(Dot(color=BLUE_C, radius=0.09), Text("dot = one output z(l)  (one neuron, one sample)", font_size=18, color=GREY_B)
                   ).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.6, color=YELLOW, stroke_width=5), Text("yellow bar = μ ± σ  (the spread)", font_size=18, color=GREY_B)
                   ).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.6).move_to([0, -2.85, 0])
        self.play(Write(title), Write(formula), run_time=1.6)
        self.play(FadeIn(intuition), Create(line), FadeIn(ticks), FadeIn(legend))

        s = ValueTracker(1.0)
        g = ValueTracker(0.5)
        mode = {"m": 0}

        def zs():
            if mode["m"] == 0:
                sv = s.get_value()
                return 0.3 * (1 - sv) + sv * BASE
            return np.tanh(g.get_value() * BASE * 2.2)

        def dots():
            z = zs()
            return VGroup(*[Dot(line.n2p(v) + UP * 0.32 * ((k % 3) - 1), radius=0.09, color=BLUE_C)
                            for k, v in enumerate(z)])

        def bracket():
            z = zs()
            mu, sd = float(np.mean(z)), float(np.std(z, ddof=1))
            y = -1.45
            lo, hi = line.n2p(mu - sd)[0], line.n2p(mu + sd)[0]
            return VGroup(Line([lo, y, 0], [hi, y, 0], color=YELLOW, stroke_width=5),
                          Line([lo, y - 0.12, 0], [lo, y + 0.12, 0], color=YELLOW, stroke_width=5),
                          Line([hi, y - 0.12, 0], [hi, y + 0.12, 0], color=YELLOW, stroke_width=5),
                          Dot([line.n2p(mu)[0], y, 0], radius=0.07, color=YELLOW))

        def readout():
            z = zs()
            return Text(f"μ = {np.mean(z):+.2f}      σ = {np.std(z, ddof=1):.2f}",
                        font_size=26, color=YELLOW).move_to([0, -2.2, 0])

        self.add(always_redraw(dots), always_redraw(bracket), always_redraw(readout))
        caption = Text("healthy: the outputs differ from each other — σ is moderate",
                       font_size=24, color=GREEN_C).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))
        self.wait(2.5)

        self.play(Transform(caption, Text("collapsing: all outputs become equal, σ → 0",
                                          font_size=22, color=ORANGE).to_edge(DOWN, buff=0.35)))
        self.play(s.animate.set_value(0.04), run_time=3.5)
        self.wait(1.8)

        self.play(s.animate.set_value(1.0), Transform(caption, Text("back to healthy …", font_size=24, color=GREEN_C).to_edge(DOWN, buff=0.35)),
                  run_time=2.0)
        mode["m"] = 1
        self.play(Transform(caption, Text("saturating: outputs cluster at ±1 — σ becomes high, gradients vanish",
                                          font_size=22, color=RED_C).to_edge(DOWN, buff=0.35)))
        self.play(g.animate.set_value(4.0), run_time=3.5)
        self.wait(2.5)

        epoch_slide(
            self, "Activation std across epochs",
            "one point per logged epoch and layer — do the outputs still differ from each other?",
            [0, 1.1, 0.25], [(0, "0"), (0.5, "0.5"), (1, "1")], "σ(z)",
            [
                (lambda t: 0.5 + 0.04 * np.sin(4 * t), GREEN_C, "healthy",
                 "healthy: stable and well above 0.1 — the outputs still differ"),
                (lambda t: 0.5 * np.exp(-1.6 * t) + 0.02, ORANGE, "collapsing",
                 "collapsing: σ falls toward 0 — all outputs become equal"),
                (lambda t: 1 - 0.55 * np.exp(-1.1 * t), RED_C, "saturating",
                 "saturating: σ climbs toward 1 — outputs cluster at ±1 (a high σ is not always good)"),
            ],
            hline=(0.1, "0.1 = collapse threshold"),
        )
