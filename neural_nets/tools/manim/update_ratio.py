import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
BAR_X0, BAR_W = -4.0, 8.0


def mt(s, size=34, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def bar(y, width, color):
    r = Rectangle(width=max(width, 0.03), height=0.4, color=color, fill_color=color, fill_opacity=0.75, stroke_width=2)
    r.move_to([BAR_X0, y, 0], aligned_edge=LEFT)
    return r


def chip(color):
    return Rectangle(width=0.35, height=0.2, color=color, fill_color=color, fill_opacity=0.75, stroke_width=1)


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


class UpdateRatio(Scene):
    def construct(self):
        title = Text("Update ratio", font_size=34).to_edge(UP, buff=0.3)
        formula = mt(r"\text{update ratio} = \frac{\text{lr}\cdot\|\nabla W\|}{\|W\|}"
                     r" = \frac{\text{size of the step}}{\text{size of the weights}}", 34).move_to([0, 2.55, 0])
        intuition = VGroup(
            Text("The absolute size of a step says little by itself.", font_size=22, color=GREY_B),
            Text("The same step is negligible for large weights and large for small weights.", font_size=20, color=GREY_B),
            Text("The ratio divides the step size by the size of the weights.", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.1).move_to([0, 1.4, 0])
        self.play(Write(title))
        self.play(Write(formula), run_time=2.0)
        self.play(FadeIn(intuition))

        w_bar = bar(0.4, BAR_W, BLUE_C)
        w_lab = mt(r"\|W\|", 30, BLUE_C).next_to(w_bar, LEFT, buff=0.3)
        s_lab = mt(r"\text{lr}\cdot\|\nabla W\|", 30).move_to([BAR_X0 - 1.0, -0.7, 0])
        legend = VGroup(
            VGroup(chip(BLUE_C), Text("upper bar = size of the weights", font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.15),
            VGroup(VGroup(chip(GREY_B), chip(GREEN_C), chip(RED_C)).arrange(RIGHT, buff=0.06),
                   Text("lower bar = size of this step  (grey: too small · green: healthy · red: too big)",
                        font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([0, -2.55, 0])
        self.play(FadeIn(w_bar), FadeIn(w_lab), FadeIn(s_lab), FadeIn(legend))

        cases = [
            (0.003, GREY_B, "ratio ≈ 0  →  the step is negligible: nothing is learned"),
            (0.05, GREEN_C, "ratio 0.01 – 0.1  →  healthy: steady progress without overshooting"),
            (1.0, RED_C, "ratio ≈ 1  →  the step is as big as the weights themselves: unstable"),
        ]
        s_bar = bar(-0.7, 0.0, WHITE)
        s_bar.set_opacity(0)
        self.add(s_bar)
        caption = Text("", font_size=24).to_edge(DOWN, buff=0.35)
        read = Text("", font_size=26).move_to([0, -1.5, 0])
        self.add(caption, read)
        for ratio, color, text in cases:
            new_bar = bar(-0.7, ratio * BAR_W, color)
            new_read = Text(f"ratio = {ratio:g}", font_size=26, color=color).move_to([0, -1.5, 0])
            new_cap = Text(text, font_size=24, color=color).to_edge(DOWN, buff=0.35)
            self.play(ReplacementTransform(s_bar, new_bar), Transform(read, new_read),
                      Transform(caption, new_cap), run_time=1.5)
            s_bar = new_bar
            self.wait(2.2)
        self.wait(1.0)

        epoch_slide(
            self, "Update ratio across epochs",
            "one point per logged epoch and layer — is learning moving at a sensible pace?",
            [0, 4, 0.5], [(0.5, "0.001"), (1.5, "0.01"), (2.5, "0.1"), (3.5, "1")], "update ratio (log scale)",
            [
                (lambda t: np.log10(0.06 * np.exp(-0.5 * t) + 0.015) + 3.5, GREEN_C, "healthy",
                 "healthy: stays inside the 0.01 – 0.1 band — steady steps"),
                (lambda t: np.log10(0.004 * np.exp(-0.8 * t)) + 3.5, ORANGE, "too small",
                 "too small: ≈ 0.001 and falling — steps are negligible, learning has stalled"),
                (lambda t: np.log10(0.35 * np.exp(0.6 * t) * (1 + 0.12 * np.sin(7 * t))) + 3.5, RED_C, "too big",
                 "too big: ≈ 1 and irregular — each step overshoots, the loss oscillates or diverges"),
            ],
            band=(1.5, 2.5, GREEN_C, "healthy zone 0.01 – 0.1"),
        )
