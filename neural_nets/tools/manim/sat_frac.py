import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
A = np.linspace(-2.2, 2.2, 16)


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


class SaturatedFraction(Scene):
    def construct(self):
        title = Text("Saturated fraction  sat_frac", font_size=34).to_edge(UP, buff=0.3)
        formula = mt(r"\text{sat\_frac} = \frac{\#\{k:\ |z_k| > 0.95\}}{n}", 36).move_to([0, 2.3, 0])
        intuition = VGroup(
            Text("tanh is nearly flat for |a| > 1.83: its derivative 1 − z² is close to 0 there,", font_size=22, color=GREY_B),
            Text("so almost no gradient flows through the neuron.  sat_frac = the fraction of outputs with |z| > 0.95.",
                 font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.12).move_to([0, 1.3, 0])
        line = NumberLine(x_range=[-1, 1, 0.5], length=10, include_numbers=False).move_to([0, -0.3, 0])
        ticks = VGroup(*[Text(t, font_size=20, color=GREY_B).next_to(line.n2p(v), DOWN, buff=0.2)
                         for t, v in (("-1", -1), ("0", 0), ("1", 1))])

        def zone(a, b):
            x0, x1 = line.n2p(a)[0], line.n2p(b)[0]
            r = Rectangle(width=x1 - x0, height=1.6, stroke_width=0, fill_color=ORANGE, fill_opacity=0.25)
            r.move_to([(x0 + x1) / 2, -0.3, 0])
            return r

        zones = VGroup(zone(0.95, 1.0), zone(-1.0, -0.95))
        zone_tag = Text("|z| > 0.95", font_size=18, color=ORANGE).move_to([line.n2p(0.97)[0], 0.7, 0])
        zone_tag2 = Text("|z| > 0.95", font_size=18, color=ORANGE).move_to([line.n2p(-0.97)[0], 0.7, 0])
        zone_chip = Rectangle(width=0.2, height=0.35, stroke_width=0, fill_color=ORANGE, fill_opacity=0.4)
        legend = VGroup(
            VGroup(Dot(color=BLUE_C, radius=0.09), Text("blue dot = active neuron", font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=ORANGE, radius=0.09), Text("orange dot = saturated neuron", font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.15),
            VGroup(zone_chip, Text("zone = flat ends of tanh", font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.5).move_to([0, -2.7, 0])

        self.play(Write(title), Write(formula), run_time=1.8)
        self.play(FadeIn(intuition), Create(line), FadeIn(ticks), FadeIn(zones), FadeIn(zone_tag), FadeIn(zone_tag2),
                  FadeIn(legend))

        gain = ValueTracker(0.35)

        def zs():
            return np.tanh(gain.get_value() * A)

        def dots():
            z = zs()
            return VGroup(*[Dot(line.n2p(v) + UP * 0.32 * ((k % 3) - 1), radius=0.09,
                                color=ORANGE if abs(v) > 0.95 else BLUE_C)
                            for k, v in enumerate(z)])

        def readout():
            z = zs()
            k = int(np.sum(np.abs(z) > 0.95))
            return Text(f"sat_frac = {k}/16 = {k / 16:.2f}", font_size=30, color=YELLOW).move_to([0, -1.8, 0])

        self.add(always_redraw(dots), always_redraw(readout))
        caption = Text("healthy: sat_frac near 0 — neurons are active and pass gradient",
                       font_size=24, color=GREEN_C).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))
        self.wait(2.0)

        self.play(Transform(caption, Text("as pre-activations grow, tanh outputs move toward ±1 …",
                                          font_size=24, color=ORANGE).to_edge(DOWN, buff=0.35)))
        self.play(gain.animate.set_value(1.4), run_time=3.0)
        self.wait(1.2)
        self.play(Transform(caption, Text("sat_frac near 1: most neurons are saturated — derivative 1 − z² ≈ 0, gradients vanish",
                                          font_size=22, color=RED_C).to_edge(DOWN, buff=0.35)))
        self.play(gain.animate.set_value(4.5), run_time=3.0)
        self.wait(2.5)

        epoch_slide(
            self, "Saturated fraction across epochs",
            "one point per logged epoch and layer — what fraction of the neurons is saturated?",
            [0, 1.1, 0.25], [(0, "0"), (0.5, "0.5"), (1, "1")], "sat_frac",
            [
                (lambda t: 0.03 + 0.015 * np.sin(4 * t), GREEN_C, "healthy",
                 "healthy: stays near 0 — almost every neuron is still active"),
                (lambda t: 0.03 + 0.5 * (t / 2.4) ** 1.5, ORANGE, "rising slowly",
                 "rising slowly: more neurons saturate over time — an early warning"),
                (lambda t: 1 - 0.95 * np.exp(-2.2 * t), RED_C, "saturated",
                 "saturated: ≈ 1 — almost every neuron is saturated, gradients vanish"),
            ],
            band=(0.5, 1.1, RED_C, "danger: over half saturated"),
        )
