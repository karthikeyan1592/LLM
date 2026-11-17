import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
BASE = np.array([-0.85, -0.7, -0.55, -0.4, -0.25, -0.1, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8])
SIGNS = np.array([1, -1] * 6)

PANELS = [
    ("A  healthy: spread out, in the middle", BASE.copy(), GREEN_C,
     "both fine: σ moderate, sat_frac ≈ 0"),
    ("B  collapsed in the middle", 0.3 + 0.03 * BASE, ORANGE,
     "σ ≈ 0 detects it; sat_frac stays 0 and does not"),
    ("C  saturated at both ends (±1)", SIGNS * (0.975 + 0.002 * np.arange(12)), RED_C,
     "sat_frac = 1 detects it; σ is high and does not"),
    ("D  saturated at one end (+1)", 0.985 + 0.0015 * BASE, RED_C,
     "both flag it: σ ≈ 0 and sat_frac = 1"),
]


def mt(s, size=34, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def make_panel(center, name, z, color, verdict):
    line = NumberLine(x_range=[-1, 1, 0.5], length=5.4, include_numbers=False).move_to(center)
    zones = VGroup(*[
        Rectangle(width=line.n2p(b)[0] - line.n2p(a)[0], height=0.9, stroke_width=0,
                  fill_color=ORANGE, fill_opacity=0.22).move_to([(line.n2p(a)[0] + line.n2p(b)[0]) / 2, center[1], 0])
        for a, b in ((0.95, 1.0), (-1.0, -0.95))])
    dots = VGroup(*[Dot(line.n2p(v) + UP * 0.22 * ((k % 3) - 1), radius=0.075,
                        color=ORANGE if abs(v) > 0.95 else BLUE_C) for k, v in enumerate(z)])
    sigma = float(np.std(z, ddof=1))
    sat = float(np.mean(np.abs(z) > 0.95))
    title = Text(name, font_size=20, color=color).next_to(line, UP, buff=0.55)
    read = Text(f"σ = {sigma:.2f}      sat_frac = {sat:.2f}", font_size=22, color=YELLOW).next_to(line, DOWN, buff=0.3)
    note = Text(verdict, font_size=18, color=color).next_to(read, DOWN, buff=0.12)
    return VGroup(zones, line, dots), title, read, note


class ActStdVsSatFrac(Scene):
    def construct(self):
        title = Text("Activation std vs saturated fraction", font_size=34).to_edge(UP, buff=0.3)
        q1 = VGroup(mt(r"\sigma", 30, YELLOW),
                    Text("asks: how SPREAD OUT are the outputs?  (ignores where they sit)", font_size=21)
                    ).arrange(RIGHT, buff=0.25)
        q2 = VGroup(mt(r"\text{sat\_frac}", 30, ORANGE),
                    Text("asks: what FRACTION of the outputs is saturated, |z| > 0.95?  (ignores spread)",
                         font_size=21)).arrange(RIGHT, buff=0.25)
        qs = VGroup(q1, q2).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([0, 2.7, 0])
        legend = VGroup(
            VGroup(Dot(color=BLUE_C, radius=0.08), Text("blue dot = active neuron", font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.12),
            VGroup(Dot(color=ORANGE, radius=0.08), Text("orange dot = saturated (|z| > 0.95)", font_size=18, color=GREY_B)).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.6).move_to([0, 2.0, 0])

        self.play(Write(title))
        self.play(FadeIn(q1, shift=RIGHT * 0.2), run_time=1.0)
        self.wait(1.2)
        self.play(FadeIn(q2, shift=RIGHT * 0.2), run_time=1.0)
        self.play(FadeIn(legend))
        self.wait(1.5)

        centers = [[-3.6, 0.7, 0], [3.6, 0.7, 0], [-3.6, -2.0, 0], [3.6, -2.0, 0]]
        made = []
        for c, (name, z, color, verdict) in zip(centers, PANELS):
            graphic, ttl, read, note = make_panel(c, name, z, color, verdict)
            made.append((graphic, ttl, read, note))
            self.play(FadeIn(ttl), FadeIn(graphic), run_time=0.8)
            self.play(FadeIn(read), run_time=0.6)
            self.play(FadeIn(note), run_time=0.8)
            self.wait(1.6)

        self.play(FadeOut(qs), FadeOut(legend))
        summary = VGroup(
            Text("σ answers: do the outputs differ from each other?", font_size=22, color=YELLOW),
            Text("sat_frac answers: are the neurons saturated (|z| > 0.95)?", font_size=22, color=ORANGE),
            Text("C is invisible to σ, B is invisible to sat_frac  —  so we track both.", font_size=22),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([0, 2.55, 0])
        self.play(FadeIn(summary, shift=DOWN * 0.2))
        self.wait(4.0)
