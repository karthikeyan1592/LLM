import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")


def mt(s, size=34, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def legend_entry(color, label):
    return VGroup(Line(ORIGIN, RIGHT * 0.7, color=color, stroke_width=6),
                  Text(label, font_size=22, color=color)).arrange(RIGHT, buff=0.2)


class WeightNorm(Scene):
    def construct(self):
        title = Text("Weight norm  ‖W‖", font_size=34).to_edge(UP, buff=0.3)
        general = mt(r"\|W\| = \sqrt{\sum_{i,j} w_{ij}^2}", 38).move_to([0, 2.3, 0])
        meaning = Text("one number: how large this layer's learned parameters have grown",
                       font_size=22, color=GREY_B).move_to([0, 1.5, 0])

        s1 = mt(r"W = \begin{bmatrix}1&2\\2&4\end{bmatrix}", 32)
        s2 = mt(r"\text{square each entry:}\ \begin{bmatrix}1&4\\4&16\end{bmatrix}", 32)
        s3 = mt(r"\text{add them up:}\ 1+4+4+16 = 25", 32)
        s4 = mt(r"\|W\| = \sqrt{25} = 5", 38, YELLOW)
        stack = VGroup(s1, s2, s3, s4).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, -1.35, 0])

        self.play(Write(title))
        self.play(Write(general), run_time=1.5)
        self.play(FadeIn(meaning))
        self.wait(0.8)
        for s in (s1, s2, s3, s4):
            self.play(Write(s), run_time=1.2)
            self.wait(0.5)
        self.wait(1.2)
        self.play(FadeOut(VGroup(general, meaning, stack)))

        intuition = VGroup(
            Text("Meaning", font_size=30, color=YELLOW),
            Text("W is the matrix of all the weights of this layer.", font_size=26),
            Text("‖W‖ is one number for the overall size of these weights.", font_size=26),
            Text("Training should change them gradually:", font_size=26, color=YELLOW),
            Text("slow, steady change  →  the layer is learning", font_size=24, color=GREEN_C),
            Text("no change  →  not being updated        very fast growth  →  unstable", font_size=24, color=ORANGE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([0, -0.2, 0])
        for line in intuition:
            self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.9)
            self.wait(0.6)
        self.wait(2.0)
        self.play(FadeOut(intuition))

        axes = Axes(x_range=[0, 2.4, 0.4], y_range=[0, 3, 1], x_length=8.4, y_length=4.0,
                    tips=False).shift(LEFT * 1.5 + DOWN * 0.3)
        x_lab = Text("epoch", font_size=20).next_to(axes, DOWN, buff=0.15)
        y_lab = mt(r"\|W\|", 28).next_to(axes, LEFT, buff=0.2)
        header = Text("Three possible runs\nfor this layer:", font_size=20, color=GREY_B)
        cases = [
            (lambda t: 1.0 + 0.3 * t, GREEN_C, "healthy",
             "healthy: slow, steady growth — the weights are being updated"),
            (lambda t: 1.0 + 0 * t, ORANGE, "frozen",
             "frozen: a flat line — the weights are not changing"),
            (lambda t: 0.5 * np.exp(0.75 * t), RED_C, "diverging",
             "diverging: rapid growth — the weights are running away"),
        ]
        entries = [legend_entry(c[1], c[2]) for c in cases]
        legend = VGroup(header, *entries).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([5.2, 0.7, 0])
        self.play(Create(axes), FadeIn(x_lab), FadeIn(y_lab), FadeIn(header))
        caption = Text("", font_size=24).to_edge(DOWN, buff=0.35)
        self.add(caption)
        for (f, color, _, text), entry in zip(cases, entries):
            curve = axes.plot(f, x_range=[0, 2.4], color=color, stroke_width=5)
            self.play(Transform(caption, Text(text, font_size=24, color=color).to_edge(DOWN, buff=0.35)),
                      Create(curve), FadeIn(entry), run_time=2.0)
            self.wait(1.0)
        self.wait(1.5)
