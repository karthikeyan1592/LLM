import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")


def mt(s, size=34, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def legend_entry(color, label):
    return VGroup(Line(ORIGIN, RIGHT * 0.7, color=color, stroke_width=6),
                  Text(label, font_size=22, color=color)).arrange(RIGHT, buff=0.2)


class GradientNorm(Scene):
    def construct(self):
        title = Text("Gradient norm  ‖∇W‖", font_size=34).to_edge(UP, buff=0.3)
        general = mt(r"\|\nabla W\| = \sqrt{\sum_{i,j}\left(\frac{\partial E}{\partial w_{ij}}\right)^2}", 36)
        general.move_to([0, 2.3, 0])
        meaning = Text("one number: how strongly the loss depends on this layer's weights",
                       font_size=22, color=GREY_B).move_to([0, 1.5, 0])

        s1 = mt(r"\nabla W = \begin{bmatrix}1&-2\\2&0\end{bmatrix}", 32)
        s2 = mt(r"\text{square each entry:}\ \begin{bmatrix}1&4\\4&0\end{bmatrix}", 32)
        s3 = mt(r"\text{add them up:}\ 1+4+4+0 = 9", 32)
        s4 = mt(r"\|\nabla W\| = \sqrt{9} = 3", 38, YELLOW)
        stack = VGroup(s1, s2, s3, s4).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, -1.35, 0])

        self.play(Write(title))
        self.play(Write(general), run_time=1.8)
        self.play(FadeIn(meaning))
        self.wait(0.8)
        for s in (s1, s2, s3, s4):
            self.play(Write(s), run_time=1.2)
            self.wait(0.5)
        self.wait(1.2)
        self.play(FadeOut(VGroup(general, meaning, stack)))

        intuition = VGroup(
            Text("Meaning", font_size=30, color=YELLOW),
            Text("∇W holds the derivative of the loss with respect to each weight.", font_size=26),
            Text("The norm combines them into one number:", font_size=26),
            Text("the magnitude of the gradient, which sets the size of the optimizer step.", font_size=26, color=YELLOW),
            Text("large  →  the loss still changes quickly with these weights", font_size=24, color=GREEN_C),
            Text("≈ 0  →  the loss hardly depends on these weights (vanishing gradient)", font_size=24, color=ORANGE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([0, -0.2, 0])
        for line in intuition:
            self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.9)
            self.wait(0.6)
        self.wait(2.0)
        self.play(FadeOut(intuition))

        axes = Axes(x_range=[0, 2.4, 0.4], y_range=[0, 3, 1], x_length=8.4, y_length=4.0,
                    tips=False).shift(LEFT * 1.5 + DOWN * 0.3)
        x_lab = Text("epoch", font_size=20).next_to(axes, DOWN, buff=0.15)
        y_lab = mt(r"\|\nabla W\|", 28).next_to(axes, LEFT, buff=0.2)
        header = Text("Three possible runs\nfor this layer:", font_size=20, color=GREY_B)
        cases = [
            (lambda t: 1.4 * np.exp(-1.2 * t) + 0.15, GREEN_C, "healthy",
             "healthy: shrinks steadily as the loss falls"),
            (lambda t: 0.06 * np.exp(-1.0 * t), ORANGE, "vanishing",
             "vanishing: ≈ 0 from the start — the layer is not learning"),
            (lambda t: 0.25 * np.exp(1.0 * t), RED_C, "exploding",
             "exploding: keeps growing — training will diverge"),
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
