from manim import *


def loss(w):
    return (w - 2.0) ** 2 + 0.5


def grad(w):
    return 2.0 * (w - 2.0)


RUNS = [
    (0.05, 8, "lr = 0.05  — too small: converges very slowly", BLUE_C),
    (0.30, 5, "lr = 0.30  — good: converges in a few steps", GREEN_C),
    (1.05, 4, "lr = 1.05  — too large: overshoots further every step", RED_C),
]


class GradientDescentStep(Scene):
    def construct(self):
        title = Text("Gradient descent:  W ← W − lr · ∇W", font_size=34).to_edge(UP, buff=0.3)
        axes = Axes(x_range=[-0.5, 4.5, 1], y_range=[0, 6, 1], x_length=8, y_length=4.2,
                    tips=False).shift(DOWN * 0.4)
        curve = axes.plot(loss, x_range=[-0.4, 4.4], color=WHITE)
        x_lab = Text("weight w", font_size=20).next_to(axes, DOWN, buff=0.15)
        y_lab = Text("loss E", font_size=20).next_to(axes, LEFT, buff=0.15).rotate(PI / 2)
        minimum = Dot(axes.c2p(2.0, loss(2.0)), color=YELLOW, radius=0.07)
        min_tag = Text("minimum", font_size=18, color=YELLOW).next_to(minimum, UP, buff=0.4)
        self.play(Write(title), Create(axes), Create(curve), FadeIn(x_lab), FadeIn(y_lab))
        self.play(FadeIn(minimum), FadeIn(min_tag))

        caption = Text("", font_size=24).to_edge(DOWN, buff=0.3)
        self.add(caption)
        info = Text("", font_size=22).next_to(title, DOWN, buff=0.25)
        self.add(info)

        for lr, n_steps, text, color in RUNS:
            w = 0.2
            dot = Dot(axes.c2p(w, loss(w)), color=color, radius=0.12)
            trail = []
            self.play(Transform(caption, Text(text, font_size=24, color=color).to_edge(DOWN, buff=0.3)),
                      FadeIn(dot))
            for _ in range(n_steps):
                g = grad(w)
                w_new = w - lr * g
                if not (-0.4 <= w_new <= 4.4):
                    break
                half = 0.4
                tangent = Line(axes.c2p(w - half, loss(w) - g * half),
                               axes.c2p(w + half, loss(w) + g * half), color=YELLOW, stroke_width=3)
                new_info = Text(f"w = {w:.2f}   slope = {g:+.2f}   step = −lr·slope = {-lr * g:+.2f}",
                                font_size=22).next_to(title, DOWN, buff=0.25)
                self.play(Create(tangent), Transform(info, new_info), run_time=0.3)
                self.play(dot.animate.move_to(axes.c2p(w_new, loss(w_new))), FadeOut(tangent), run_time=0.45)
                ghost = Dot(axes.c2p(w_new, loss(w_new)), color=color, radius=0.05)
                trail.append(ghost)
                self.add(ghost)
                w = w_new
            self.wait(0.8)
            self.play(FadeOut(dot), *[FadeOut(t) for t in trail])

        outro = Text("This step size, relative to the weights, is the update_ratio plot  (healthy: 0.01 – 0.1)",
                     font_size=22).to_edge(DOWN, buff=0.3)
        self.play(Transform(caption, outro), FadeOut(info))
        self.wait(2.0)
