from manim import *

TRUE_GRADS = [0.30, 0.25, 0.22]
SCALE = 4.0
BASE_Y = -2.0
BAR_W = 1.6


def bar(x, value, color):
    r = Rectangle(width=BAR_W, height=max(value * SCALE, 0.02), color=color, fill_color=color, fill_opacity=0.7)
    r.move_to([x, BASE_Y, 0], aligned_edge=DOWN)
    return r


class ZeroGradAccumulation(Scene):
    def construct(self):
        title = Text("Why optimizer.zero_grad()?", font_size=36).to_edge(UP, buff=0.3)
        xl, xr = -3.5, 3.5
        head_l = Text("WITHOUT zero_grad()", font_size=26, color=RED_C).move_to([xl, 1.8, 0])
        head_r = Text("WITH zero_grad()", font_size=26, color=GREEN_C).move_to([xr, 1.8, 0])
        base_l = Line([xl - 1.5, BASE_Y, 0], [xl + 1.5, BASE_Y, 0], color=GREY_B)
        base_r = Line([xr - 1.5, BASE_Y, 0], [xr + 1.5, BASE_Y, 0], color=GREY_B)
        self.play(Write(title), FadeIn(head_l), FadeIn(head_r), Create(base_l), Create(base_r))

        bar_l = bar(xl, 0, RED_C)
        bar_r = bar(xr, 0, GREEN_C)
        tag_l = Text(".grad = 0.00", font_size=22).move_to([xl, BASE_Y - 0.4, 0])
        tag_r = Text(".grad = 0.00", font_size=22).move_to([xr, BASE_Y - 0.4, 0])
        self.add(bar_l, bar_r, tag_l, tag_r)

        stored_l = 0.0
        for e, g in enumerate(TRUE_GRADS, start=1):
            epoch = Text(f"epoch {e}:  true gradient = {g:.2f}", font_size=28, color=YELLOW).move_to([0, 2.65, 0])
            self.play(FadeIn(epoch), run_time=0.4)

            new_bar_r0 = bar(xr, 0, GREEN_C)
            new_tag_r0 = Text(".grad = 0.00  (cleared)", font_size=22).move_to([xr, BASE_Y - 0.4, 0])
            self.play(ReplacementTransform(bar_r, new_bar_r0), Transform(tag_r, new_tag_r0), run_time=0.6)
            bar_r = new_bar_r0

            stored_l += g
            new_bar_l = bar(xl, stored_l, RED_C)
            new_tag_l = Text(f".grad = {stored_l:.2f}  (stored)", font_size=22).move_to([xl, BASE_Y - 0.4, 0])
            new_bar_r = bar(xr, g, GREEN_C)
            new_tag_r = Text(f".grad = {g:.2f}  (stored)", font_size=22).move_to([xr, BASE_Y - 0.4, 0])
            self.play(ReplacementTransform(bar_l, new_bar_l), Transform(tag_l, new_tag_l),
                      ReplacementTransform(bar_r, new_bar_r), Transform(tag_r, new_tag_r), run_time=0.9)
            bar_l, bar_r = new_bar_l, new_bar_r

            verdict_l = Text("correct" if e == 1 else f"WRONG (true = {g:.2f})", font_size=22,
                             color=GREEN_C if e == 1 else RED_C).move_to([xl, BASE_Y - 0.9, 0])
            verdict_r = Text("correct", font_size=22, color=GREEN_C).move_to([xr, BASE_Y - 0.9, 0])
            self.play(FadeIn(verdict_l), FadeIn(verdict_r), run_time=0.4)
            self.wait(1.0)
            self.play(FadeOut(epoch), FadeOut(verdict_l), FadeOut(verdict_r), run_time=0.4)

        final = Text("backward() ADDS to .grad — zero_grad() makes every epoch start from zero",
                     font_size=24).to_edge(DOWN, buff=0.3)
        self.play(Write(final))
        self.wait(2.0)
