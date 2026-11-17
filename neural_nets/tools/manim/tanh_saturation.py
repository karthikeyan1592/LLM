import numpy as np
from manim import *

SAT = float(np.arctanh(0.95))


class TanhSaturation(Scene):
    def construct(self):
        title = Text("Why saturated tanh units stop learning", font_size=34).to_edge(UP, buff=0.3)
        axes = Axes(x_range=[-4, 4, 1], y_range=[-1.2, 1.2, 0.5], x_length=9, y_length=4.0,
                    tips=False).shift(DOWN * 0.5)
        tanh_curve = axes.plot(np.tanh, x_range=[-4, 4], color=BLUE_C)
        deriv_curve = axes.plot(lambda a: 1 - np.tanh(a) ** 2, x_range=[-4, 4], color=RED_C)
        legend = VGroup(Text("tanh(a)  = z", font_size=20, color=BLUE_C),
                        Text("1 − tanh²(a)  = how much error passes back", font_size=20, color=RED_C)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(UL, buff=0.4).shift(DOWN * 0.7)
        x_lab = Text("pre-activation a", font_size=20).next_to(axes, DOWN, buff=0.15)

        def zone(x0, x1):
            p0, p1 = axes.c2p(x0, -1.2), axes.c2p(x1, 1.2)
            r = Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1], stroke_width=0,
                          fill_color=ORANGE, fill_opacity=0.18)
            r.move_to((p0 + p1) / 2)
            return r

        zones = VGroup(zone(SAT, 4), zone(-4, -SAT))
        zone_tag = Text("|z| > 0.95\ncounted by\nsat_frac", font_size=16, color=ORANGE
                        ).move_to(axes.c2p(-2.9, 0.55))

        self.play(Write(title), Create(axes), FadeIn(x_lab))
        self.play(Create(tanh_curve), Create(deriv_curve), FadeIn(legend))
        self.play(FadeIn(zones), FadeIn(zone_tag))

        a = ValueTracker(0.0)
        dot_t = always_redraw(lambda: Dot(axes.c2p(a.get_value(), np.tanh(a.get_value())), color=BLUE_C, radius=0.1))
        dot_d = always_redraw(lambda: Dot(axes.c2p(a.get_value(), 1 - np.tanh(a.get_value()) ** 2),
                                          color=RED_C, radius=0.1))
        guide = always_redraw(lambda: DashedLine(axes.c2p(a.get_value(), -1.2), axes.c2p(a.get_value(), 1.2),
                                                 color=GREY_B, stroke_width=2))

        def readout():
            v = a.get_value()
            t, d = np.tanh(v), 1 - np.tanh(v) ** 2
            return Text(f"a = {v:.2f}    z = {t:.3f}    error passed back = {d * 100:.0f}%",
                        font_size=24, color=YELLOW).to_edge(DOWN, buff=0.35)

        info = always_redraw(readout)
        self.add(guide, dot_t, dot_d, info)
        self.play(a.animate.set_value(0.7), run_time=2.0)
        self.wait(0.5)
        self.play(a.animate.set_value(SAT), run_time=2.5)
        self.wait(1.0)
        self.play(a.animate.set_value(3.2), run_time=1.5)
        self.wait(1.0)

        self.remove(info)
        chain = Text("3 saturated layers:  0.097³ ≈ 0.001  of the error reaches layer 1",
                     font_size=24, color=ORANGE).to_edge(DOWN, buff=0.35)
        self.play(Write(chain))
        self.wait(2.5)
