from manim import *

X0 = [0.5348, 0.7438, -1.0427]
Z1 = [0.3005, 0.5153, -0.3102]
Z2 = [-0.1853, 0.3145]
Z3 = [0.4371]
D1 = [-0.0509, -0.0186, -0.0371]
D2 = [-0.1375, -0.0327]
D3 = [0.4371]
LR = 0.1
FWD, BWD, UPD = BLUE_C, RED_C, YELLOW
STEPS = ["1 zero_grad()", "2 forward", "3 loss", "4 backward()", "5 step()"]


class TrainingStep(Scene):
    def construct(self):
        title = Text("One training epoch  (3 → 3 → 2 → 1 MLP)", font_size=32).to_edge(UP, buff=0.25)
        step_row = VGroup(*[Text(s, font_size=20, color=GREY_B) for s in STEPS])
        step_row.arrange(RIGHT, buff=0.55).next_to(title, DOWN, buff=0.25)

        xs = [-4.8, -1.6, 1.6, 4.8]
        sizes = [3, 3, 2, 1]
        values = [X0, Z1, Z2, Z3]
        circles, labels = [], []
        for x, n, vals in zip(xs, sizes, values):
            ys = [(n - 1) / 2 * 1.4 - i * 1.4 for i in range(n)]
            cs = VGroup(*[Circle(radius=0.42, color=GREY_B, stroke_width=3).move_to([x, y - 0.3, 0]) for y in ys])
            ls = VGroup(*[Text(f"{v:.3f}", font_size=15).move_to(c) for v, c in zip(vals, cs)])
            circles.append(cs)
            labels.append(ls)

        edges = []
        for l in range(3):
            edges.append([Line(a.get_right(), b.get_left(), stroke_width=1.5, color=GREY_D)
                          for a in circles[l] for b in circles[l + 1]])

        layer_names = ["z(0) = x", "z(1)", "z(2)", "z(3) = p"]
        layer_tags = VGroup(*[Text(t, font_size=20, color=GREY_B).move_to([x, -2.45, 0])
                              for x, t in zip(xs, layer_names)])
        caption = Text("", font_size=22).to_edge(DOWN, buff=0.35)

        def set_step(i, text):
            new_caption = Text(text, font_size=22).to_edge(DOWN, buff=0.35)
            anims = [t.animate.set_color(YELLOW if j == i else GREY_B) for j, t in enumerate(step_row)]
            anims.append(Transform(caption, new_caption))
            return anims

        self.play(Write(title), FadeIn(step_row))
        self.play(*[Create(c) for c in circles], *[Create(e) for g in edges for e in g], FadeIn(layer_tags))
        self.play(FadeIn(labels[0]))
        self.add(caption)

        self.play(*set_step(0, "Clear last epoch's .grad  (PyTorch accumulates by default)"))
        self.wait(1.0)

        self.play(*set_step(1, "Forward: a(l) = W(l)·z(l-1) + b(l),  z(l) = activation(a(l))"))
        for l in range(3):
            flashes = [ShowPassingFlash(e.copy().set_color(FWD).set_stroke(width=5), time_width=0.7) for e in edges[l]]
            self.play(*flashes, run_time=0.9)
            self.play(FadeIn(labels[l + 1]), *[c.animate.set_color(FWD) for c in circles[l + 1]], run_time=0.5)
        self.wait(0.5)

        self.play(*set_step(2, "Loss: p = 0.437, y = 0  →  BCE = −log(1 − p) = 0.575"))
        self.play(Indicate(circles[3][0], color=ORANGE, scale_factor=1.4))
        self.wait(1.0)

        self.play(*set_step(3, "Backward: δ(3) = p − y,   δ(l) = W(l+1)ᵀ·δ(l+1) ⊙ (1 − z²)"))
        deltas = [D1, D2, D3]
        delta_tags = []
        for l in (2, 1, 0):
            flashes = [ShowPassingFlash(Line(e.get_end(), e.get_start()).set_color(BWD).set_stroke(width=5),
                                        time_width=0.7) for e in edges[l]]
            self.play(*flashes, run_time=0.9)
            tags = VGroup(*[Text(f"δ={d:+.3f}", font_size=14, color=BWD).next_to(c, UP, buff=0.06)
                            for d, c in zip(deltas[l], circles[l + 1])])
            delta_tags.append(tags)
            self.play(FadeIn(tags), *[c.animate.set_color(BWD) for c in circles[l + 1]], run_time=0.5)
        self.wait(0.5)

        self.play(*set_step(4, "step: W ← W − lr·∇W  (lr = 0.1) — the ONLY line that changes weights"))
        widths = [3.5, 1.0, 4.5, 2.0, 0.8, 3.0, 1.5, 4.0, 2.5]
        anims = []
        for l in range(3):
            for k, e in enumerate(edges[l]):
                anims.append(e.animate.set_color(UPD).set_stroke(width=widths[(k + l) % len(widths)]))
        self.play(*anims, run_time=1.5)
        self.wait(0.8)

        self.play(*[FadeOut(t) for t in delta_tags],
                  *[c.animate.set_color(GREY_B) for g in circles for c in g],
                  *[e.animate.set_color(GREY_D) for g in edges for e in g])
        self.play(*set_step(0, "Next epoch: the loop repeats with the updated weights — loss goes down"))
        self.wait(1.5)
