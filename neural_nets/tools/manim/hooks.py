import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")
TAP = YELLOW
OK_C, BAD_C = GREEN_C, RED_C
TRAIN_C, VAL_C = BLUE_C, ORANGE
MODS = ["Linear 1", "Tanh", "Linear 2", "Tanh", "Linear 3", "Sigmoid"]


def mt(s, size=30, color=WHITE):
    return MathTex(s, tex_template=TEX, font_size=size, color=color)


def mono(s, size=20, color=WHITE):
    return Text(s, font="Monospace", font_size=size, color=color)


def char_w(size):
    return Text("M" * 20, font="Monospace", font_size=size).width / 20


def code_block(lines, size=18, color=WHITE, buff=0.13):
    cw = char_w(size)
    items = []
    for ln in lines:
        stripped = ln.lstrip(" ")
        items.append((mono(stripped, size, color), len(ln) - len(stripped)))
    g = VGroup(*[t for t, _ in items]).arrange(DOWN, aligned_edge=LEFT, buff=buff)
    for t, indent in items:
        t.shift(RIGHT * cw * indent)
    return g


def para(s, size=22, color=WHITE, width=None):
    t = Text(s, font_size=size, color=color, line_spacing=0.85)
    if width is not None and t.width > width:
        t.scale_to_fit_width(width)
    return t


class Chain:
    pass


def build_chain(y):
    ch = Chain()
    ch.boxes = []
    for n in MODS:
        act = n in ("Tanh", "Sigmoid")
        rect = RoundedRectangle(width=1.5, height=0.8, corner_radius=0.12,
                                color=BLUE_C if act else GREY_B, stroke_width=3)
        ch.boxes.append(VGroup(rect, Text(n, font_size=20)))
    row = VGroup(*ch.boxes).arrange(RIGHT, buff=0.5).move_to([0, y, 0])
    arrows = [Arrow(ch.boxes[i].get_right(), ch.boxes[i + 1].get_left(), buff=0.03, stroke_width=3,
                    color=GREY_B, max_tip_length_to_length_ratio=0.35) for i in range(5)]
    ch.x_lab = mt("x", 28).move_to([-6.55, y, 0])
    ch.y_lab = mt("y", 28).move_to([6.55, y, 0])
    ch.x_arr = Arrow(ch.x_lab.get_right() + RIGHT * 0.05, ch.boxes[0].get_left(), buff=0.03, stroke_width=3,
                     color=GREY_B, max_tip_length_to_length_ratio=0.35)
    ch.y_arr = Arrow(ch.boxes[5].get_right(), ch.y_lab.get_left() + LEFT * 0.05, buff=0.03, stroke_width=3,
                     color=GREY_B, max_tip_length_to_length_ratio=0.35)
    ch.arrows = arrows
    ch.acts = [ch.boxes[1], ch.boxes[3], ch.boxes[5]]
    ch.all = VGroup(row, *arrows, ch.x_lab, ch.y_lab, ch.x_arr, ch.y_arr)
    return ch


def make_tap(box, idx, color=TAP):
    top = box.get_bottom()
    stem = Line(top, top + DOWN * 0.42, color=color, stroke_width=3)
    dot = Dot(top + DOWN * 0.55, radius=0.13, color=color)
    lab = Text(f"idx {idx}", font_size=16, color=color).next_to(dot, RIGHT, buff=0.12)
    return VGroup(stem, dot, lab)


def make_slot(x, y, idx, w=1.7, h=0.6):
    rect = Rectangle(width=w, height=h, color=GREY_B, stroke_width=2).move_to([x, y, 0])
    lab = Text(f"key {idx}", font_size=16, color=GREY_B).next_to(rect, UP, buff=0.06)
    return VGroup(rect, lab)


class HookBasics(Scene):
    def construct(self):
        title = Text("Forward hooks — reading intermediate outputs", font_size=32).to_edge(UP, buff=0.25)
        ch = build_chain(1.6)
        self.play(Write(title))
        self.play(FadeIn(ch.all), run_time=1.4)

        # --- the problem ---------------------------------------------------------
        dot = Dot(color=YELLOW, radius=0.1).move_to(ch.x_lab.get_right() + RIGHT * 0.15)
        self.add(dot)
        self.play(dot.animate.move_to(ch.y_lab.get_left() + LEFT * 0.15), run_time=2.4, rate_func=linear)
        self.play(FadeOut(dot))
        q = para("model(x) runs every module in turn but hands back only y.", 24).move_to([0, 0.35, 0])
        q2 = para("We want the values in the middle: the output z of each activation.", 24, YELLOW).move_to([0, -0.15, 0])
        self.play(FadeIn(q))
        self.play(FadeIn(q2))
        self.wait(1.2)
        bad1 = VGroup(Text("BAD", font_size=20, color=BAD_C),
                      para("edit forward() to also return every z:  the model changes just so we can watch it", 20)
                      ).arrange(RIGHT, buff=0.3)
        bad2 = VGroup(Text("BAD", font_size=20, color=BAD_C),
                      para("recompute the layers by hand:  duplicated logic that can drift from the real model", 20)
                      ).arrange(RIGHT, buff=0.3)
        bads = VGroup(bad1, bad2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -1.5, 0])
        self.play(FadeIn(bad1, shift=RIGHT * 0.2))
        self.wait(0.8)
        self.play(FadeIn(bad2, shift=RIGHT * 0.2))
        self.wait(1.5)
        self.play(FadeOut(VGroup(q, q2, bads)))

        # --- the idea ------------------------------------------------------------
        idea1 = para("A hook is a function you hand to PyTorch.", 26).move_to([0, 0.35, 0])
        idea2 = para("PyTorch calls it for you, automatically, at a fixed moment.", 26, YELLOW).move_to([0, -0.25, 0])
        idea3 = para("A forward hook is called right after a module's forward() has produced its output.", 22, GREY_B)
        idea3.move_to([0, -0.95, 0])
        self.play(FadeIn(idea1))
        self.play(FadeIn(idea2))
        self.play(FadeIn(idea3))
        self.wait(1.5)
        self.play(FadeOut(VGroup(idea1, idea2, idea3)))

        # --- anatomy -------------------------------------------------------------
        tanh1 = ch.acts[0]
        tap = make_tap(tanh1, 0)
        hook_box = VGroup(RoundedRectangle(width=3.4, height=0.7, corner_radius=0.1, color=TAP, stroke_width=3),
                          mono("hook(module, inp, out)", 17, TAP))
        hook_box.move_to(tanh1.get_bottom() + DOWN * 1.15)
        tapline = DashedLine(tap[1].get_bottom(), hook_box.get_top(), color=TAP, stroke_width=3)
        reg = code_block(["handle = act.register_forward_hook(hook)"], 20, TAP).move_to([0, -3.05, 0])
        self.play(Create(tap), Create(hook_box), Create(tapline), run_time=1.4)
        self.play(FadeIn(reg))
        self.wait(0.8)

        lab_in = Text("inp", font_size=20, color=GREEN_C).next_to(ch.arrows[0], UP, buff=0.55)
        lab_mod = Text("module", font_size=20, color=BLUE_C).next_to(tanh1, UP, buff=0.55)
        lab_out = Text("out", font_size=20, color=ORANGE).next_to(ch.arrows[1], UP, buff=0.55)
        notes = [
            (lab_in, ch.arrows[0], GREEN_C, "inp: a tuple of what the module received (here the pre-activation a)"),
            (lab_mod, tanh1, BLUE_C, "module: the Tanh object that just ran"),
            (lab_out, ch.arrows[1], ORANGE, "out: what forward() returned (here z)"),
        ]
        note_t = None
        ptrs = []
        for lab, target, col, txt in notes:
            ptr = Line(lab.get_bottom(), target.get_top(), color=col, stroke_width=2)
            ptrs.append(ptr)
            new = para(txt, 22, col).move_to([0, -1.15, 0])
            if note_t is None:
                self.play(FadeIn(lab), Create(ptr), FadeIn(new), run_time=1.2)
                note_t = new
            else:
                self.play(FadeIn(lab), Create(ptr), FadeOut(note_t), FadeIn(new), run_time=1.2)
                note_t = new
            self.wait(1.3)
        self.play(FadeOut(VGroup(lab_in, lab_mod, lab_out, note_t, *ptrs)))

        # --- call order ----------------------------------------------------------
        steps = [
            "1   forward() runs and produces out",
            "2   PyTorch calls hook(module, inp, out)",
            "3   the hook returns None, so out continues to the next module unchanged",
        ]
        step_t = [para(s, 22).move_to([0, -1.15 - 0.5 * i, 0]).align_to([-5.2, 0, 0], LEFT) for i, s in enumerate(steps)]
        d = Dot(color=YELLOW, radius=0.1).move_to(ch.arrows[0].get_start())
        self.add(d)
        self.play(d.animate.move_to(tanh1.get_center()), FadeIn(step_t[0]), run_time=1.6)
        copy = Dot(color=TAP, radius=0.1).move_to(tanh1.get_center())
        self.add(copy)
        self.play(copy.animate.move_to(hook_box.get_center()), FadeIn(step_t[1]),
                  Indicate(hook_box, color=TAP, scale_factor=1.08), run_time=1.8)
        self.play(FadeOut(copy), d.animate.move_to(ch.arrows[1].get_end()), FadeIn(step_t[2]), run_time=1.6)
        self.wait(1.2)
        self.play(FadeOut(d), FadeOut(VGroup(*step_t)))

        # --- rules ---------------------------------------------------------------
        r1 = VGroup(Text("returns None", font_size=22, color=OK_C),
                    para("the output is untouched, a hook only observes", 22)).arrange(RIGHT, buff=0.3)
        r2 = VGroup(Text("returns a value", font_size=22, color=BAD_C),
                    para("that value would REPLACE the module's output", 22)).arrange(RIGHT, buff=0.3)
        r3 = VGroup(Text("handle", font_size=22, color=TAP),
                    para("register_forward_hook gives one back; handle.remove() detaches the hook", 22)
                    ).arrange(RIGHT, buff=0.3)
        r4 = VGroup(Text("other hooks", font_size=22, color=GREY_B),
                    para("pre-hook sees the input; backward hook sees gradients (used for Grad-CAM)", 22, GREY_B)
                    ).arrange(RIGHT, buff=0.3)
        rules = VGroup(r1, r2, r3, r4).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -0.2, 0])
        if rules.width > 13:
            rules.scale_to_fit_width(13)
        self.play(FadeOut(VGroup(reg, tap, hook_box, tapline)), run_time=0.8)
        for r in (r1, r2, r3, r4):
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=1.0)
            self.wait(1.0)
        self.wait(2.0)


class HookInOurLoop(Scene):
    def construct(self):
        title = Text("Our hooks inside the training loop", font_size=32).to_edge(UP, buff=0.25)
        ch = build_chain(2.25)
        self.play(Write(title))
        self.play(FadeIn(ch.all), run_time=1.2)

        # --- our code, piece by piece -------------------------------------------
        lines = [
            "_act_store = {}",
            "def _make_fwd_hook(idx):",
            "    def hook(module, inp, out):",
            "        _act_store[idx] = out.detach().cpu()",
            "    return hook",
            "handles = []",
            "for i, act in enumerate(model.acts):",
            "    handles.append(",
            "        act.register_forward_hook(_make_fwd_hook(i)))",
        ]
        code = code_block(lines, 16).move_to([-3.05, -0.55, 0])
        self.play(FadeIn(code), run_time=1.0)
        groups = [
            ([0], "an empty dict that lives outside any function.\nThe hook writes into it, the training loop reads it.\nkey i = output of activation module i"),
            ([1, 4], "the factory: it builds one hook and hands it back.\nEach hook it builds remembers its own idx\n(more on why in a moment)"),
            ([2], "the hook itself: PyTorch will call it with\n(module, inp, out) after the module's forward()"),
            ([3], "save a copy of the output under this hook's key.\ndetach(): the copy leaves the autograd graph.\ncpu(): the copy lives in CPU memory, not on the GPU"),
            ([5], "one handle per hook, kept so we can remove them later"),
            ([6, 7, 8], "visit the three activation modules\n[Tanh, Tanh, Sigmoid] and attach hook i to module i"),
        ]
        rect = SurroundingRectangle(VGroup(code[0]), color=TAP, buff=0.07, stroke_width=3)
        expl = None
        taps = VGroup()
        for gi, (idxs, txt) in enumerate(groups):
            target = VGroup(*[code[i] for i in idxs])
            new_rect = SurroundingRectangle(target, color=TAP, buff=0.07, stroke_width=3)
            new_txt = para(txt, 21, WHITE, width=5.4).move_to([3.95, -0.55, 0])
            if gi == 0:
                self.play(Create(new_rect), FadeIn(new_txt), run_time=1.0)
                rect, expl = new_rect, new_txt
            else:
                self.play(Transform(rect, new_rect), FadeOut(expl), FadeIn(new_txt), run_time=1.0)
                expl = new_txt
            if gi == 5:
                for k, act in enumerate(ch.acts):
                    t = make_tap(act, k)
                    taps.add(t)
                    self.play(Create(t), run_time=0.8)
            self.wait(2.2)
        self.play(FadeOut(VGroup(code, rect, expl)))

        # --- one forward call ----------------------------------------------------
        slots = VGroup(*[make_slot(a.get_center()[0], 0.35, k) for k, a in enumerate(ch.acts)])
        store_lab = mono("_act_store", 18, GREY_B).next_to(slots, LEFT, buff=0.5).align_to([-6.6, 0, 0], LEFT)
        note = para("every time the model runs, each hook copies its module's output into its slot", 22).move_to([0, -1.5, 0])
        self.play(FadeIn(slots), FadeIn(store_lab), FadeIn(note))
        names = [r"z^{(1)}", r"z^{(2)}", r"y"]
        d = Dot(color=YELLOW, radius=0.1).move_to(ch.arrows[0].get_start())
        self.add(d)
        target_pts = [ch.boxes[1].get_center(), ch.boxes[3].get_center(), ch.boxes[5].get_center()]
        prev = ch.x_arr.get_start()
        fills = []
        for k in range(3):
            self.play(d.animate.move_to(target_pts[k]), run_time=1.3, rate_func=linear)
            copy = Dot(color=TAP, radius=0.1).move_to(target_pts[k])
            self.add(copy)
            val = mt(names[k], 24, TRAIN_C).move_to(slots[k][0])
            self.play(copy.animate.move_to(slots[k][0].get_center()), Indicate(taps[k], color=WHITE), run_time=1.1)
            self.play(FadeOut(copy), FadeIn(val), run_time=0.5)
            fills.append(val)
            if k < 2:
                self.play(d.animate.move_to(ch.boxes[2 * k + 2].get_center()), run_time=0.9, rate_func=linear)
        self.play(FadeOut(d))
        n2 = para("detach() and cpu(): the saved copy takes no part in backprop\nand uses no GPU memory.", 22, YELLOW).move_to([0, -2.4, 0])
        self.play(FadeIn(n2))
        self.wait(2.2)
        self.play(FadeOut(VGroup(n2, note)))

        # --- why a factory -------------------------------------------------------
        self.play(FadeOut(VGroup(ch.all, taps, slots, store_lab, *fills)))
        left_h = para("hook defined directly in the loop", 24, BAD_C).move_to([-3.6, 2.5, 0])
        right_h = para("hook built by the factory", 24, OK_C).move_to([3.6, 2.5, 0])
        note = para("A function looks up its variables when it RUNS, not when it was defined.", 22, YELLOW).move_to([0, 3.15, 0])
        self.play(FadeIn(note), FadeIn(left_h), FadeIn(right_h))

        def hook_dot(x, y, k, col):
            c = Dot([x, y, 0], radius=0.16, color=col)
            return VGroup(c, Text(f"hook {k}", font_size=16, color=col).next_to(c, UP, buff=0.1))

        def slot_row(cx, y, cols):
            return VGroup(*[make_slot(cx + dx, y, k).set_color(c) for k, (dx, c) in enumerate(zip((-2, 0, 2), cols))])

        wrong_hooks = VGroup(*[hook_dot(-3.6 + dx, 1.5, k, BAD_C) for k, dx in enumerate((-2, 0, 2))])
        shared = VGroup(Rectangle(width=2.4, height=0.7, color=BAD_C, stroke_width=3), mono("i  (shared)", 18, BAD_C))
        shared.move_to([-3.6, 0.3, 0])
        shared_after = mono("i = 2 (loop over)", 17, BAD_C).move_to(shared[1])
        wrong_slots = slot_row(-3.6, -1.7, [GREY_D, GREY_D, BAD_C])
        right_hooks = VGroup(*[hook_dot(3.6 + dx, 1.5, k, OK_C) for k, dx in enumerate((-2, 0, 2))])
        own = VGroup(*[VGroup(Rectangle(width=1.5, height=0.6, color=OK_C, stroke_width=3),
                              mono(f"idx = {k}", 17, OK_C)).move_to([3.6 + dx, 0.3, 0])
                       for k, dx in enumerate((-2, 0, 2))])
        right_slots = slot_row(3.6, -1.7, [OK_C, OK_C, OK_C])
        self.play(FadeIn(wrong_hooks), FadeIn(right_hooks), FadeIn(wrong_slots), FadeIn(right_slots), run_time=1.0)
        self.play(FadeIn(shared), FadeIn(own), run_time=0.8)
        w_arrows = [Arrow(wrong_hooks[k][0].get_bottom(), shared.get_top() + RIGHT * (k - 1) * 0.5, buff=0.1,
                          stroke_width=3, color=BAD_C, max_tip_length_to_length_ratio=0.2) for k in range(3)]
        r_arrows = [Arrow(right_hooks[k][0].get_bottom(), own[k].get_top(), buff=0.1, stroke_width=3,
                          color=OK_C, max_tip_length_to_length_ratio=0.3) for k in range(3)]
        self.play(*[Create(a) for a in w_arrows], *[Create(a) for a in r_arrows], run_time=1.2)
        self.wait(0.8)
        self.play(Transform(shared[1], shared_after))
        self.wait(1.0)
        w_out = Arrow(shared.get_bottom(), wrong_slots[2].get_top(), buff=0.1, stroke_width=5, color=BAD_C,
                      max_tip_length_to_length_ratio=0.2)
        r_out = [Arrow(own[k].get_bottom(), right_slots[k].get_top(), buff=0.1, stroke_width=3, color=OK_C,
                       max_tip_length_to_length_ratio=0.3) for k in range(3)]
        self.play(Create(w_out), *[Create(a) for a in r_out], run_time=1.4)
        w_res = para("the loop ends before any hook fires, so i = 2:\nall three hooks write to slot 2,\nslots 0 and 1 are never filled", 20, BAD_C).move_to([-3.6, -3.0, 0])
        r_res = para("each hook keeps its own idx:\nslots 0, 1, 2 are all filled", 20, OK_C).move_to([3.6, -3.0, 0])
        self.play(FadeIn(w_res), FadeIn(r_res))
        self.wait(3.0)
        self.play(FadeOut(VGroup(note, left_h, right_h, wrong_hooks, right_hooks, shared, own, wrong_slots, right_slots,
                                 w_res, r_res, *w_arrows, *r_arrows, w_out, *r_out)))

        # --- when is the store written and read ---------------------------------
        head = para("One epoch: when the store is written and when it is read", 26).move_to([0, 3.0, 0])
        labels = ["forward on\nX_train", "loss.backward()", "read the store:\nact_std, sat_frac", "optimizer.step()",
                  "forward on\nX_val"]
        boxes = VGroup(*[VGroup(RoundedRectangle(width=2.3, height=1.0, corner_radius=0.1, color=GREY_B, stroke_width=3),
                                Text(t, font_size=18, line_spacing=0.8)) for t in labels]).arrange(RIGHT, buff=0.35)
        boxes.move_to([0, 1.5, 0])
        arrs = [Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.03, stroke_width=3, color=GREY_B,
                      max_tip_length_to_length_ratio=0.35) for i in range(4)]
        store = VGroup(Rectangle(width=4.2, height=0.8, color=GREY_B, stroke_width=3),
                       mono("_act_store", 18, GREY_B).shift(UP * 0.0)).move_to([0, -0.7, 0])
        content = Text("(empty)", font_size=20, color=GREY_B).next_to(store, DOWN, buff=0.2)
        self.play(FadeIn(head), FadeIn(boxes), *[Create(a) for a in arrs], FadeIn(store), FadeIn(content))
        self.wait(0.6)
        hl = SurroundingRectangle(boxes[0], color=TAP, buff=0.08, stroke_width=4)
        self.play(Create(hl))
        fill1 = Text("TRAIN batch activations", font_size=20, color=TRAIN_C).next_to(store, DOWN, buff=0.2)
        self.play(Transform(content, fill1), store[0].animate.set_color(TRAIN_C))
        self.wait(1.4)
        self.play(Transform(hl, SurroundingRectangle(boxes[1], color=TAP, buff=0.08, stroke_width=4)))
        self.wait(0.8)
        self.play(Transform(hl, SurroundingRectangle(boxes[2], color=TAP, buff=0.08, stroke_width=4)))
        rd = Arrow(store.get_top(), boxes[2].get_bottom(), buff=0.1, stroke_width=4, color=OK_C,
                   max_tip_length_to_length_ratio=0.2)
        self.play(Create(rd))
        okn = para("the diagnostics describe the TRAIN batch", 22, OK_C).move_to([0, -2.4, 0])
        self.play(FadeIn(okn))
        self.wait(2.0)
        self.play(Transform(hl, SurroundingRectangle(boxes[3], color=TAP, buff=0.08, stroke_width=4)), FadeOut(rd))
        self.wait(0.8)
        self.play(Transform(hl, SurroundingRectangle(boxes[4], color=TAP, buff=0.08, stroke_width=4)))
        fill2 = Text("VAL batch activations  (overwritten!)", font_size=20, color=VAL_C).next_to(store, DOWN, buff=0.2)
        self.play(Transform(content, fill2), store[0].animate.set_color(VAL_C))
        warn = para("the hooks fire on every forward pass, validation included.\nThat is why the loop reads the store BEFORE this step.", 21, VAL_C).move_to([0, -3.1, 0])
        self.play(FadeIn(warn), FadeOut(okn))
        self.wait(3.0)
        nxt = Text("next epoch: forward on X_train overwrites it again", font_size=20, color=GREY_B).move_to([0, -2.4, 0])
        self.play(FadeIn(nxt))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head, boxes, *arrs, store, content, hl, warn, nxt)))

        # --- handles -------------------------------------------------------------
        h1 = para("Hooks stay attached until you remove them.", 26).move_to([0, 2.2, 0])
        h2 = para("Every later forward pass runs them and overwrites the store.", 22, GREY_B).move_to([0, 1.5, 0])
        rm = code_block(["for h in handles:", "    h.remove()"], 22, TAP).move_to([0, 0.2, 0])
        h3 = para("that is why we kept the handles:\nremove() detaches each hook and the model is back to normal", 22, OK_C)
        h3.move_to([0, -1.1, 0])
        ch2 = build_chain(-2.4)
        taps2 = VGroup(*[make_tap(a, k) for k, a in enumerate(ch2.acts)])
        self.play(FadeIn(h1), FadeIn(h2))
        self.play(FadeIn(ch2.all), FadeIn(taps2))
        self.play(FadeIn(rm))
        self.wait(1.0)
        self.play(*[FadeOut(t, shift=DOWN * 0.3) for t in taps2], FadeIn(h3), run_time=1.6)
        self.wait(3.0)
