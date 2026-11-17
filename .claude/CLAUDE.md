# Neural Networks Lab — Project Guide

## Ultimate Goal
**Implement Qwen3 inference from scratch and understand it deeply (inference engineering).**
Everything else in this repo is groundwork for that goal.

Success criteria (from the recipe the user is following):
1. **Own implementation.** A PyTorch implementation of the Qwen3 forward pass and the
   autoregressive generation loop (embeddings, attention with GQA, MLP, RMSNorm, logits →
   sampling, KV cache).
2. **Match Hugging Face.** The activations of every layer match those of the Hugging Face
   model. Forward hooks on both models make this comparison (see the hooks work in
   `neural_nets` notebook 01 and `tools/observability/hooks.py`).
3. **Profile.** Run the profiler, export the trace, open it in Perfetto, map the kernels back
   to the code, and understand where most of the time is spent.
4. **Work with a coding agent as a collaborator** while building it.

Model choice: the smallest Qwen3 that fits the hardware. The GPU has 4 GB, so 0.6B fits easily;
about 1.7B needs quantization or offload. The user's notes say "Qwen3-2B" with 28 layers; this is
unverified. Read the real sizes from the model's `config.json` before building.

Remaining topics on the path (from the user's study notes): attention (Q/K/V), multi-head
attention and GQA, the transformer block (attention + MLP + RMSNorm, stacked), from logits to
tokens (softmax, sampling, the generation loop), then the implementation matched against Hugging
Face layer by layer.

When choosing what to do next, prefer work that moves toward this goal. The neural-net series
builds the vocabulary (neuron, backprop, activations, hooks); the image and audio tracks are
supporting breadth, not the critical path.

## Purpose of the Lab
A hands-on learning lab for building and understanding neural networks from scratch.
Every notebook is self-contained, runnable top-to-bottom, and designed to be read
alongside the user's own notes.

## Top-level Layout
```
LLM/
  .claude/CLAUDE.md          ← this file
  neural_nets/               ← COMPLETE: tabular / MLP series (legacy structure)
  image_nets/                ← IN PROGRESS: CNN on images (MNIST → CIFAR-10)
  audio_nets/                ← PLANNED: spectrogram + 1D-CNN / RNN
  qwen/                      ← MAIN PATH, folder created: own Qwen3-0.6B → 1.7B (the ultimate goal)
  venv/                      ← shared Python virtual environment
```

## Per-domain Layout (image_nets, audio_nets; qwen follows its own layout, see its section)
```
<domain>_nets/
  notebooks/
    00_data_and_model.ipynb
    01_training.ipynb
    02_inference.ipynb       ← architecture, WeightWatcher, layer visualizations
    03_<domain_task>.ipynb   ← optional domain extension (e.g. cats_and_dogs)
  artifacts/                 ← checkpoints, scalers, plots (git-ignored large files)
  tools/                     ← domain-specific importable library
    nets/                    ← model definitions
    viz/                     ← domain-specific plots
    observability/           ← hooks, diagnostics (copied/extended from neural_nets)
    hardware/                ← Timer, memory utils (copied from neural_nets)
```

## Notebook Pattern (3 notebooks per domain — no CV notebook)
Neural networks don't use K-fold cross-validation in practice (training is too
expensive). The standard pipeline is:

```
train set  → fit the model
val set    → tune hyperparameters / early stopping
test set   → final score, reported once
```

| # | Notebook | What it covers |
|---|----------|---------------|
| 00 | data_and_model | Load/build dataset, define architecture, verify forward pass |
| 01 | training | Train loop, loss/acc curves, observability hooks, save checkpoint |
| 02 | inference | Architecture diagram, WeightWatcher health check, domain-specific visualizations |
| 03+ | (optional) | Domain extension — e.g. transfer learning on a new task |

## Environment
- WSL2, NVIDIA RTX 2050, 4 GB VRAM, Compute 8.6, CUDA 12.4
- PyTorch 2.6.0+cu124, numpy 2.x, matplotlib, tensorboard, scikit-learn
- Activate: `source /root/LLM/venv/bin/activate`
- Launch Jupyter: `jupyter notebook --notebook-dir=<domain>_nets/notebooks`

## Notebook Conventions (apply to every notebook in every domain)
1. Every code cell is preceded by a markdown cell with exactly three parts:
   **What we do** / **Why** / **What you should see**
2. One idea per cell; runs in seconds; prints at most ~30 lines.
3. Every logical step ends with `assert ... , "FAIL: ..."` + `print("PASS: ...")`.
4. Fixed seeds — `torch.manual_seed(SEED)` and `np.random.seed(SEED)` at the top.
5. `FORCE_CPU = False` near the top; setting it `True` must work end-to-end.
6. Notebooks share state only through files in `artifacts/`; no cross-notebook imports.
7. Each notebook imports from `tools.*` by adding the domain root to `sys.path`:
   ```python
   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path().resolve().parent))   # <domain>_nets/
   ```
8. Do not add comments unless the WHY is non-obvious. No docstrings in cells.

## Notation (match the user's notes exactly)
- `W(l)` — weight matrix for layer l, shape **(neurons_l × neurons_{l-1})**
- `a(l)` — pre-activation (weighted sum): `a = W(l) · z(l-1) + b(l)`
- `z(l)` — post-activation: `z = activation(a)`
- `delta(l)` — error signal: `dE/da(l)` (not `dE/dz`)
- Hidden layers: ReLU (CNNs) or tanh (MLPs)
- Output layer: softmax (multi-class) or sigmoid (binary)
- Loss: CrossEntropy for multi-class, BCE for binary

## Terminology (notebooks, code comments, plot labels, animations)
Use the standard terms, never analogies or figurative wording:
- backprop: *error signal* δ(l) = ∂E/∂a(l), *gradient*, *per-sample gradient*, *layer input* z(l-1) (x for layer 1) — not "blame", "push", "value on the wire"
- structure: *weight*, *connection*, *path* from an input to the output — not "wire", "pipe", "chain of blame"
- activations: *saturated* (|z| > 0.95), *outputs differ / are equal* — not "stuck", "neurons say the same thing"
- training phases of the loss: *plateau*, *fast decrease*, *slowing decrease* — not "take-off" / "slow-down"
- when wording in a scene file changes, re-render that video and re-run the notebook

## Domain-specific Visualization Tools

### image_nets — what each layer sees
| Tool | Function |
|------|----------|
| `tools/viz/filters.py` | `plot_conv_filters(model, layer_idx)` — conv kernel weights as an image grid |
| `tools/viz/feature_maps.py` | `plot_feature_maps(model, image)` — each channel's activation for one input |
| `tools/viz/gradcam.py` | `gradcam(model, image, target_layer)` — heatmap over input showing which pixels drove the prediction |

### audio_nets — planned
- Log-mel spectrogram visualization
- Learned filter responses on audio

### qwen — planned
- Token embedding scatter (UMAP/t-SNE)
- Attention heatmaps per head

---

## neural_nets (tabular / MLP series) — COMPLETE

### Directory
```
neural_nets/
  notebooks/  00–05
  artifacts/
  tools/
    nets/mlp.py
    observability/{hooks,diagnostics,logger}.py
    hardware/gpu.py
    viz/{curves,weights,activations}.py
    manim/
```

### Notebooks Status
| # | Name | Status |
|---|------|--------|
| 00 | environment_and_gpu | done + executed |
| 01 | data_model_training | done + executed — dataset, MLP, hand forward/backward, training loop, diagnostics, weight heatmaps |
| 02 | test_and_inference | done + executed |
| 03 | model_intuition | done + executed |

### Tools — neural_nets
#### tools/nets/mlp.py
`MLP(sizes, output="sigmoid")` — shared MLP. `nn.ModuleList` for layers and
activations so hooks attach to activation modules directly.

#### tools/observability/hooks.py
`register_activation_hooks(model)` → `(store_dict, handle_list)`.
`store_dict[l]` is the last-seen `z(l)` tensor. Remove with `for h in handles: h.remove()`.

#### tools/observability/diagnostics.py
`layer_stats(model, act_store, lr)` — after `loss.backward()`, returns flat dict:
grad norm, weight norm, update ratio, act std, saturated-tanh fraction per layer.
`gpu_stats(device)` — GPU allocated + peak MB.

#### tools/observability/logger.py
`JsonlLogger(path)` — `.log(entry)` appends JSON line. `.read_all()` returns list of dicts.

#### tools/hardware/gpu.py
`Timer` — context manager, `.ms` / `.s` properties, syncs CUDA.
`reset_memory_stats(device)` — resets CUDA peak counters.

#### tools/viz/curves.py
`plot_training_curves(history, save_path)` — 2-panel: loss + accuracy.

#### tools/viz/weights.py
`plot_weight_heatmaps(weight_snaps, epochs, save_path)` — before/after/delta grid.

#### tools/viz/activations.py
`plot_activation_landscape(model, device, layer_idx, save_path)` — 2D (x1,x2) grid, contour per neuron.
`plot_layer_representations(model, X, y, device, save_path)` — scatter of each layer's output coloured by class.
`plot_neuron_weights(model, save_path, feature_names)` — L1 weight bars.

#### tools/manim/ — training animations (notebook 01, training section)
Each file is one self-contained Scene. Render with
`manim -ql tools/manim/<file>.py <SceneName>`, then copy the mp4 into
`artifacts/manim/` (the notebook embeds it from there).

| File | Scene | Shows |
|------|-------|-------|
| `training_step.py` | `TrainingStep` | One epoch on 3→3→2→1: zero_grad → forward → loss → backward → step, with sample-0 values |
| `zero_grad.py` | `ZeroGradAccumulation` | `.grad` with vs without `zero_grad()` |
| `gradient_descent.py` | `GradientDescentStep` | lr too small / good / too large on a 1D loss curve (update_ratio) |
| `tanh_saturation.py` | `TanhSaturation` | tanh vs `1 − tanh²`; why sat_frac matters |
| `forward_pass.py` | `ForwardPass` | Symbolic forward pass after the Deep Dive notes: a/z-split neurons, numbering rule `w₂₃⁽¹⁾`, per-neuron equations → `a = W·x + b`, layers 2–3, whole-network formula (~75 s) |
| `backward_pass.py` | `BackwardPass` | Symbolic backprop after the Deep Dive notes: three tools, stages 1–7 lit on the network, route sum, Moves A/B/C (MSE vs BCE), `Wᵀ` and outer product, six code lines, update; stages 1–5 derived line by line (~3 min) |
| `grad_norm.py` | `GradientNorm` | Worked example of `‖∇W‖`, then healthy / vanishing / exploding curves |
| `weight_norm.py` | `WeightNorm` | Worked example of `‖W‖`, then healthy / frozen / diverging curves |
| `update_ratio.py` | `UpdateRatio` | Step bar vs weights bar at ratio ≈ 0, 0.05, 1, then the ratio over epochs (log scale, healthy 0.01–0.1 band) |
| `activation_std.py` | `ActivationStd` | Neuron outputs as dots with `μ ± σ`, then σ over epochs: stable, collapsing, saturating |
| `sat_frac.py` | `SaturatedFraction` | Dots entering the `\|z\|>0.95` zone as pre-activations grow, then `sat_frac` over epochs |
| `std_vs_sat.py` | `ActStdVsSatFrac` | Four output patterns with σ and sat_frac side by side: why both are tracked |
| `hooks.py` | `HookBasics` | What a forward hook is: the problem, `register_forward_hook`, `(module, inp, out)`, order of events, returning `None`, handles (~50 s) |
| `hooks.py` | `HookInOurLoop` | Our hook code line by line, one forward call filling `_act_store`, why the factory, when the store is written/read in an epoch, removing hooks (~90 s) |
| `training_story.py` | `TrainingStory` | The story of the five diagnostic plots for this run: loss anchor, then per plot what we see / why / so what, how they connect, verdict (~6 min). Run data is embedded as `DATA` in the file |
| `weight_story.py` | `WeightStory` | The story of the weight heatmaps: the three tables per layer, the chain of weights that carries x3, the correlation and shuffling tests, and what longer training does (~4 min). Weights and check numbers are embedded as `DATA_W` / `CHECK` |

Metric scenes (and any text-heavy scene) render with `manim -r 1280,720 --fps 15 <file>.py <Scene>` — readable and ~0.5 MB each.

Manim needs apt packages `libcairo2-dev libpango1.0-dev pkg-config ffmpeg texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm`.
`MathTex` works, but only with the minimal template `TexTemplate(preamble=r"\usepackage{amsmath}\usepackage{amssymb}")` (see `forward_pass.py`). Avoid `DecimalNumber`.

---

## image_nets — IN PROGRESS

### Notebooks Status
| # | Name | Status |
|---|------|--------|
| 00 | data_and_model | done + executed |
| 01 | training | done + executed |
| 02 | inference | done + executed — architecture diagram (torchinfo + block diagram), WeightWatcher (RMT), conv filters, feature maps, Grad-CAM, embedding PCA, confused samples |
| 03 | cats_and_dogs | done + executed — CIFAR-10 binary, frozen ResNet18, BCE loss, Grad-CAM, embedding PCA, error analysis |

### Architecture Plan
- Dataset: MNIST (28×28 grayscale, 10 classes, 60k train / 10k test)
- Model: small CNN — Conv(1→16,3×3)→ReLU→MaxPool → Conv(16→32,3×3)→ReLU→MaxPool → FC(→128)→ReLU → FC(→10)
- Loss: CrossEntropy
- Optimizer: Adam (LR=1e-3, no LR grid search needed for this architecture)
- Val split: 10% of training set, fixed split

### Tools Plan
| File | Functions |
|------|-----------|
| `tools/nets/cnn.py` | `SmallCNN()` — the shared CNN class |
| `tools/viz/filters.py` | `plot_conv_filters(model, layer_name)` |
| `tools/viz/feature_maps.py` | `plot_feature_maps(model, image, device)` |
| `tools/viz/gradcam.py` | `gradcam(model, image, target_layer, device)` |
| `tools/observability/hooks.py` | same pattern as neural_nets |
| `tools/hardware/gpu.py` | copied Timer + reset_memory_stats |

## qwen — Qwen3 implementation (main path, in progress: notebook 00 done)

**Everything for the Qwen work lives in `qwen/` (notebooks, tools, weights, videos, profiler traces), kept apart from the neural-net learning folders.** Only reuse code from them by copying or importing the specific helper (hooks, Timer).

Build our own Qwen3 (model code in `qwen/tools/nets/qwen3.py`), load the open Hugging Face
weights, match the Hugging Face activations layer by layer, then profile. **Qwen3-0.6B first,
then 1.7B with the same code.** No training: the weights are the downloaded ones.

### Facts for Qwen3-0.6B (from its `config.json`)
| Item | Value |
|------|-------|
| layers | 28 |
| hidden size | 1024 |
| attention | 16 query heads, 8 key/value heads (GQA, 2 query heads per K/V head), head_dim 128 |
| projections | q: 1024 → 2048, k and v: 1024 → 1024, o: 2048 → 1024 (no biases) |
| MLP | SwiGLU: gate and up 1024 → 3072, down 3072 → 1024, SiLU |
| norms | RMSNorm, eps 1e-6; plus a per-head RMSNorm (weight size 128) on q and k, applied after the reshape to heads and before RoPE (verified in HF `modeling_qwen3.py`) |
| positions | RoPE, theta 1,000,000, no rope scaling |
| vocabulary | 151,936; the output head is tied to the embedding table |
| dtype | bfloat16 |
| parameters | 596,049,920 (computed from the config) |
| KV cache | 28 × 2 × 8 × 128 × 2 bytes = 112 KiB per token |

### Hardware limits (checked)
- GPU: RTX 2050, 4 GB (3.96 GB free), compute 8.6 (bf16 works). 0.6B in fp32 is about 2.4 GB, so
  the first matching can be done in fp32 with a tight tolerance, then in bf16 (measured in nb 00: GPU free 3.22 GiB at start;
  HF fp32 reference peaks at 2.23 GiB, so our own fp32 copy and the reference do not fit together: compare layer by layer or keep one on CPU).
- RAM: only 4.8 GiB in WSL2. For 1.7B (about 3.4 GB in bf16, more than fits on the GPU) load tensors
  straight to the GPU with `safetensors` and use quantization or offload.
- Installed: `torch` 2.6.0+cu124, `safetensors`, `transformers` 5.22.1, `huggingface_hub`, `tokenizers`.
- Weights downloaded to `qwen/artifacts/Qwen3-0.6B/` (`model.safetensors` 1.50 GB, 311 tensors, 751,632,384 elements:
  it stores `lm_head.weight` as a separate tensor, bit-identical to `embed_tokens.weight`, so the unique parameters are 596,049,920).
  Layer tensor names: `model.layers.N.{input_layernorm, post_attention_layernorm, self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm, mlp.{gate,up,down}_proj}.weight`.

### Planned notebooks (each checks its result against PyTorch or Hugging Face)
| # | Notebook | What it covers | Status |
|---|----------|----------------|--------|
| 00 | environment_and_config | install, read `config.json`, parameter count by hand, memory budget, HF reference runs | done + executed |
| 01 | weights_and_tokenizer | list safetensors tensor names and shapes, map each to a component, tokenizer round trip | todo |
| 02 | rmsnorm_and_mlp | RMSNorm and the SwiGLU MLP, layer 0 vs the Hugging Face modules | todo |
| 03 | attention | Q/K/V, q/k norm, causal mask, softmax; one head vs `F.scaled_dot_product_attention`, then multi-head and GQA | todo |
| 04 | rope | rotary embeddings, applied to q and k, vs Hugging Face | todo |
| 05 | transformer_block | pre-norm residual block, layer 0 vs Hugging Face with forward hooks | todo |
| 06 | full_model | 28 layers, embeddings, final norm, tied head; every layer's activations and the logits match | todo |
| 07 | generation_and_kv_cache | greedy decoding, KV cache, prefill vs decode, sampling; text equals Hugging Face greedy output | todo |
| 08 | profiling | `torch.profiler` → Chrome trace → Perfetto; map kernels to code; where the time goes | todo |
| 09 | scale_to_1p7b | same code, 1.7B config, loading, memory fit, how the profile changes | todo |

Reuse: forward hooks from `neural_nets/tools/observability/hooks.py` for the activation matching,
`Timer` from `neural_nets/tools/hardware/gpu.py` for timing. Manim scenes go in `qwen/tools/manim/`
(the attention Q/K/V geometry is the target demo).

## Manim — Math Visualization (Planned)

Manim will be used to animate **what is happening inside neural networks from
a mathematical perspective** — calculus, optimization, statistics, and linear
algebra, made visible step by step.

### Vision

The goal is not "here is what the code does" — it is "here is the geometry,
the gradient, the transformation, playing out in front of you."

Examples of planned animations:

| Concept | What to animate |
|---------|----------------|
| Forward pass | Vectors flowing through weight matrices; each layer as a linear map + nonlinearity |
| Backpropagation | Chain rule as a reverse sweep; delta signals flowing backward |
| Gradient descent | Loss surface contour; parameter point moving toward the minimum |
| Conv filter | 3×3 kernel sliding over a feature map; each multiplication shown |
| Softmax | Raw logit vector → probability simplex; temperature effect |
| Attention (Q/K/V) | Each token is a vector; Q and K rotate/scale vectors to compute similarity scores; V adjusts the value vector to fit context — how math redistributes meaning |

### Scope

Start with neural nets (MLPs, CNNs) before tackling LLM-scale topics. The
Transformer Q/K/V geometry is the target demo once the neural-net foundations
are animated, because it is where linear algebra + statistics intersect most
visibly.

### Location

```
neural_nets/tools/manim/    ← MLP / gradient / backprop scenes (first)
image_nets/tools/manim/     ← Conv / feature map scenes (next)
qwen/tools/manim/      ← Attention / embedding scenes (later)
```

Each `.py` file is a self-contained Manim `Scene` subclass.  Render with:
```
manim -pql <scene_file>.py <SceneName>
```

---

## Adding a New Tool
1. Create the function in the appropriate submodule.
2. Export it from the submodule's `__init__.py`.
3. Write a minimal test (assert-based, no framework) under `if __name__ == "__main__":`.
4. Document it in this file under the relevant domain section.
