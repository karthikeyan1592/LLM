# Neural Nets → LLMs: Study Notes

Oct 26, 2026 · @Karthikeyan

## The Single Neuron

A neuron multiplies each input by a weight, adds a bias, and passes the sum through a non-linear activation function.

```latex
a = \sum_{i=1}^{D} w_i x_i + b
```

```latex
z = h(a)
```

- xᵢ — inputs (for a token, its embedding values)
- wᵢ — weights, fixed after training; what you download from HuggingFace
- b — bias, a learned constant offset
- h(·) — activation function: sigmoid, tanh, ReLU, or SiLU
- a — pre-activation, the weighted sum before squashing
- z — the neuron's output

Without h, stacking layers collapses into one linear function — depth adds nothing. The non-linearity is what makes 28 stacked layers genuinely different from one.

&#91;embedded content: single neuron · weighted sum then activation\]

**Worked example** — one neuron, two inputs, tanh activation:

Inputs x₁ = 1.0, x₂ = 0.5. Weights w₁ = 1.0, w₂ = 2.0, bias b = -0.5.

```latex
a = (-0.5) + (1.0)(1.0) + (2.0)(0.5) = 1.5
```

```latex
z = \tanh(1.5) \approx 0.905
```

## Forward Pass — Worked Example

A full tiny network: 2 inputs, a bias, 2 hidden neurons (tanh), 1 output neuron (sigmoid). This is the same weighted-sum-then-squash step a transformer layer repeats, just on a tiny scale.

&#91;embedded content: 2-2-1 network · every node connects to every node in the next layer\]

**Setup.** Inputs x₁ = 1.0, x₂ = 0.5 (bias input x₀ = 1.0). Target output t = 1.0.

**Step 1 — hidden neuron z₁.** Weights into it: bias w₁₀⁽¹⁾ = -0.5, w₁₁⁽¹⁾ = 1.0, w₁₂⁽¹⁾ = 2.0.

```latex
a_1 = (-0.5)(1) + (1.0)(1.0) + (2.0)(0.5) = 1.5
```

```latex
z_1 = \tanh(1.5) \approx 0.9051
```

**Step 2 — hidden neuron z₂.** Weights into it: bias w₂₀⁽¹⁾ = 0.2, w₂₁⁽¹⁾ = -1.5, w₂₂⁽¹⁾ = 0.5.

```latex
a_2 = (0.2)(1) + (-1.5)(1.0) + (0.5)(0.5) = -1.05
```

```latex
z_2 = \tanh(-1.05) \approx -0.7818
```

**Step 3 — output y.** Weights into it: bias w₁₀⁽²⁾ = 0.1, w₁₁⁽²⁾ = 0.8, w₁₂⁽²⁾ = -0.4.

```latex
a_{out} = 0.1 + (0.8)(0.9051) + (-0.4)(-0.7818) = 1.1368
```

```latex
y = \sigma(1.1368) \approx 0.7571
```

The network's output is 75.7% — the predicted probability for whatever class t = 1 represents.

**The full formula, in one line** — everything above is one expression:

```latex
y(x, w) = \sigma\left( \sum_{j} w_{j}^{(2)} \, \tanh\left( \sum_{i} w_{ji}^{(1)} x_i \right) \right)
```

Each layer is: multiply by weights, sum, squash. Stack enough of these and the composition can approximate very complex functions — that is the entire mechanism a 28-layer transformer scales up.

## Why the Error Function Looks Like a Bowl

An error function measures distance between a prediction and the target. For that measurement to be usable by gradient descent it must satisfy three properties:

1. **Non-negative** — E(w) ≥ 0, zero only when the prediction is exact
2. **Monotonic penalty** — a bigger miss always costs more, never less
3. **Smooth** — differentiable everywhere, so a slope exists to walk downhill on

The simplest function meeting all three is the squared difference:

```latex
E(w) = \frac{1}{2}(y - t)^2
```

Since x² has a positive second derivative everywhere, it is a strictly convex bowl with exactly one minimum, at y = t.

&#91;embedded content: squared-error curve · conceptual sketch, not plotted from data\]

**Why not a sine wave?** sin(w) is periodic — it would mean a catastrophic mistake can cost the same as a perfect prediction. It also has flat regions with zero gradient, where descent gets stuck with no signal to follow.

**Why not tan(w)?** tan(w) has vertical asymptotes — the gradient explodes to infinity, and a loss of −∞ would mean being wrong grants infinite reward. Nonsensical.

**Where the bowl comes from, probabilistically.** If target noise is assumed Gaussian, maximizing the likelihood of the data is equivalent to minimizing squared error:

```latex
-\ln p(y \mid \hat{y}) = \frac{1}{2\sigma^2}(y - \hat{y})^2 + \text{const}
```

The log cancels the Gaussian's exponential, leaving a pure quadratic bowl.

**Even in a deep, non-convex network.** The full loss surface of a 28-layer model has many hills and valleys — it is not one clean bowl globally. But near any local minimum w\*, a Taylor expansion gives:

```latex
E(w) \approx E(w^*) + \frac{1}{2}(w - w^*)^T H (w - w^*)
```

The gradient is zero at a minimum, so the linear term vanishes — locally, the surface always reduces to a quadratic bowl, whatever it looks like globally.

**For LLMs specifically:** token prediction uses cross-entropy loss, E = −log(probability assigned to the correct token), not squared error — but it is built the same way: non-negative, monotonic, smooth, and convex in the predicted probability. Same bowl logic, different distance measure.

## Gradient Descent — The Update Rule

```latex
w^{(\tau+1)} = w^{(\tau)} - \eta \nabla E(w^{(\tau)})
```

Every weight in the network gets this update, every training step.

- w⁽τ⁾ — the current weight vector at step τ
- ∇E(w⁽τ⁾) — the gradient: the vector of partial derivatives ∂E/∂wi, pointing toward steepest *increase* in error
- −∇E(w⁽τ⁾) — flip the sign, now it points toward steepest *decrease*
- η — learning rate, a positive scalar controlling step size. Too small: training crawls. Too large: the update overshoots the minimum and the loss can oscillate or diverge
- Δw⁽τ⁾ = −η∇E(w⁽τ⁾) — the actual adjustment added to the weight

&#91;embedded content: gradient descent step · conceptual sketch\]

**Why subtract, not add?** If the slope at the current point is positive (error rises as w increases), subtracting the gradient pushes w down — toward lower error. If the slope is negative, subtracting a negative pushes w up. Either way, the step moves downhill.

**Batch vs. stochastic.** Batch gradient descent sums the gradient over the entire training set before taking one step — accurate but slow. Stochastic gradient descent (SGD) updates after each example (or small mini-batch) — noisier per step, but far more steps per second, and the noise itself can help the optimizer escape shallow local minima. In practice, LLM training uses mini-batch SGD variants (Adam, AdamW) almost exclusively.

## Backpropagation — Chain Rule, Step by Step

Uses the same network and numbers as the forward pass above: z₁ = 0.9051, z₂ = −0.7818, y = 0.7571, target t = 1.0.

```latex
E = \frac{1}{2}(y-t)^2 = \frac{1}{2}(0.7571 - 1.0)^2 \approx 0.0295
```

### The sigmoid derivative

The output neuron uses y = σ(a) = 1/(1+e⁻ᵃ). Its derivative is needed at every backward step, so it is worth deriving once:

```latex
\frac{dy}{da} = -(1+e^{-a})^{-2}\cdot(-e^{-a}) = \frac{e^{-a}}{(1+e^{-a})^2} = \left(\frac{1}{1+e^{-a}}\right)\left(\frac{e^{-a}}{1+e^{-a}}\right)
```

The first factor is y itself. The second factor is (1 − y), since 1 − y = e⁻ᵃ/(1+e⁻ᵃ). So:

```latex
\frac{dy}{da} = y(1-y)
```

This is why sigmoid is convenient: the derivative is computed purely from the forward pass's own output — no need to re-evaluate the exponential.

&#91;embedded content: backward pass · error signal scaled by the same weights, in reverse\]

### Layer 2 — gradient for output weight w₁₁⁽²⁾

Three links in the chain:

```latex
\frac{\partial E}{\partial w_{11}^{(2)}} = \frac{\partial E}{\partial y}\cdot\frac{\partial y}{\partial a_{out}}\cdot\frac{\partial a_{out}}{\partial w_{11}^{(2)}}
```

**Output error signal δ\_out:**

```latex
\frac{\partial E}{\partial y} = y - t = 0.7571 - 1.0 = -0.2429
```

```latex
\frac{\partial y}{\partial a_{out}} = y(1-y) = 0.7571 \times 0.2429 = 0.1839
```

```latex
\delta_{out} = (-0.2429)(0.1839) = -0.04467
```

**Gradient w.r.t. the weight itself.** Since a\_out = w₁₀⁽²⁾ + w₁₁⁽²⁾z₁ + w₁₂⁽²⁾z₂, the derivative ∂a\_out/∂w₁₁⁽²⁾ is just z₁ — every other term is constant with respect to this one weight:

```latex
\frac{\partial E}{\partial w_{11}^{(2)}} = \delta_{out}\cdot z_1 = (-0.04467)(0.9051) \approx -0.04043
```

### Layer 1 — gradient for hidden weight w₁₁⁽¹⁾

The error has to pass back through the hidden unit first. Its error signal δ₁ is the output's δ, scaled by the weight connecting them, scaled again by how sensitive z₁'s own activation is:

```latex
\delta_1 = \delta_{out}\cdot w_{11}^{(2)}\cdot h'(a_1)
```

```latex
h'(a_1) = 1 - z_1^2 = 1 - (0.9051)^2 \approx 0.1808
```

```latex
\delta_1 = (-0.04467)(0.8)(0.1808) \approx -0.00646
```

Since a₁ = w₁₀⁽¹⁾x₀ + w₁₁⁽¹⁾x₁ + w₁₂⁽¹⁾x₂, ∂a₁/∂w₁₁⁽¹⁾ = x₁ = 1.0:

```latex
\frac{\partial E}{\partial w_{11}^{(1)}} = \delta_1 \cdot x_1 = (-0.00646)(1.0) = -0.00646
```

### Applying the update (η = 0.1)

```latex
w_{11}^{(2),new} = 0.8 - 0.1\times(-0.04043) = 0.80404
```

```latex
w_{11}^{(1),new} = 1.0 - 0.1\times(-0.00646) = 1.00065
```

Both gradients were negative, so both weights increased slightly — nudging the next prediction closer to t = 1.0.

### Why this matters computationally

Computing each of a network's W gradients by perturbing one weight at a time (finite differences) costs O(W) forward passes per gradient — O(W²) total. Backprop computes every gradient in the network in one backward sweep: O(W). For a 2B-parameter model, that difference is the entire reason training is feasible at all.

## Backpropagation in plain words (no symbols)

The forward pass ends with one number: the error E. Training asks, for every weight in the network, one question: **if I nudge this weight up a little, how much does E change?** That answer is the weight's gradient.

Think of it as a chain of blame. The error is felt at the output, and it has to be traced backwards through every wire that contributed to it.

1. **Start at the output.** The error depends directly on the final answer y. If y is too high, the error says "lower y".
2. **Go through the last neuron.** y came from a weighted sum pushed through a sigmoid. The sigmoid is steep in the middle and flat at the ends, so a nudge to the sum matters a lot or hardly at all. That steepness scales the blame. Call the result the **blame at that neuron**.
3. **Blame each incoming weight.** A weight sits on a wire carrying some value. If the wire carries a big value, nudging the weight changes the sum a lot; if it carries zero, nudging does nothing. So a weight's gradient is the blame at the neuron it feeds, multiplied by the value on its own wire.
4. **Pass blame to the neurons behind.** A neuron in the previous layer sent its value along a wire with a weight. It receives the blame of the neuron it fed, multiplied by that wire's weight. A strong wire passes on strong blame.
5. **If a neuron feeds several neurons, add.** A middle neuron can influence the error through more than one route. Each route delivers its own share of blame and the shares are added. The earlier 2-2-1 example could not show this, because each of its neurons had only one route.
6. **Pass through that neuron's own activation.** Multiply by the steepness of its activation at the value it had during the forward pass. This turns blame on the neuron's output into blame on its weighted sum.
7. **Repeat from step 3 one layer further back,** until the inputs are reached.

Two things to hold on to. Every step uses numbers the forward pass already produced (the values on wires, the steepness of each activation), which is why training must keep them in memory. And every step is a multiplication or an addition, so the whole sweep costs about as much as one forward pass.

## Backpropagation layer by layer: error for every weight

The plain-words recipe becomes equations for the 2-2-2-1 network. We go from the last layer back to the first, and we keep the forward-pass numbers (every a and z) because every formula below reuses them.

### The two ideas everything is built from

**Idea 1 — one number per neuron: δ.** A neuron's weighted sum a affects the error only by passing through the neuron's activation and then travelling on to later layers. Define

```latex
\delta_k^{(l)} = \frac{\partial E}{\partial a_k^{(l)}}
```

In words: "if the weighted sum of neuron k in layer l moved up by a tiny amount, the error would move by δ times that amount." Big |δ| means this neuron matters a lot for the error right now; δ near 0 means it hardly matters. The sign says which way to push.

**Idea 2 — once δ is known, its weights are free.** Neuron k computes a\_k = … + w\_kj z\_j + … + b\_k. Changing w\_kj by a tiny amount changes a\_k by z\_j times that amount (z\_j is the value riding on that wire), and changing b\_k changes a\_k by exactly that amount. Chain rule: first how the error reacts to a\_k (that is δ), then how a\_k reacts to the weight:

```latex
\frac{\partial E}{\partial w_{kj}^{(l)}} = \underbrace{\frac{\partial E}{\partial a_k^{(l)}}}_{\delta_k^{(l)}} \cdot \underbrace{\frac{\partial a_k^{(l)}}{\partial w_{kj}^{(l)}}}_{z_j^{(l-1)}}, \qquad \frac{\partial E}{\partial b_k^{(l)}} = \delta_k^{(l)} \cdot 1
```

So **weight gradient = δ of the neuron the wire points into × the value on the wire.** That is the only formula we ever use for weights. The whole job is therefore finding δ for every neuron, last layer first.

Two small facts about the activations, both computed from numbers we already stored (no recomputation):

| Activation | Steepness (derivative) | In terms of stored values |
| --- | --- | --- |
| tanh, z = tanh(a) | 1 − tanh²(a) | 1 − z² |
| sigmoid, y = σ(a) | σ(a)(1 − σ(a)) | y(1 − y) |

### Step A — layer 3 (output): the error, δ⁽³⁾ and its two weights

&#91;embedded content: Layer 3 · error to δ⁽³⁾ to its two weights\]

The error E = ½(y−t)² sits at the far right. Moving left, there are two links in the chain: E depends on y, and y depends on a⁽³⁾ through the sigmoid.

```latex
\frac{\partial E}{\partial y} = y - t, \qquad \sigma'\left(a^{(3)}\right) = y(1-y), \qquad \delta^{(3)} = (y - t)\cdot y\,(1 - y)
```

Both factors are small when y is close to the target or saturated at 0 or 1, so δ⁽³⁾ is small exactly when the output neuron has little to correct. The two weights into the output neuron carry z₁⁽²⁾ and z₂⁽²⁾ (the blue wires in the picture), so by Idea 2:

```latex
\frac{\partial E}{\partial w_{11}^{(3)}} = \delta^{(3)} z_1^{(2)}, \qquad \frac{\partial E}{\partial w_{12}^{(3)}} = \delta^{(3)} z_2^{(2)}, \qquad \frac{\partial E}{\partial b^{(3)}} = \delta^{(3)}
```

Three gradients are done. Notice they cost one multiplication each.

### Step B — from layer 3 back to layer 2: pass the blame, then apply steepness

&#91;embedded content: Layer 3 → layer 2 · one route per neuron\]

We now need δ for the two layer-2 neurons. Their a-values reach the error only through their own z, and each z has **exactly one wire** onward: z₁⁽²⁾ goes through w₁₁⁽³⁾ into the output neuron, z₂⁽²⁾ through w₁₂⁽³⁾. A wire with weight w multiplies any change by w, so the blame arriving at the z side is the downstream δ times that weight. With a single route there is nothing to add:

```latex
\frac{\partial E}{\partial z_1^{(2)}} = \delta^{(3)} w_{11}^{(3)}, \qquad \frac{\partial E}{\partial z_2^{(2)}} = \delta^{(3)} w_{12}^{(3)}
```

That is blame on the neuron's output z. To get blame on its weighted sum a, step through the neuron's own activation, multiplying by the tanh steepness from the table above:

```latex
\delta_1^{(2)} = \delta^{(3)} w_{11}^{(3)} \left(1 - {z_1^{(2)}}^2\right), \qquad \delta_2^{(2)} = \delta^{(3)} w_{12}^{(3)} \left(1 - {z_2^{(2)}}^2\right)
```

Now Idea 2 again. Each layer-2 weight is scored with the δ of the neuron it points into and the z at its left end (the dashed wires in the picture). Read the grid row by row: row 1 is everything feeding neuron 1, row 2 everything feeding neuron 2.

```latex
\begin{aligned}
\frac{\partial E}{\partial w_{11}^{(2)}} &= \delta_1^{(2)} z_1^{(1)} & \frac{\partial E}{\partial w_{12}^{(2)}} &= \delta_1^{(2)} z_2^{(1)} & \frac{\partial E}{\partial b_1^{(2)}} &= \delta_1^{(2)} \\
\frac{\partial E}{\partial w_{21}^{(2)}} &= \delta_2^{(2)} z_1^{(1)} & \frac{\partial E}{\partial w_{22}^{(2)}} &= \delta_2^{(2)} z_2^{(1)} & \frac{\partial E}{\partial b_2^{(2)}} &= \delta_2^{(2)}
\end{aligned}
```

Six more gradients, nine in total so far.

### Step C — from layer 2 back to layer 1: this is where routes add

&#91;embedded content: Layer 2 → layer 1 · two routes, blame adds\]

The new situation: z₁⁽¹⁾ feeds **both** layer-2 neurons. Change z₁⁽¹⁾ a little and two things happen at once: a₁⁽²⁾ moves (through w₁₁⁽²⁾) and a₂⁽²⁾ moves (through w₂₁⁽²⁾). Each of those moves changes the error, and the total change is the sum of the two effects. That is the multivariable chain rule, and it is the one new ingredient in this section:

```latex
\frac{\partial E}{\partial z_1^{(1)}} = \underbrace{\delta_1^{(2)} w_{11}^{(2)}}_{\text{route via } a_1^{(2)}} + \underbrace{\delta_2^{(2)} w_{21}^{(2)}}_{\text{route via } a_2^{(2)}}, \qquad \frac{\partial E}{\partial z_2^{(1)}} = \underbrace{\delta_1^{(2)} w_{12}^{(2)}}_{\text{route via } a_1^{(2)}} + \underbrace{\delta_2^{(2)} w_{22}^{(2)}}_{\text{route via } a_2^{(2)}}
```

A quick check on the indices: the wire from layer-1 neuron j to layer-2 neuron k has weight w\_kj⁽²⁾. So for j = 1 the two wires are w₁₁ and w₂₁ (first index changes, second stays 1), which is why the sum above runs down a column of W⁽²⁾. That column-reading is exactly the transpose in the matrix form coming next.

Then step through each layer-1 neuron's tanh, and score the layer-1 weights. Their "values on the wire" are the raw inputs x:

```latex
\delta_j^{(1)} = \frac{\partial E}{\partial z_j^{(1)}} \left(1 - {z_j^{(1)}}^2\right), \qquad \frac{\partial E}{\partial w_{jm}^{(1)}} = \delta_j^{(1)} x_m, \qquad \frac{\partial E}{\partial b_j^{(1)}} = \delta_j^{(1)}
```

Six more gradients: all 15 parameters now have a formula (4 + 4 + 2 = 10 weights and 2 + 2 + 1 = 5 biases).

### The whole procedure on one page

| Step | New quantity | Formula | Needs from the forward pass |
| --- | --- | --- | --- |
| A | δ⁽³⁾ | (y−t) · y(1−y) | y |
| A | weight gradients, layer 3 | δ⁽³⁾ · z⁽²⁾ | z₁⁽²⁾, z₂⁽²⁾ |
| B | δ⁽²⁾ | (w⁽³⁾ δ⁽³⁾) · (1−z⁽²⁾²) | z⁽²⁾ |
| B | weight gradients, layer 2 | δ⁽²⁾ · z⁽¹⁾ | z₁⁽¹⁾, z₂⁽¹⁾ |
| C | δ⁽¹⁾ | (sum over routes of δ⁽²⁾ · w⁽²⁾) · (1−z⁽¹⁾²) | z⁽¹⁾ |
| C | weight gradients, layer 1 | δ⁽¹⁾ · x | x |

Every row uses something the forward pass produced, which is why training has to store all the activations. The full numbers for this exact network, with each product written out, are in the "LLM Example" section below.

## Backpropagation in matrix form

The same wires carry the blame backwards, so the weight matrix reappears, **transposed**. The picture shows the flow; the equations below say it for a whole layer at once.

&#91;embedded content: Backward pass · blame δ travels along the same wires, in reverse\]

Stacking the δ of every neuron in a layer into a vector δ⁽ˡ⁾, one backward step is:

```latex
\delta^{(L)} = \frac{\partial E}{\partial y} \odot h'\left(a^{(L)}\right), \qquad \delta^{(l)} = \left( {W^{(l+1)}}^{\top} \delta^{(l+1)} \right) \odot h'\left(a^{(l)}\right), \qquad l = L-1, \dots, 1
```

The symbol ⊙ means multiply entry by entry. The transpose is the "sum over routes" from the previous section: row k of W⁽ˡ⁺¹⁾ holds the weights from layer-l neurons into neuron k, so Wᵀδ adds up, for each layer-l neuron, every route's δ × w. For our network:

```latex
\begin{bmatrix} \partial E / \partial z_1^{(1)} \\ \partial E / \partial z_2^{(1)} \end{bmatrix} = \begin{bmatrix} w_{11}^{(2)} & w_{21}^{(2)} \\ w_{12}^{(2)} & w_{22}^{(2)} \end{bmatrix} \begin{bmatrix} \delta_1^{(2)} \\ \delta_2^{(2)} \end{bmatrix} = \begin{bmatrix} w_{11}^{(2)} \delta_1^{(2)} + w_{21}^{(2)} \delta_2^{(2)} \\ w_{12}^{(2)} \delta_1^{(2)} + w_{22}^{(2)} \delta_2^{(2)} \end{bmatrix}
```

This reproduces the two-route sums written out one by one earlier. Once δ⁽ˡ⁾ is known, all weight gradients of that layer come from one outer product (every δ times every incoming z):

```latex
\frac{\partial E}{\partial W^{(l)}} = \delta^{(l)} \left( z^{(l-1)} \right)^{\top}, \qquad \frac{\partial E}{\partial b^{(l)}} = \delta^{(l)}
```

Shapes check out: δ⁽ˡ⁾ is (neurons in l)×1 and z⁽ˡ⁻¹⁾ᵀ is 1×(neurons in l−1), so the product has the shape of W⁽ˡ⁾. Per layer, backward costs two matrix products (Wᵀδ and the outer product) against one for forward, so a training step costs roughly three times a forward pass, and it needs every z and a from the forward pass kept in memory.

## LLM Example: all 15 gradients with real numbers

*LLM Example: the numbers below were chosen by me for illustration, then computed in Python and checked against finite differences.*

**Setup.** Input x = (1.0, 0.5), target t = 1. Layers 1 and 2 use tanh, layer 3 uses sigmoid.

| Layer | Weights (row k = destination neuron k) | Biases |
| --- | --- | --- |
| 1 | w₁₁ = 1.0, w₁₂ = 2.0; w₂₁ = −1.5, w₂₂ = 0.5 | b₁ = −0.5, b₂ = 0.2 |
| 2 | w₁₁ = 0.5, w₁₂ = −0.3; w₂₁ = 0.2, w₂₂ = 0.7 | b₁ = 0.1, b₂ = −0.2 |
| 3 | w₁₁ = 0.8, w₁₂ = −0.4 | b = 0.1 |

### Forward pass (keep every number)

```text
Layer 1: a1 = 1.0*1.0 + 2.0*0.5 - 0.5  =  1.5      z1 = tanh(1.5)   =  0.90515
         a2 = -1.5*1.0 + 0.5*0.5 + 0.2 = -1.05     z2 = tanh(-1.05) = -0.78181
Layer 2: a1 = 0.5*0.90515 - 0.3*(-0.78181) + 0.1 =  0.78712   z1 = tanh(0.78712)  =  0.65677
         a2 = 0.2*0.90515 + 0.7*(-0.78181) - 0.2 = -0.56623   z2 = tanh(-0.56623) = -0.51259
Layer 3: a  = 0.8*0.65677 - 0.4*(-0.51259) + 0.1 = 0.83045    y = sigmoid(0.83045) = 0.69645
Error:   E = 0.5*(0.69645 - 1)^2 = 0.046071
```

### Backward pass, last layer first

**Layer 3.** ∂E/∂y = y−t = −0.30355. Sigmoid steepness y(1−y) = 0.21141. So δ⁽³⁾ = −0.30355 × 0.21141 = **−0.064172**.

**Layer 3 → layer 2.** Each layer-2 neuron has one route:

```text
dE/dz1(2) = d3 * w11(3) = -0.064172 *  0.8 = -0.051338      tanh' = 1 - 0.65677^2 = 0.56865
dE/dz2(2) = d3 * w12(3) = -0.064172 * -0.4 =  0.025669      tanh' = 1 - 0.51259^2 = 0.73725
d1(2) = -0.051338 * 0.56865 = -0.029193
d2(2) =  0.025669 * 0.73725 =  0.018925
```

**Layer 2 → layer 1.** Each layer-1 neuron has two routes, and the two terms are added:

```text
dE/dz1(1) = d1(2)*w11(2) + d2(2)*w21(2) = -0.029193*0.5 + 0.018925*0.2    = -0.014597 + 0.003785 = -0.010812
dE/dz2(1) = d1(2)*w12(2) + d2(2)*w22(2) = -0.029193*(-0.3) + 0.018925*0.7 =  0.008758 + 0.013247 =  0.022005
tanh'(a1(1)) = 1 - 0.90515^2  = 0.18071      tanh'(a2(1)) = 1 - 0.78181^2 = 0.38878
d1(1) = -0.010812 * 0.18071 = -0.001954
d2(1) =  0.022005 * 0.38878 =  0.008555
```

Notice how small δ gets going backwards (−0.064 → −0.029 → −0.002 for neuron 1). Every layer multiplies by a weight and by a tanh steepness below 1. In a network with dozens of layers this shrinking is the vanishing-gradient problem, and it is part of why transformers use residual connections and normalisation.

### All 15 gradients

Each row is δ of the destination neuron × the value on the weight's own wire.

| Parameter | δ × input on its wire | ∂E/∂parameter |
| --- | --- | --- |
| w₁₁⁽³⁾ | −0.064172 × 0.65677 | −0.042147 |
| w₁₂⁽³⁾ | −0.064172 × (−0.51259) | 0.032894 |
| b⁽³⁾ | −0.064172 × 1 | −0.064172 |
| w₁₁⁽²⁾ | −0.029193 × 0.90515 | −0.026424 |
| w₁₂⁽²⁾ | −0.029193 × (−0.78181) | 0.022824 |
| w₂₁⁽²⁾ | 0.018925 × 0.90515 | 0.017130 |
| w₂₂⁽²⁾ | 0.018925 × (−0.78181) | −0.014795 |
| b₁⁽²⁾, b₂⁽²⁾ | δ₁⁽²⁾, δ₂⁽²⁾ | −0.029193, 0.018925 |
| w₁₁⁽¹⁾ | −0.001954 × 1.0 | −0.001954 |
| w₁₂⁽¹⁾ | −0.001954 × 0.5 | −0.000977 |
| w₂₁⁽¹⁾ | 0.008555 × 1.0 | 0.008555 |
| w₂₂⁽¹⁾ | 0.008555 × 0.5 | 0.004278 |
| b₁⁽¹⁾, b₂⁽¹⁾ | δ₁⁽¹⁾, δ₂⁽¹⁾ | −0.001954, 0.008555 |

### Checking the answer

A gradient can be tested without any calculus: nudge one parameter by a tiny ε = 10⁻⁵, rerun the forward pass, and compute (E(w+ε) − E(w−ε)) / 2ε. I did this for all 15 parameters. The biggest disagreement with backprop was 2.3 × 10⁻¹¹, which is rounding noise. This test is exactly why backprop is valuable: finite differences need two forward passes per parameter (30 here, billions in an LLM), while backprop got all 15 from one forward and one backward pass.

### One gradient-descent step

With learning rate η = 0.1, every parameter moves by −η × its gradient. Running the forward pass again: y goes from 0.69645 to 0.69977 (closer to the target 1) and E drops from 0.046071 to 0.045070. The step is small because the gradients are small, which is normal. Training is thousands of such steps.

## Next Up

Building toward the full LLM picture, each as its own section here as we cover it:

- **Attention** — how a token looks at other tokens before its own layer computes anything; the piece a plain feedforward net above doesn't have
- **Multi-head attention & GQA** — why Qwen3 uses fewer key/value heads than query heads
- **The transformer block** — attention + MLP + RMSNorm, stacked 28 times for Qwen3-2B
- **From logits to tokens** — softmax, sampling, the autoregressive generation loop
- **Implementing it** — writing the forward pass in PyTorch and matching activations against HuggingFace's Qwen3-2B, layer by layer
