# Multi-Layer Neural Nets, then Backpropagation

Oct 26, 2026 · @Karthikeyan

## 1.1 The network, drawn with every neuron, weight and activation

This part builds a multi-layer neural network from scratch: first the picture, then the equations neuron by neuron, then the matrices. Part 2 uses the same network to follow backpropagation. Nothing in the picture is decoration; each part of it matches one symbol in the formulas below.

&#91;embedded content: 3-3-2-1 network · 23 parameters: 17 weights, 6 biases\]

### Reading the picture

| In the picture | What it is | Symbol |
| --- | --- | --- |
| Circles on the left | the three input numbers fed to the network | x₁, x₂, x₃ |
| A grey arrow | one weight: the input value is multiplied by it on the way to the next neuron | wⱼᵢ⁽ˡ⁾ |
| Top half of a neuron | the weighted sum of everything arriving, plus the neuron's bias | aⱼ⁽ˡ⁾ |
| Bottom half of a neuron | the neuron's output: an activation function h applied to a | zⱼ⁽ˡ⁾ = h(aⱼ⁽ˡ⁾) |
| "+ b" above a neuron | its bias, added inside the weighted sum | bⱼ⁽ˡ⁾ |
| The last neuron | same two steps, but h is the sigmoid σ, squeezing the result into (0, 1) | y = σ(a⁽³⁾) |

### The numbering rule

The weight w₂₃⁽¹⁾ has three labels, and each one answers a question:

- the **bracket** (¹) says which layer the weight belongs to;
- the **first** subscript (₂) says which neuron it feeds, the destination;
- the **second** subscript (₃) says where it comes from, the source: here input x₃.

So w₂₃⁽¹⁾ is the wire from x₃ into neuron 2 of layer 1, and the picture labels exactly that wire. Destination first, source second is the order that makes the matrix form in section 1.2 work out, so it is worth fixing in your head now.

### How many numbers does the network contain?

| Layer | Shape of weights | Weights | Biases |
| --- | --- | --- | --- |
| 1 | 3 neurons × 3 inputs | 9 | 3 |
| 2 | 2 neurons × 3 inputs (the 3 layer-1 outputs) | 6 | 2 |
| 3 | 1 neuron × 2 inputs (the 2 layer-2 outputs) | 2 | 1 |
| Total |  | 17 | 6 |

That is **23 numbers** the network can adjust. Training means finding good values for them; Part 2 shows how to get the slope of the error for each one of the 23.

## 1.2 Layer 1: one equation per neuron, then the matrix

Start with just the first layer: three inputs flowing into three neurons. In the picture, every input has a wire to every neuron, so each neuron combines all three inputs.

### Step 1 — one scalar equation for each neuron

Neuron 1 collects the three arrows that end in it, multiplies each input by the weight on its arrow, adds the three products, and adds its own bias:

```latex
\begin{aligned}
a_1^{(1)} &= w_{11}^{(1)} x_1 + w_{12}^{(1)} x_2 + w_{13}^{(1)} x_3 + b_1^{(1)} \\
a_2^{(1)} &= w_{21}^{(1)} x_1 + w_{22}^{(1)} x_2 + w_{23}^{(1)} x_3 + b_2^{(1)} \\
a_3^{(1)} &= w_{31}^{(1)} x_1 + w_{32}^{(1)} x_2 + w_{33}^{(1)} x_3 + b_3^{(1)}
\end{aligned}
```

To check this against the picture, take the weight w₂₃⁽¹⁾. It is the arrow from x₃ into neuron 2, so it sits in the equation for a₂⁽¹⁾, multiplied by x₃. Every weight in the picture appears exactly once in these three equations: nine weights, three biases.

### Step 2 — notice the pattern

Stack the three equations and look at the columns. Row j is one neuron; column i is one input; the weight wⱼᵢ sits where row j meets column i. All three rows have the same shape, and only the numbers differ. That regularity is the reason matrices exist.

### Step 3 — factor into a matrix equation

Separate the coefficients (weights), the variables (inputs) and the constants (biases):

```latex
\begin{bmatrix} a_1^{(1)} \\ a_2^{(1)} \\ a_3^{(1)} \end{bmatrix} = \begin{bmatrix} w_{11}^{(1)} & w_{12}^{(1)} & w_{13}^{(1)} \\ w_{21}^{(1)} & w_{22}^{(1)} & w_{23}^{(1)} \\ w_{31}^{(1)} & w_{32}^{(1)} & w_{33}^{(1)} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} + \begin{bmatrix} b_1^{(1)} \\ b_2^{(1)} \\ b_3^{(1)} \end{bmatrix}
```

Multiplying a matrix by a vector means: take each row of the matrix, multiply it entry by entry with the vector, and add. Row 1 times x gives w₁₁x₁ + w₁₂x₂ + w₁₃x₃, which is the first equation without its bias. Rows 2 and 3 do the same for the other two. So the matrix equation is the three scalar equations, written once. In short form:

```latex
a^{(1)} = W^{(1)} x + b^{(1)}
```

| Symbol | Meaning | Shape |
| --- | --- | --- |
| x | the input vector | 3 × 1 |
| W⁽¹⁾ | weight matrix: **row j = all weights flowing into neuron j** | 3 × 3 |
| b⁽¹⁾ | bias vector | 3 × 1 |
| a⁽¹⁾ | weighted sums, one per neuron | 3 × 1 |

### Step 4 — apply the activation, neuron by neuron

Each weighted sum then goes through the activation function h on its own:

```latex
z_1^{(1)} = h\left(a_1^{(1)}\right), \qquad z_2^{(1)} = h\left(a_2^{(1)}\right), \qquad z_3^{(1)} = h\left(a_3^{(1)}\right)
```

For vectors, "h applied to a vector" simply means h applied to each entry separately:

```latex
z^{(1)} = h\left(a^{(1)}\right) = h\left(W^{(1)} x + b^{(1)}\right)
```

In this document h is tanh in layers 1 and 2. Writing the whole of layer 1 as one line is the key move: a layer is "multiply by a matrix, add a vector, bend with h". Everything in a transformer's feed-forward block is built from that same line.

### A note on the bias

Some books write the bias as a weight on an extra input that is always 1: x₀ = 1, with bⱼ = wⱼ₀. Then x = (1, x₁, x₂, x₃) and the matrix gains one column holding the biases. It is the same arithmetic, so we keep b separate: it makes the layer's two roles (mixing and shifting) easier to see, and it matches how PyTorch stores a layer.

## 1.3 Layer 1 to layer 2: carrying the notation forward

Layer 2 is built exactly like layer 1, with one change of inputs: it does not read x, it reads the **outputs of layer 1**, the vector z⁽¹⁾. The superscript on every symbol now matters, because two layers share the same letters.

| Symbol | Meaning |
| --- | --- |
| wₖⱼ⁽²⁾ | weight in layer 2 on the wire from layer-1 output j into layer-2 neuron k |
| bₖ⁽²⁾ | bias of neuron k in layer 2 |
| h⁽¹⁾, h⁽²⁾ | the activation function of layer 1 and of layer 2 (tanh for both here, but they are allowed to differ) |
| z⁽¹⁾ | output vector of layer 1, which is the input vector of layer 2 |

### Step 1 — layer 1's output, as we just derived it

```latex
z^{(1)} = \begin{bmatrix} z_1^{(1)} \\ z_2^{(1)} \\ z_3^{(1)} \end{bmatrix} = h^{(1)}\left( W^{(1)} x + b^{(1)} \right)
```

### Step 2 — one scalar equation per layer-2 neuron

Layer 2 has two neurons. Each one receives three arrows, one from each layer-1 neuron:

```latex
\begin{aligned}
a_1^{(2)} &= w_{11}^{(2)} z_1^{(1)} + w_{12}^{(2)} z_2^{(1)} + w_{13}^{(2)} z_3^{(1)} + b_1^{(2)} \\
a_2^{(2)} &= w_{21}^{(2)} z_1^{(1)} + w_{22}^{(2)} z_2^{(1)} + w_{23}^{(2)} z_3^{(1)} + b_2^{(2)}
\end{aligned}
```

Compare with layer 1: the shape is identical, and the only differences are the superscripts and that z⁽¹⁾ has taken the place of x. Check one weight against the picture: w₁₃⁽²⁾ is the arrow from layer-1 neuron 3 into layer-2 neuron 1, so it multiplies z₃⁽¹⁾ inside the equation for a₁⁽²⁾.

### Step 3 — matrix form

```latex
\begin{bmatrix} a_1^{(2)} \\ a_2^{(2)} \end{bmatrix} = \begin{bmatrix} w_{11}^{(2)} & w_{12}^{(2)} & w_{13}^{(2)} \\ w_{21}^{(2)} & w_{22}^{(2)} & w_{23}^{(2)} \end{bmatrix} \begin{bmatrix} z_1^{(1)} \\ z_2^{(1)} \\ z_3^{(1)} \end{bmatrix} + \begin{bmatrix} b_1^{(2)} \\ b_2^{(2)} \end{bmatrix}, \qquad a^{(2)} = W^{(2)} z^{(1)} + b^{(2)}
```

| Object | Shape | Why |
| --- | --- | --- |
| W⁽²⁾ | 2 × 3 | 2 neurons in this layer (rows), 3 inputs from the layer before (columns) |
| z⁽¹⁾ | 3 × 1 | the three layer-1 outputs |
| b⁽²⁾, a⁽²⁾ | 2 × 1 | one per layer-2 neuron |

The rule that carries to any layer: **W has one row per neuron in this layer and one column per neuron in the previous layer**. A (2 × 3) times (3 × 1) gives (2 × 1), and the inner 3s always agree because they count the same neurons.

### Step 4 — apply layer 2's activation

```latex
z^{(2)} = h^{(2)}\left( a^{(2)} \right) = \begin{bmatrix} h^{(2)}\left(a_1^{(2)}\right) \\ h^{(2)}\left(a_2^{(2)}\right) \end{bmatrix} = \begin{bmatrix} z_1^{(2)} \\ z_2^{(2)} \end{bmatrix}
```

### Step 5 — substitute layer 1 into layer 2

Because z⁽¹⁾ is itself a formula in x, we can replace it and get layer 2's output directly from the input:

```latex
z^{(2)} = h^{(2)}\left( W^{(2)} \cdot h^{(1)}\left( W^{(1)} x + b^{(1)} \right) + b^{(2)} \right)
```

Spelled out for a single output, to see the signal move through the nest of sums:

```latex
z_1^{(2)} = h^{(2)}\left( \sum_{j=1}^{3} w_{1j}^{(2)} \cdot h^{(1)}\left( \sum_{i=1}^{3} w_{ji}^{(1)} x_i + b_j^{(1)} \right) + b_1^{(2)} \right)
```

Read from the inside out: the inner sum is the weighted sum of neuron j in layer 1, h⁽¹⁾ turns it into that neuron's output, the outer sum mixes those three outputs into neuron 1 of layer 2, and h⁽²⁾ finishes it. This nesting is what backpropagation will have to unpack in Part 2: changing a layer-1 weight changes z⁽¹⁾, which changes **both** a₁⁽²⁾ and a₂⁽²⁾, which changes the output.

## 1.4 Layer 2 to the sigmoid output, and the whole network in one formula

The last layer is one neuron. It reads the two layer-2 outputs, forms a weighted sum, and passes it through a sigmoid so the answer lands between 0 and 1.

### Step 1 — the output neuron's weighted sum

```latex
a^{(3)} = w_{11}^{(3)} z_1^{(2)} + w_{12}^{(3)} z_2^{(2)} + b^{(3)} = W^{(3)} z^{(2)} + b^{(3)}
```

Here W⁽³⁾ is a 1 × 2 matrix (one neuron, two inputs), so a⁽³⁾ is a single number. The picture shows it as the two arrows into the orange circle.

### Step 2 — the sigmoid

```latex
y = \sigma\left(a^{(3)}\right) = \frac{1}{1 + e^{-a^{(3)}}}
```

The sigmoid maps any real number into (0, 1): large positive sums give values near 1, large negative sums near 0, and a sum of 0 gives exactly 0.5. That makes y readable as a probability, "how strongly does the network say yes".

### Step 3 — the full composite function

Stack the three layers by substituting each into the next. One scalar formula holds the entire network, with the input x on the inside and y on the outside:

```latex
y = \sigma\left( \sum_{k=1}^{2} w_{1k}^{(3)} \cdot h^{(2)}\left( \sum_{j=1}^{3} w_{kj}^{(2)} \cdot h^{(1)}\left( \sum_{i=1}^{3} w_{ji}^{(1)} x_i + b_j^{(1)} \right) + b_k^{(2)} \right) + b^{(3)} \right)
```

In matrix form it reads the same way, outermost operation last:

```latex
y = \sigma\left( W^{(3)} \; h^{(2)}\left( W^{(2)} \; h^{(1)}\left( W^{(1)} x + b^{(1)} \right) + b^{(2)} \right) + b^{(3)} \right)
```

The summary of the whole forward pass is a three-line program:

```text
z1 = h1( W1 @ x  + b1 )      # 3 numbers
z2 = h2( W2 @ z1 + b2 )      # 2 numbers
y  = sigmoid( W3 @ z2 + b3 ) # 1 number
```

### Step 4 — many examples at once: the batched form

In practice we do not push one input vector through the network at a time. We stack T inputs as the rows of a matrix X (T × 3) and process them all with one matrix multiplication per layer. PyTorch's `nn.Linear` stores W with one row per output neuron, exactly as above, and computes the product with the transpose:

```latex
Z^{(l)} = h\left( Z^{(l-1)} \left(W^{(l)}\right)^{\top} + b^{(l)} \right)
```

The row-vector convention flips the order of the product (data on the left, transposed weights on the right), but each row of the result is the same computation as the column-vector equations above. The bias vector is added to every row.

### Why this matters for the language model you are building

The feed-forward block inside every transformer layer is exactly this pattern with bigger numbers. In Qwen-style models it is three weight matrices applied to a batch of token vectors X:

```latex
\text{out} = \left( \mathrm{SiLU}\left(X W_{\text{gate}}^{\top}\right) \odot \left(X W_{\text{up}}^{\top}\right) \right) W_{\text{down}}^{\top}
```

The gate and up matrices are the "layer 1" (they widen each token vector), SiLU is the h, the ⊙ multiplies the two branches entry by entry, and the down matrix is the "layer 2" that shrinks back. These layers have no bias terms. When we reach the model code, each of these is one `nn.Linear`, and training one of them uses the backprop equations from Part 2, just with matrices instead of a handful of numbers.

## 1.5 LLM Example: the forward pass in real numbers

*I chose these numbers for illustration. Every result was computed in Python and also checked by hand against the equations above.*

Part 2 reuses this exact network and these exact values, so they are worth a careful read.

| Layer | Weights (row = destination neuron) | Biases |
| --- | --- | --- |
| 1 (tanh) | row 1: 0.2, −0.5, 0.3 · row 2: 0.4, 0.1, −0.2 · row 3: −0.3, 0.6, 0.5 | 0.1, −0.1, 0.2 |
| 2 (tanh) | row 1: 0.5, −0.4, 0.3 · row 2: −0.2, 0.6, 0.7 | 0.1, −0.1 |
| 3 (sigmoid) | row 1: 0.8, −0.6 | 0.2 |

Input x = (1.0, 0.5, −1.0). Target for the error in Part 2: t = 1.

### Layer 1

Each line is one scalar equation from section 1.2 with the numbers filled in, followed by the activation:

```text
a1 = 0.2*1.0 + (-0.5)*0.5 + 0.3*(-1.0) + 0.1  = 0.2 - 0.25 - 0.3 + 0.1  = -0.25     z1 = tanh(-0.25) = -0.244919
a2 = 0.4*1.0 +   0.1 *0.5 + (-0.2)*(-1.0) - 0.1 = 0.4 + 0.05 + 0.2 - 0.1  =  0.55     z2 = tanh( 0.55) =  0.500520
a3 = -0.3*1.0 +  0.6 *0.5 + 0.5*(-1.0) + 0.2   = -0.3 + 0.3 - 0.5 + 0.2  = -0.30     z3 = tanh(-0.30) = -0.291313
```

The matrix view of the same numbers, a⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾:

```latex
\begin{bmatrix} 0.2 & -0.5 & 0.3 \\ 0.4 & 0.1 & -0.2 \\ -0.3 & 0.6 & 0.5 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.5 \\ -1.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \\ 0.2 \end{bmatrix} = \begin{bmatrix} -0.35 \\ 0.65 \\ -0.50 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \\ 0.2 \end{bmatrix} = \begin{bmatrix} -0.25 \\ 0.55 \\ -0.30 \end{bmatrix}
```

### Layer 2

The inputs are now the three layer-1 outputs z⁽¹⁾ = (−0.244919, 0.500520, −0.291313):

```text
a1 = 0.5*(-0.244919) + (-0.4)*0.500520 + 0.3*(-0.291313) + 0.1
   = -0.122460 - 0.200208 - 0.087394 + 0.1        = -0.310062   z1 = tanh(-0.310061) = -0.300493
a2 = -0.2*(-0.244919) + 0.6*0.500520 + 0.7*(-0.291313) - 0.1
   =  0.048984 + 0.300312 - 0.203919 - 0.1        =  0.045377   z2 = tanh( 0.045377) =  0.045346
```

(The first sum prints as −0.310062 only because each product is rounded to six places; carried at full precision it is −0.310061.)

### Layer 3 and the error

```text
a  = 0.8*(-0.300493) + (-0.6)*0.045346 + 0.2 = -0.240394 - 0.027208 + 0.2 = -0.067602
y  = sigmoid(-0.067602) = 0.483106
E  = 0.5 * (0.483106 - 1)^2 = 0.5 * 0.267181 = 0.133590
```

So the untrained network answers 0.483, close to a coin flip, while the target says 1. The error is 0.1336. Part 2 answers one question about this network: for each of its 23 numbers, which direction and how hard should it be pushed to bring y toward 1?

## 2.1 Three tools you need first

Backprop is three ideas used over and over: a slope, the chain rule, and adding up routes. If these are solid, everything after this section is bookkeeping.

### Tool 1 — slope means "how much does the output move per nudge"

If a quantity E depends on a number w, the slope ∂E/∂w says: nudge w up by a tiny amount, and E changes by the slope times that amount. A positive slope means raising w raises E. A negative slope means raising w lowers E. A slope near zero means w hardly matters right now. In training we want to lower E, so we move each weight in the direction opposite to its slope.

### Tool 2 — the chain rule: slopes along a chain multiply

Suppose w feeds a, a feeds z, and z feeds the error. A nudge to w moves a, that moves z, and that moves E. Each link has its own slope, and the effects stack, so the total slope is the **product** of the link slopes.

### LLM Example: a three-link chain

*Chosen to be small enough to check by hand.*

Let a = 3w, then z = a², then E = z + 1. Take w = 2.

```text
Forward:  a = 3*2 = 6      z = 6^2 = 36      E = 37
Link slopes:  da/dw = 3      dz/da = 2a = 12      dE/dz = 1
Chain rule:   dE/dw = dE/dz * dz/da * da/dw = 1 * 12 * 3 = 36
```

Check without calculus: nudge w from 2 to 2.001. Then a = 6.003, z = 36.036009, E = 37.036009. E rose by 0.036 for a nudge of 0.001, so the slope is 0.036 / 0.001 = 36. ✓

Notice two things. We computed the slope by multiplying numbers that the forward pass had already produced (a = 6 was needed for dz/da = 2a). And we worked from the output end backwards: dE/dz first, then dz/da, then da/dw. That backward order is what the word "backpropagation" refers to.

### Tool 3 — when a value feeds two things, add the routes

Now suppose u feeds two different quantities p and q, and both feed the error. A nudge to u travels along two routes at once, and the total effect is the **sum** of the two routes. Each route is itself a chain, so multiply along it, then add across routes.

### LLM Example: two routes

*Chosen for easy arithmetic.*

Let p = 2u and q = 5u, and E = 3p + q². Take u = 1, so p = 2, q = 5.

```text
Route via p:  dE/dp * dp/du = 3     * 2 = 6
Route via q:  dE/dq * dq/du = 2q    * 5 = 10 * 5 = 50
Total:        dE/du = 6 + 50 = 56
```

Check: E as a function of u is 6u + 25u², whose slope is 6 + 50u = 56 at u = 1. ✓

In our network, each neuron in layer 1 feeds both neurons in layer 2, so this "add the routes" step will appear exactly once per layer-1 neuron. It is the part of backprop people most often get wrong, and the reason this document uses a network with two neurons per layer instead of one.

## 2.2 Backprop in plain words: the chain of blame

No symbols in this section, only the calculations in the order you perform them. Picture the network from the first section. The error sits at the far right. Training asks each wire and each bias: "how much of this error is yours, and which way should you move?"

There are only **six kinds of calculation**. Every one of the 15 results is made from them.

1. **How wrong is the output?** Take prediction minus target. This is the slope of the error with respect to the prediction. Positive means the prediction is too high; negative means too low.
2. **Blame on the output neuron's sum.** The output neuron did not hand over its raw sum; it squeezed it through a sigmoid curve first. Multiply the number from step 1 by the steepness of that curve at the point where the sum landed. Call the result the output neuron's **blame**. If the curve was flat there, a nudge to the sum would barely change the prediction, so the blame is small.
3. **Blame on a weight or bias.** A weight multiplies the value on its wire before the sum. So the weight's blame is the blame of the neuron it points into, multiplied by the value that was on that wire. If the wire carried a big value, the weight matters a lot. If it carried zero, the weight does not matter at all for this example. A bias is like a weight on a wire that always carries 1, so its blame equals the neuron's blame.
4. **Blame on a neuron's output.** Go one step left. A neuron in the previous layer sent its output down a wire into the neuron we just scored. The blame it receives is that neuron's blame multiplied by the weight on the wire. A strong wire carries strong blame back; a wire with a zero weight carries none.
5. **Add the routes.** If a neuron sent its output into more than one neuron, calculation 4 happens once per wire, and the results are added. Each wire delivers its own share of the blame; the neuron's total blame on its output is the sum of those shares.
6. **Blame on that neuron's own sum.** Now pass back through the neuron's own activation. Multiply by the steepness of its curve at the value its sum had during the forward pass (for tanh, steepness is 1 minus the output squared). The result is that neuron's blame, ready to be used in calculation 3 for the weights feeding it, and in calculation 4 for the layer before it.

The whole algorithm is then a loop, going from the last layer to the first:

```text
Start:   calculation 1, then 2   -> blame on the output neuron
For each layer, from last to first:
    calculation 3  -> gradients for every weight and bias feeding this layer
    calculation 4  -> blame arriving at the previous layer's outputs
    calculation 5  -> add up the routes
    calculation 6  -> blame on the previous layer's sums
```

Three things are worth holding on to.

- **Everything is a multiplication or an addition** of numbers the forward pass already produced. That is why one backward sweep costs about the same as one forward pass.
- **Blame only travels along existing wires**, in the reverse direction. A weight that is zero blocks the blame; a neuron that feeds nothing receives none.
- **Steepness shrinks blame.** Each time blame passes through a curve that is flatter than 1, it gets smaller. Many layers in a row can shrink it toward zero. We will meet that again as the vanishing-gradient problem.

## 2.3 The chain, stage by stage

To get the gradient of one weight, follow the dependencies from the error back to that weight. Each link in the path says "this value depends on that one", and each link has one small slope. Multiply the slopes along the path and you have the gradient. This section builds the chain for one weight of layer 2, then extends it one layer further back to a weight of layer 1.

In every diagram, the highlighted neurons and wires are the dependency being differentiated at that stage. Everything faded is not involved in that stage. Arrows point the way the forward pass flows. Index reminder: w\_jk⁽ˡ⁾ is the weight into neuron j of layer l, coming from neuron k of the layer below. No numbers appear here, only the rules; the numeric walkthrough is in 2.4.

### The path from one weight to the error

Take a layer-2 weight w\_jk⁽²⁾. It changes the sum a\_j⁽²⁾ of its neuron. That sum becomes the output z\_j⁽²⁾. That output enters the last sum a⁽³⁾ through the weight w\_1j⁽³⁾. The last sum becomes y, and y sets E. Five links, in the order the forward pass meets them:

```text
w_jk(2)  →  a_j(2)  →  z_j(2)  →  a(3)  →  y  →  E
```

The chain rule says the slope of the whole path is the product of the slopes of its links. Written from the error backwards, it is five factors, one per stage below:

```latex
\frac{\partial E}{\partial w_{jk}^{(2)}}
= \frac{\partial E}{\partial y}\cdot
  \frac{\partial y}{\partial a^{(3)}}\cdot
  \frac{\partial a^{(3)}}{\partial z_j^{(2)}}\cdot
  \frac{\partial z_j^{(2)}}{\partial a_j^{(2)}}\cdot
  \frac{\partial a_j^{(2)}}{\partial w_{jk}^{(2)}}
```

### Stage 1 — the error depends on y

&#91;embedded content: Stage 1 · E depends on y\]

E = ½(y − t)² and t is a fixed target, so E depends on y only. The slope of a square is twice the thing being squared, and the ½ cancels the 2:

```latex
\frac{\partial E}{\partial y} = \tfrac{1}{2}\cdot 2\,(y - t) = y - t
```

The ½ exists only to make this clean. If y is below the target the slope is negative: raising y would lower E.

### Stage 2 — y depends on its sum a⁽³⁾ through the sigmoid

&#91;embedded content: Stage 2 · y depends on a(3) through the sigmoid\]

The output neuron computes y = 1 / (1 + e⁻ᵃ) from its sum a⁽³⁾. Differentiating gives

```latex
\frac{dy}{da} = \frac{e^{-a}}{\left(1 + e^{-a}\right)^2}
= \underbrace{\frac{1}{1 + e^{-a}}}_{y}\cdot\underbrace{\frac{e^{-a}}{1 + e^{-a}}}_{1 - y}
= y\,(1 - y)
```

The useful part: y is already known from the forward pass, so this slope costs one multiplication. It is largest when y is 0.5 and shrinks towards zero as y nears 0 or 1.

### Stage 3 — the last sum depends on each layer-2 output

&#91;embedded content: Stage 3 · a(3) depends on each layer-2 output through its weight\]

The last sum is a⁽³⁾ = w₁₁⁽³⁾ z₁⁽²⁾ + w₁₂⁽³⁾ z₂⁽²⁾ + b⁽³⁾. Pull out one term and the rest is constant, so a⁽³⁾ changes by w\_1j⁽³⁾ for every unit that z\_j⁽²⁾ changes:

```latex
\frac{\partial a^{(3)}}{\partial z_j^{(2)}} = w_{1j}^{(3)}
```

This is what "blame passes back through a wire by multiplying with its weight" means. This is also the link you described as "y with respect to h1 and h2": it is the step from the last sum to the outputs of layer 2. The sigmoid sits in between, which is why Stage 2 is separate.

### Stage 4 — a layer-2 output depends on its own sum through tanh

&#91;embedded content: Stage 4 · a layer-2 output depends on its sum through tanh\]

For z = tanh(a) = (eᵃ − e⁻ᵃ) / (eᵃ + e⁻ᵃ), the quotient rule simplifies to

```latex
\frac{dz}{da} = 1 - \tanh^2(a) = 1 - z^2
```

Again no extra work, because z is stored from the forward pass. Each layer-2 neuron has its own z\_j and its own slope. This is a separate link per neuron, which is why the diagram highlights neuron 1 only.

### Stage 5 — a layer-2 sum depends on its weights; the slope is the value on the wire

&#91;embedded content: Stage 5 · a layer-2 sum depends on its weights; the slope is the value on the wire\]

A layer-2 sum is a\_j⁽²⁾ = Σₖ w\_jk⁽²⁾ z\_k⁽¹⁾ + b\_j⁽²⁾. Pull out one weight and the rest is constant, so a\_j⁽²⁾ changes by z\_k⁽¹⁾ for every unit that w\_jk⁽²⁾ changes. The bias is a weight on a wire carrying the constant 1:

```latex
\frac{\partial a_j^{(2)}}{\partial w_{jk}^{(2)}} = z_k^{(1)},
\qquad
\frac{\partial a_j^{(2)}}{\partial b_j^{(2)}} = 1
```

So the slope of a weight is the value that arrives on its wire. A weight on a wire carrying a large value matters a lot; a weight on a wire carrying zero cannot matter at all.

### Multiply the five stages

Stages 1 to 5 give every factor of the chain:

```latex
\frac{\partial E}{\partial w_{jk}^{(2)}}
= \underbrace{(y-t)\;y(1-y)}_{\delta^{(3)}}\;\cdot\; w_{1j}^{(3)}\;\bigl(1 - {z_j^{(2)}}^{2}\bigr)\;\cdot\; z_k^{(1)}
```

Grouping the factors from the left gives the quantities the rest of Part 2 is built on:

```latex
\delta^{(3)} = (y - t)\,y(1 - y),\qquad
\delta_j^{(2)} = \delta^{(3)}\,w_{1j}^{(3)}\,\bigl(1 - {z_j^{(2)}}^{2}\bigr),\qquad
\frac{\partial E}{\partial w_{jk}^{(2)}} = \delta_j^{(2)}\,z_k^{(1)}
```

### Stage 6 — one layer further back: the routes

&#91;embedded content: Stage 6 · one layer-1 output feeds two layer-2 sums: two routes\]

Now aim at a layer-1 weight. The error has to reach z\_k⁽¹⁾ first. But z\_k⁽¹⁾ is an input to both layer-2 sums, so it affects E by two routes: one through a₁⁽²⁾ and one through a₂⁽²⁾. Each route has the same two links as before (Stage 4 and Stage 3, applied to layer 2), and the slope across the wire is w\_jk⁽²⁾ as in Stage 3. When one value affects E by several routes, you add the routes:

```latex
\frac{\partial E}{\partial z_k^{(1)}} = \sum_j \delta_j^{(2)}\,w_{jk}^{(2)}
```

### Stage 7 — a layer-1 sum depends on its weights

&#91;embedded content: Stage 7 · a layer-1 sum depends on its weights; the slope is the input on the wire\]

From z\_k⁽¹⁾ back to a\_k⁽¹⁾ is the tanh rule of Stage 4 again. The weights of layer 1 sit on wires that carry the inputs x\_m, so by the same argument as Stage 5:

```latex
\frac{\partial a_k^{(1)}}{\partial w_{km}^{(1)}} = x_m
```

Putting Stages 6 and 7 together:

```latex
\delta_k^{(1)} = \Bigl(\sum_j \delta_j^{(2)}\,w_{jk}^{(2)}\Bigr)\bigl(1 - {z_k^{(1)}}^{2}\bigr),
\qquad
\frac{\partial E}{\partial w_{km}^{(1)}} = \delta_k^{(1)}\,x_m
```

### All stages in one table

| Stage | Link (value ← what it depends on) | Slope |
| --- | --- | --- |
| 1 | E ← y | y − t |
| 2 | y ← a⁽³⁾ | y (1 − y) |
| 3 | a⁽³⁾ ← z\_j⁽²⁾ | w\_1j⁽³⁾ |
| 4 | z\_j⁽²⁾ ← a\_j⁽²⁾ (same rule for every tanh neuron) | 1 − (z\_j⁽²⁾)² |
| 5 | a\_j⁽²⁾ ← w\_jk⁽²⁾ | z\_k⁽¹⁾ |
| 6 | a\_j⁽²⁾ ← z\_k⁽¹⁾ (one factor per route, add the routes) | w\_jk⁽²⁾ |
| 7 | a\_k⁽¹⁾ ← w\_km⁽¹⁾ | x\_m |

### What the curves' steepness looks like

Both curves are steep near zero and flat in the tails. That decides how much blame survives passing through a neuron.

| Activation | Sum a | Output | Steepness |
| --- | --- | --- | --- |
| tanh | 0 | 0 | 1.000 |
| tanh | 1 | 0.762 | 0.420 |
| tanh | 3 | 0.995 | 0.010 |
| sigmoid | 0 | 0.500 | 0.250 |
| sigmoid | 2 | 0.881 | 0.105 |
| sigmoid | 6 | 0.998 | 0.002 |

A neuron whose sum is far from zero is "saturated": it lets almost no blame through, so its weights barely learn. Sigmoid is worst, since even its best steepness is only 0.25.

## 2.4 Walkthrough: every gradient of the 3-3-2-1 network, with real numbers

Now the six calculations from section 2.2 are applied to the network of Part 1, using the forward-pass numbers from section 1.5. For each step we write the equation first, then the arithmetic. To keep the equations short, δ stands for "the blame on a neuron's weighted sum": δ = ∂E/∂a. It is the one number that every weight in front of that neuron will reuse.

The order is the order of the work: output neuron first, then layer 2, then layer 1.

Here is the finished answer on the network drawing before we derive it. Each neuron shows its error δ, and each wire shows the gradient of the weight on it. Steps 1 to 6 below produce these numbers in order, starting from the output on the right.

&#91;embedded content: backprop on the 3-3-2-1 network · δ on every neuron, gradient on every wire\]

### Step 1 — the output neuron's blame (calculations 1 and 2)

```latex
\frac{\partial E}{\partial y} = y - t, \qquad \delta^{(3)} = \frac{\partial E}{\partial y} \cdot y\,(1 - y)
```

```text
dE/dy            = 0.483106 - 1            = -0.516894    (prediction too low)
steepness y(1-y) = 0.483106 * 0.516894     =  0.249715    (sigmoid near its steepest point)
delta3           = -0.516894 * 0.249715     = -0.129076
```

### Step 2 — the three parameters of layer 3 (calculation 3)

The output neuron's sum is a = w₁₁z₁ + w₁₂z₂ + b, with z₁ = −0.300493 and z₂ = 0.045346 coming from layer 2. By Stage 5 of 2.3 each weight's slope is the value on its own wire times the neuron's blame:

```latex
\frac{\partial E}{\partial w_{11}^{(3)}} = \delta^{(3)} z_1^{(2)}, \qquad \frac{\partial E}{\partial w_{12}^{(3)}} = \delta^{(3)} z_2^{(2)}, \qquad \frac{\partial E}{\partial b^{(3)}} = \delta^{(3)}
```

```text
dE/dw11(3) = -0.129076 * -0.300493 = +0.038786
dE/dw12(3) = -0.129076 *  0.045346 = -0.005853
dE/db(3)   = -0.129076
```

Reading the signs: w₁₁⁽³⁾ has a positive slope, so lowering it lowers the error, and gradient descent will lower it. The reason is that z₁ is negative: making w₁₁ smaller makes the product w₁₁z₁ larger, which pushes a up and y toward the target. w₁₂⁽³⁾ has a tiny slope because its wire carried almost nothing (0.045). The bias slope is negative, so raising b raises y, which is what we want.

### Step 3 — blame moves back to layer 2 (calculations 4 and 6)

Each layer-2 neuron has exactly one wire into the output neuron, so there is nothing to add yet. The blame on a layer-2 output is δ⁽³⁾ times the weight on that wire. Then it passes through the neuron's own tanh steepness:

```latex
\frac{\partial E}{\partial z_k^{(2)}} = \delta^{(3)} w_{1k}^{(3)}, \qquad \delta_k^{(2)} = \frac{\partial E}{\partial z_k^{(2)}} \cdot \left(1 - {z_k^{(2)}}^2\right)
```

```text
neuron 1:  dE/dz = -0.129076 *  0.8 = -0.103261     steepness 1 - (-0.300493)^2 = 0.909704
           delta1(2) = -0.103261 * 0.909704 = -0.093937
neuron 2:  dE/dz = -0.129076 * -0.6 = +0.077446     steepness 1 - ( 0.045346)^2 = 0.997944
           delta2(2) = +0.077446 * 0.997944 = +0.077286
```

### Step 4 — the eight parameters of layer 2 (calculation 3 again)

These weights sit on wires that carry the layer-1 outputs z⁽¹⁾ = (−0.244919, 0.500520, −0.291313). Each weight uses the blame of the neuron it points into:

```text
weights into layer-2 neuron 1 (delta = -0.093937):
  dE/dw11(2) = -0.093937 * -0.244919 = +0.023007
  dE/dw12(2) = -0.093937 *  0.500520 = -0.047017
  dE/dw13(2) = -0.093937 * -0.291313 = +0.027365
  dE/db1(2)  = -0.093937
weights into layer-2 neuron 2 (delta = +0.077286):
  dE/dw21(2) = +0.077286 * -0.244919 = -0.018929
  dE/dw22(2) = +0.077286 *  0.500520 = +0.038683
  dE/dw23(2) = +0.077286 * -0.291313 = -0.022514
  dE/db2(2)  = +0.077286
```

### Step 5 — blame moves back to layer 1, and the routes add (calculations 4, 5 and 6)

This is the step that uses Tool 3 from section 2.1. Each layer-1 neuron sends its output to **both** layer-2 neurons, along two different wires. Each wire returns its own share of the blame, and the shares add:

```latex
\frac{\partial E}{\partial z_j^{(1)}} = \sum_{k=1}^{2} \delta_k^{(2)} \, w_{kj}^{(2)}
```

```text
layer-1 neuron 1 (wires w11(2) = 0.5 and w21(2) = -0.2):
  via layer-2 neuron 1:  delta1(2) * w11(2) = -0.093937 *  0.5 = -0.046968
  via layer-2 neuron 2:  delta2(2) * w21(2) = +0.077286 * -0.2 = -0.015457
  sum:  dE/dz1(1) = -0.062426
layer-1 neuron 2 (wires w12(2) = -0.4 and w22(2) = 0.6):
  via layer-2 neuron 1:  delta1(2) * w12(2) = -0.093937 * -0.4 = +0.037575
  via layer-2 neuron 2:  delta2(2) * w22(2) = +0.077286 *  0.6 = +0.046372
  sum:  dE/dz2(1) = +0.083947
layer-1 neuron 3 (wires w13(2) = 0.3 and w23(2) = 0.7):
  via layer-2 neuron 1:  delta1(2) * w13(2) = -0.093937 *  0.3 = -0.028181
  via layer-2 neuron 2:  delta2(2) * w23(2) = +0.077286 *  0.7 = +0.054100
  sum:  dE/dz3(1) = +0.025919
```

Neuron 3 shows why the adding matters. Its two routes disagree: one says "lower z₃" (−0.0282) and the other says "raise it" (+0.0541). The second route is stronger, so the total is positive. If you forgot the second route you would get −0.0282, which has the **wrong sign**, and gradient descent would push that neuron the wrong way.

Now each sum passes through the neuron's own tanh steepness:

```text
neuron 1: steepness 1 - (-0.244919)^2 = 0.940015     delta1(1) = -0.062426 * 0.940015 = -0.058681
neuron 2: steepness 1 - ( 0.500520)^2 = 0.749480     delta2(1) = +0.083947 * 0.749480 = +0.062916
neuron 3: steepness 1 - (-0.291313)^2 = 0.915137     delta3(1) = +0.025919 * 0.915137 = +0.023720
```

### Step 6 — the twelve parameters of layer 1 (calculation 3 one last time)

The wires into layer 1 carry the raw inputs x = (1.0, 0.5, −1.0), so each row of weight gradients is that neuron's δ times the input vector:

```text
weights into layer-1 neuron 1 (delta = -0.058681):
  dE/dw11(1) = -0.058681 *  1.0 = -0.058681
  dE/dw12(1) = -0.058681 *  0.5 = -0.029341
  dE/dw13(1) = -0.058681 * -1.0 = +0.058681
  dE/db1(1)  = -0.058681
weights into layer-1 neuron 2 (delta = +0.062916):
  dE/dw21(1) = +0.062916 *  1.0 = +0.062916
  dE/dw22(1) = +0.062916 *  0.5 = +0.031458
  dE/dw23(1) = +0.062916 * -1.0 = -0.062916
  dE/db2(1)  = +0.062916
weights into layer-1 neuron 3 (delta = +0.023720):
  dE/dw31(1) = +0.023720 *  1.0 = +0.023720
  dE/dw32(1) = +0.023720 *  0.5 = +0.011860
  dE/dw33(1) = +0.023720 * -1.0 = -0.023720
  dE/db3(1)  = +0.023720
```

The first and third columns have equal size and opposite sign because x₃ = −x₁ for this input: the same blame, multiplied by values of opposite sign.

### All 23 results in one table

| Layer | Neuron | δ of the neuron | ∂E/∂w (inputs on the wires: first, second, third) | ∂E/∂b |
| --- | --- | --- | --- | --- |
| 3 | output | −0.129076 | +0.038786, −0.005853 | −0.129076 |
| 2 | 1 | −0.093937 | +0.023007, −0.047017, +0.027365 | −0.093937 |
| 2 | 2 | +0.077286 | −0.018929, +0.038683, −0.022514 | +0.077286 |
| 1 | 1 | −0.058681 | −0.058681, −0.029341, +0.058681 | −0.058681 |
| 1 | 2 | +0.062916 | +0.062916, +0.031458, −0.062916 | +0.062916 |
| 1 | 3 | +0.023720 | +0.023720, +0.011860, −0.023720 | +0.023720 |

Counting: 2 + 6 + 9 = 17 weight gradients and 1 + 2 + 3 = 6 bias gradients, 23 in all. Each weight gradient took one multiplication; the rest of the work was the blame values δ and the route sums.

### One gradient-descent step

With learning rate η = 0.1 every parameter moves by −η times its slope. Re-running the forward pass with the 23 updated numbers gives y = 0.495656 (it was 0.483106, target 1) and E = 0.127181 (it was 0.133590). The step is small because the slopes are small, which is normal: training is many such steps.

## 2.5 Adding layer notation: one rule for every layer

The walkthrough repeated the same three moves in every layer. Symbols let us write each move once, for any layer and any number of neurons. They are the same symbols as in Part 1, plus δ.

| Symbol | Meaning | In our network |
| --- | --- | --- |
| L | number of layers | 3 |
| l | layer number; layer 0 is the input, z⁽⁰⁾ = x | 1, 2 or 3 |
| Mₗ | number of neurons in layer l | M₀ = 3, M₁ = 3, M₂ = 2, M₃ = 1 |
| j, k | neuron numbers. j is a neuron in layer l, k a neuron in layer l+1 |  |
| wₖⱼ⁽ˡ⁾ | weight on the wire from neuron j of layer l−1 into neuron k of layer l |  |
| aⱼ⁽ˡ⁾, zⱼ⁽ˡ⁾ | weighted sum and activation output of neuron j in layer l |  |
| h⁽ˡ⁾ | activation function of layer l | tanh, tanh, sigmoid |
| δⱼ⁽ˡ⁾ | the blame ∂E/∂aⱼ⁽ˡ⁾ on neuron j's weighted sum |  |

The forward pass in these symbols (from sections 1.2 to 1.4) is aⱼ⁽ˡ⁾ = Σᵢ wⱼᵢ⁽ˡ⁾ zᵢ⁽ˡ⁻¹⁾ + bⱼ⁽ˡ⁾ and zⱼ⁽ˡ⁾ = h⁽ˡ⁾(aⱼ⁽ˡ⁾).

### Move A — start at the output (calculations 1 and 2)

```latex
\delta^{(L)} = \frac{\partial E}{\partial y} \cdot h'^{(L)}\left(a^{(L)}\right)
```

In our network (L = 3, sigmoid, squared error) this is (y − t) · y(1 − y), the −0.129076 from Step 1. Whatever loss function and output activation you use, only this line changes; everything below stays the same. (For a softmax output with cross-entropy loss it collapses to just y − t, one reason that pairing is so common in language models.)

### Move B — turn blame into gradients (calculation 3)

For every layer l, including the last:

```latex
\frac{\partial E}{\partial w_{kj}^{(l)}} = \delta_k^{(l)} \; z_j^{(l-1)}, \qquad \frac{\partial E}{\partial b_k^{(l)}} = \delta_k^{(l)}
```

In words: the blame of the neuron the wire points into, times the value the wire carried. The first subscript k is the destination and the second subscript j is the source, matching the weight's own subscripts, so the formula can be read straight off the picture. For layer 1, z⁽⁰⁾ is the input x.

### Move C — pass blame one layer back (calculations 4, 5 and 6)

For l = L−1 down to 1:

```latex
\delta_j^{(l)} = \left( \sum_{k=1}^{M_{l+1}} w_{kj}^{(l+1)} \, \delta_k^{(l+1)} \right) \cdot h'^{(l)}\left(a_j^{(l)}\right)
```

Read it from the inside out. The sum runs over every neuron k in the next layer that neuron j feeds. Each term is the next layer's blame times the weight on the wire from j to k (calculation 4). The sum is the adding of routes (calculation 5). The last factor is the neuron's own steepness (calculation 6). Notice the index pattern: the weight has **k first, j second**, but we sum over k. That reads the weight matrix down a column instead of along a row, which is exactly what "transpose" will mean in the next section.

### Check the notation against the numbers

Take layer 1, neuron j = 3, which feeds both layer-2 neurons (k = 1, 2):

```text
delta3(1) = ( w13(2)*delta1(2) + w23(2)*delta2(2) ) * tanh'(a3(1))
          = ( 0.3*(-0.093937) + 0.7*(0.077286) ) * 0.915137
          = ( -0.028181 + 0.054100 ) * 0.915137
          =    0.025919 * 0.915137 = +0.023720
```

The same number as in Step 5 of the walkthrough. For the output layer, Move A gives δ⁽³⁾ = −0.516894 × 0.249715 = −0.129076, and Move B for w₁₂⁽³⁾ gives −0.129076 × 0.045346 = −0.005853, again matching.

### The whole procedure as a loop

```text
forward:   for l = 1..L:  a(l)_j = sum_i w(l)_ji * z(l-1)_i + b(l)_j ;  z(l)_j = h(l)(a(l)_j)    (keep every a and z)
start:     delta(L) = dE/dy * h'(a(L))
backward:  for l = L down to 1:
               grad w(l)_kj = delta(l)_k * z(l-1)_j        grad b(l)_k = delta(l)_k
               if l > 1:  delta(l-1)_j = ( sum_k w(l)_kj * delta(l)_k ) * h'(a(l-1)_j)
```

## 2.6 Matrix form: the same moves, a whole layer at a time

Move B and Move C each loop over neurons. Stacking the neurons of a layer into vectors turns each loop into one matrix operation, which is what PyTorch executes. Nothing new is calculated; it is the same arithmetic, grouped.

### Shapes

| Object | Shape | Layer 1 | Layer 2 | Layer 3 |
| --- | --- | --- | --- | --- |
| z⁽ˡ⁾, a⁽ˡ⁾, δ⁽ˡ⁾, b⁽ˡ⁾ (vectors) | Mₗ × 1 | 3 × 1 | 2 × 1 | 1 × 1 |
| W⁽ˡ⁾ (row k = weights into neuron k) | Mₗ × Mₗ₋₁ | 3 × 3 | 2 × 3 | 1 × 2 |
| ∂E/∂W⁽ˡ⁾ | same as W⁽ˡ⁾ | 3 × 3 | 2 × 3 | 1 × 2 |

The gradient of a matrix always has the shape of that matrix, one entry per weight. That single rule catches most shape mistakes.

### Move A in matrix form

```latex
\delta^{(L)} = \frac{\partial E}{\partial y} \odot h'\left(a^{(L)}\right)
```

The symbol ⊙ means "multiply entry by entry". With one output neuron this is just the number −0.129076.

### Move B as an outer product

The gradient of weight (k, j) is δₖ times zⱼ. Written for all k and j at once, it is a column times a row:

```latex
\frac{\partial E}{\partial W^{(l)}} = \delta^{(l)} \left( z^{(l-1)} \right)^{\top}, \qquad \frac{\partial E}{\partial b^{(l)}} = \delta^{(l)}
```

For layer 1 (δ⁽¹⁾ = (−0.058681, 0.062916, 0.023720), x = (1.0, 0.5, −1.0)) the outer product is a 3 × 3 matrix, one entry per layer-1 weight, in the same positions as the weights themselves:

```latex
\begin{bmatrix} -0.058681 \\ 0.062916 \\ 0.023720 \end{bmatrix} \begin{bmatrix} 1.0 & 0.5 & -1.0 \end{bmatrix} = \begin{bmatrix} -0.058681 & -0.029341 & 0.058681 \\ 0.062916 & 0.031458 & -0.062916 \\ 0.023720 & 0.011860 & -0.023720 \end{bmatrix}
```

These are the nine layer-1 weight gradients from Step 6. Layer 2 works the same way with δ⁽²⁾ (2 × 1) and z⁽¹⁾ transposed (1 × 3), giving a 2 × 3 matrix.

### Move C as a transposed matrix product

The route sum Σₖ wₖⱼ⁽ˡ⁺¹⁾ δₖ⁽ˡ⁺¹⁾ adds down column j of W⁽ˡ⁺¹⁾. Multiplying by the transpose reads columns as rows, so the whole layer is one product:

```latex
\delta^{(l)} = \left( {W^{(l+1)}}^{\top} \delta^{(l+1)} \right) \odot h'\left(a^{(l)}\right)
```

Going from layer 2 back to layer 1 in our network. W⁽²⁾ is 2 × 3, so its transpose is 3 × 2, and (3 × 2) times (2 × 1) gives the three route sums at once:

```latex
{W^{(2)}}^{\top} \delta^{(2)} = \begin{bmatrix} 0.5 & -0.2 \\ -0.4 & 0.6 \\ 0.3 & 0.7 \end{bmatrix} \begin{bmatrix} -0.093937 \\ 0.077286 \end{bmatrix} = \begin{bmatrix} -0.062426 \\ 0.083947 \\ 0.025919 \end{bmatrix}
```

```latex
\delta^{(1)} = \begin{bmatrix} -0.062426 \\ 0.083947 \\ 0.025919 \end{bmatrix} \odot \begin{bmatrix} 0.940015 \\ 0.749480 \\ 0.915137 \end{bmatrix} = \begin{bmatrix} -0.058681 \\ 0.062916 \\ 0.023720 \end{bmatrix}
```

Compare with Step 5: the three numbers in the middle vector are the summed routes, and the final vector is δ⁽¹⁾. Row 1 of Wᵀ is column 1 of W⁽²⁾ = (0.5, −0.2), exactly the pair of weights leaving layer-1 neuron 1. This is what "transpose" means here: the forward pass uses W to combine many inputs into one neuron, and the backward pass uses Wᵀ to spread one neuron's blame back over its many inputs.

### The full backward sweep, layer by layer

| Layer | Blame δ | Weight gradient | Bias gradient |
| --- | --- | --- | --- |
| 3 (output) | (∂E/∂y) ⊙ σ′(a⁽³⁾) | δ⁽³⁾ (z⁽²⁾)ᵀ | δ⁽³⁾ |
| 2 | ((W⁽³⁾)ᵀ δ⁽³⁾) ⊙ (1 − z⁽²⁾²) | δ⁽²⁾ (z⁽¹⁾)ᵀ | δ⁽²⁾ |
| 1 | ((W⁽²⁾)ᵀ δ⁽²⁾) ⊙ (1 − z⁽¹⁾²) | δ⁽¹⁾ xᵀ | δ⁽¹⁾ |

As a short program, with the shapes in comments:

```text
delta3  = (y - t) * y * (1 - y)                    # 1x1
gradW3  = delta3 @ z2.T      gradb3 = delta3       # 1x2, 1x1
delta2  = (W3.T @ delta3) * (1 - z2**2)            # 2x1
gradW2  = delta2 @ z1.T      gradb2 = delta2       # 2x3, 2x1
delta1  = (W2.T @ delta2) * (1 - z1**2)            # 3x1
gradW1  = delta1 @ x.T       gradb1 = delta1       # 3x3, 3x1
```

Six lines for the whole network, and every line is one matrix product or one entry-wise multiply. The forward pass uses W; the backward pass uses Wᵀ. That symmetry is why a backward sweep costs about twice a forward pass: per layer, one product for δ and one for the weight gradient, against one product forward.

## 2.7 Checking the answer, cost, and what can go wrong

Backprop is a pile of hand-derived formulas, so you need a way to know they are right. The check is simple and needs no calculus: nudge one weight a tiny bit, rerun the forward pass, and see how much E moved.

**Finite-difference check** (this is the slope idea from 2.1, done numerically):

- Take w11 of layer 3 (the weight from layer-2 neuron 1 into the output), currently 0.8. Use ε = 0.00001.
- Set it to 0.80001, run the forward pass: E⁺ = 0.13359009545.
- Set it to 0.79999, run the forward pass: E⁻ = 0.13358931972.
- Slope = (E⁺ − E⁻) / (2ε) = 0.0000007757 / 0.00002 = **+0.038786**.
- Backprop said ∂E/∂w11⁽³⁾ = δ₁⁽³⁾ × z₁⁽²⁾ = (−0.129076) × (−0.300493) = **+0.038786**. They agree.

The same test on a layer-1 weight, w31 of layer 1 (input x₁ into layer-1 neuron 3, the one with two routes): finite difference gives +0.023720, backprop gives δ₃⁽¹⁾ × x₁ = 0.023720 × 1.0 = **+0.023720**. Agree again. Run the same test on all 23 parameters and the worst disagreement is 5.0 × 10⁻¹¹, which is rounding noise.

**Why we do not just use finite differences for training.** Count the work:

- Finite differences: each of the 23 parameters needs 2 forward passes, so 46 forward passes per update.
- Backprop: 1 forward pass plus 1 backward pass. The backward pass costs about the same as 2 forward passes, so roughly 3 forward passes in total.

The gap grows with size. Finite differences cost 2 forward passes per parameter, so a model with a billion weights would need two billion forward passes for one update. Backprop still costs about 3, whether the model has 23 parameters or a billion. That is the whole reason it exists. Finite differences remain useful as a unit test, on a tiny network, to catch a mistake in your backward code.

**Why activations must be stored during training.** Look back at the two formulas the backward pass uses:

- weight gradient = δ\_destination × z\_source, so it needs the z of the layer below;
- tanh′(a) = 1 − z², so it needs the z of the same layer.

During inference you compute layer 1, use its output for layer 2, and throw layer 1's output away. During training you must keep z⁽¹⁾, z⁽²⁾ and y until the backward pass has used them. For a big model with a long input, this stored activation memory is often the biggest memory cost, which is why tricks like gradient checkpointing (recompute some activations instead of storing them) exist.

**Gradients shrink as they travel backwards.** Follow the error along one chain in our network: |δ| at the output is 0.129, at layer-2 neuron 1 it is 0.094, at layer-1 neuron 1 it is 0.059. Each step multiplies by a weight and by a steepness (h′), and both can be below 1. In 2.3 the steepness table showed sigmoid′ never exceeds 0.25 and tanh′ falls toward 0 when a neuron is saturated (|z| near 1). Stack many such layers and δ at the early layers can become tiny, so those weights barely learn. This is the vanishing-gradient problem. It is one reason modern LLM layers use residual connections and non-saturating activations such as SiLU.

**LLM Example: four mistakes that give wrong gradients.** Each one is easy to make, and each is caught by the finite-difference test:

- **Forgetting the route sum.** Layer-1 neuron 3 feeds both layer-2 neurons. ∂E/∂z₃⁽¹⁾ = 0.3 × (−0.093937) + 0.7 × 0.077286 = −0.028181 + 0.054100 = **+0.025919**. Keeping only the first route gives −0.028181, which has the wrong sign, so that weight would be pushed the wrong way.
- **Transpose or index mix-up.** Going backwards uses W⁽²⁾ᵀ δ⁽²⁾ (shape 3×2 times 2×1), not W⁽²⁾ δ⁽²⁾, which does not even have matching shapes. If your code runs only because the layer is square, you may be silently using the wrong weights.
- **Evaluating the steepness at the wrong point.** h′ is taken at that neuron's own a (or equivalently its own z), not at the neuron above or below.
- **Sign of (y − t).** δ⁽³⁾ starts as (y − t) × y(1 − y). Here y − t = 0.483106 − 1 = −0.516894, negative because the output is below the target. Flip it and every gradient flips sign, and training makes E larger.

## 2.8 PyTorch autograd, the Qwen3 layers, and practice

Everything in 2.4 to 2.6 is what `loss.backward()` does for you. PyTorch records each operation during the forward pass (matrix multiply, add, tanh, sigmoid, square). When you call `backward()`, it walks that record in reverse and applies the rule for each operation: the chain rule from 2.1 applied one operation at a time. For a linear layer a = W z + b the two rules are exactly the ones you derived: the gradient for W is δ zᵀ, and the error passed down to the layer below is Wᵀ δ.

**The same 3-3-2-1 network in PyTorch.** This code was not run in this workspace (PyTorch could not be installed here). The expected values below are the ones I computed and finite-difference checked in NumPy, so if your run prints something different, suspect a typo in the code first.

```python
import torch

x  = torch.tensor([1.0, 0.5, -1.0])
t  = torch.tensor([1.0])

W1 = torch.tensor([[0.2, -0.5, 0.3], [0.4, 0.1, -0.2], [-0.3, 0.6, 0.5]], requires_grad=True)
b1 = torch.tensor([0.1, -0.1, 0.2], requires_grad=True)
W2 = torch.tensor([[0.5, -0.4, 0.3], [-0.2, 0.6, 0.7]], requires_grad=True)
b2 = torch.tensor([0.1, -0.1], requires_grad=True)
W3 = torch.tensor([[0.8, -0.6]], requires_grad=True)
b3 = torch.tensor([0.2], requires_grad=True)

z1 = torch.tanh(W1 @ x  + b1)        # layer 1
z2 = torch.tanh(W2 @ z1 + b2)        # layer 2
y  = torch.sigmoid(W3 @ z2 + b3)     # layer 3
E  = 0.5 * ((y - t) ** 2).sum()

E.backward()
print(E.item(), y.item())   # expect 0.133590, 0.483106
print(W3.grad)              # expect [[ 0.038786, -0.005853]]
print(W2.grad)              # expect [[ 0.023007, -0.047017,  0.027365], [-0.018929,  0.038683, -0.022514]]
print(W1.grad)              # expect [[-0.058681, -0.029341,  0.058681], [ 0.062916,  0.031458, -0.062916], [ 0.023720,  0.011860, -0.023720]]
print(b1.grad)              # expect [-0.058681,  0.062916,  0.023720]
```

How this maps to the notes:

- `W1 @ x + b1` is a⁽¹⁾ = W⁽¹⁾ x + b⁽¹⁾, with W⁽¹⁾ shaped 3×3 (neurons in this layer × neurons in the layer below), as in 1.2.
- `W2.grad` has the same shape as `W2`, and its entry at row k, column j is δₖ⁽²⁾ × zⱼ⁽¹⁾ from 2.4.
- `b1.grad` equals δ⁽¹⁾, since a bias is a weight whose input is always 1.
- You never write δ yourself. It exists inside autograd as the gradient that flows backwards through the tanh and sigmoid steps.

**`nn.Linear` is one layer of this.** `torch.nn.Linear(3, 3)` holds a `weight` of shape (out, in), here 3×3, and a `bias`. For a batch X of shape (batch, in) it computes `X @ weight.T + bias`, which is the batched form from 1.4. Its backward pass uses two matrix multiplies: `grad_out.T @ X` for the weight gradient and `grad_out @ weight` for the error sent to the layer below. That is the matrix form from 2.6, with the batch summed in.

**Link to Qwen3.** A Qwen-style transformer block contains an MLP that is the same machinery with a different layer layout: out = down( SiLU(gate(x)) ⊙ up(x) ), where gate, up and down are three bias-free linear layers and ⊙ is element-wise multiplication. Backprop through it needs nothing new. The gradient splits at the ⊙ into two branches (one into `gate` through SiLU, one into `up`), and the two branches add back together where they both read x, which is the same route-sum rule you used for layer-1 neuron 3. I have not pinned down the exact layer sizes for the 2B model you are targeting, so I am not quoting them here. Read them from the model's `config.json` when you start building.

**Practice: a 1-1-1 network by hand.** Input x = 1, target t = 0. Hidden neuron: w₁ = 0.5, b₁ = 0, tanh. Output neuron: w₂ = −1.0, b₂ = 0, sigmoid. Loss E = ½(y − t)². Compute the forward pass, then ∂E/∂w₂ and ∂E/∂w₁. Try it before reading the answers.

**Answers:**

- Forward: a₁ = 0.5, z₁ = tanh(0.5) = 0.462117, a₂ = −1.0 × 0.462117 = −0.462117, y = sigmoid(−0.462117) = 0.386484, E = 0.074685.
- Output error: δ₂ = (y − t) × y(1 − y) = 0.386484 × 0.386484 × 0.613516 = 0.091641.
- ∂E/∂w₂ = δ₂ × z₁ = 0.091641 × 0.462117 = **+0.042349**.
- Hidden error: δ₁ = (w₂ × δ₂) × (1 − z₁²) = (−1.0 × 0.091641) × 0.786448 = −0.072071.
- ∂E/∂w₁ = δ₁ × x = **−0.072071**.
- Finite-difference check gives −0.0720706 and +0.0423487, so both agree.

Once that works, change t to 1 and see which signs flip. Then add a second hidden neuron feeding the output, and you will need the route sum from 2.4.
