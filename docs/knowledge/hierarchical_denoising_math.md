# Hierarchical Multi-Resolution Quantum Image Representation
## Mathematical Foundations for Seam-Aware Denoising

> **Abstract**: This document derives the mathematical foundations for denoising algorithms that exploit the hierarchical position encoding in MHRQI and MHRQIB quantum image representations. We address the critical issue of *seam artifacts* that arise at hierarchical block boundaries and propose seam-aware adaptations of classical denoising methods (NL-Means, BM3D, SRAD) for quantum circuits.

---

## 1. Hierarchical Position Encoding

### 1.1 Qudit-Based Coordinate Representation

In MHRQI, an $N \times N$ image where $N = d^L$ is encoded using $2L$ position qudits (dimension $d$), where $L = \log_d N$ is the number of hierarchy levels.

**Definition (Hierarchical Coordinate Vector):**
For a pixel at position $(r, c)$, the *hierarchical coordinate vector* (HCV) is:

$$\mathbf{h}(r,c) = (q_{y,0}, q_{x,0}, q_{y,1}, q_{x,1}, \ldots, q_{y,L-1}, q_{x,L-1})$$

where $q_{y,k}$ and $q_{x,k}$ are the $k$-th level qudit values (each in $\{0, 1, \ldots, d-1\}$).

The coordinate decomposition follows:

$$r = \sum_{k=0}^{L-1} q_{y,k} \cdot d^{L-1-k}$$

$$c = \sum_{k=0}^{L-1} q_{x,k} \cdot d^{L-1-k}$$

### 1.2 Hierarchical Block Structure

**Definition (Level-k Block):**
A level-$k$ block is defined by fixing the first $k$ pairs of position qudit values. It contains $d^{2(L-k)}$ pixels arranged in a $d^{L-k} \times d^{L-k}$ sub-grid.

For $d=2$ (qubits), this creates a **quad-tree structure**:
- Level 0: Entire image ($N \times N$)
- Level 1: 4 quadrants ($N/2 \times N/2$ each)
- Level $k$: $4^k$ blocks ($N/2^k \times N/2^k$ each)
- Level $L$: Individual pixels

**Property (Sibling Relationship):**
Two pixels are *$k$-siblings* if their HCVs agree on the first $k$ position pairs but differ on pair $k+1$. They belong to the same level-$k$ parent block but different level-$(k+1)$ child blocks.

---

## 2. The Seam Problem ⚠️

### 2.1 Origin of Seam Artifacts

When denoising operations are applied within hierarchical blocks without cross-boundary awareness, discontinuities emerge at block boundaries.

**Definition (Seam Artifact):**
A seam artifact occurs when the denoising operation produces a discontinuity at the boundary between adjacent hierarchical blocks. Mathematically, for adjacent pixels $p_1$ and $p_2$ that are $(k+1)$-neighbors but $k$-siblings:

$$|I'(p_1) - I'(p_2)| \gg |I(p_1) - I(p_2)|$$

where $I$ is the original intensity and $I'$ is the denoised intensity.

### 2.2 Seam-Aware Constraint

**Theorem (Boundary Consistency):**
Let $\mathcal{F}$ be a denoising operator. $\mathcal{F}$ is seam-free if for any two adjacent pixels $p_1, p_2$ that are $k$-siblings for some $k$:

$$|\mathcal{F}[I](p_1) - \mathcal{F}[I](p_2)| \leq \alpha \cdot |I(p_1) - I(p_2)| + \beta \cdot \sigma_{\text{local}}$$

where:
- $\alpha \leq 1$ is a smoothing factor
- $\beta$ is a noise tolerance
- $\sigma_{\text{local}}$ is the local noise estimate

### 2.3 Multi-Scale Smoothing Weight

To achieve seam-free denoising, we propose a *hierarchical smoothing weight*:

$$w(p_1, p_2) = \begin{cases}
w_{\text{intra}}(k) & \text{if } p_1, p_2 \text{ share level-}k\text{ parent, same child} \\
w_{\text{cross}}(k) & \text{if } p_1, p_2 \text{ are } k\text{-siblings} \\
0 & \text{if } k < k_{\min}
\end{cases}$$

where $w_{\text{cross}}(k)$ ensures continuity:

$$w_{\text{cross}}(k) = w_{\text{intra}}(k) \cdot \gamma^{(L-k)}$$

with $\gamma \in (0, 1]$ being a cross-boundary coupling factor.

---

## 3. Adapted Denoising Algorithms

### 3.1 Hierarchical Non-Local Means (H-NLM)

Classical NL-Means:

$$\text{NLM}[I](p) = \frac{1}{Z(p)} \sum_{q \in \Omega(p)} w(p, q) \cdot I(q)$$

where $w(p, q) = \exp\left(-\frac{\|P(p) - P(q)\|^2}{h^2}\right)$ and $P(p)$ is the patch around $p$.

#### Hierarchical Patch Distance

$$d_H(p, q) = \sum_{k=0}^{L-1} \lambda^k \cdot \|P_k(p) - P_k(q)\|^2$$

where $P_k(p)$ is the mean intensity of the level-$k$ block containing $p$, and $\lambda > 1$ weights coarser levels more heavily.

#### H-NLM Weight with Seam Penalty

$$w_H(p, q) = \exp\left(-\frac{d_H(p, q)}{h^2}\right) \cdot \Phi(p, q)$$

where the seam-penalty function is:

$$\Phi(p, q) = \prod_{k: p,q \text{ are } k\text{-siblings}} (1 - \eta_k)$$

with $\eta_k \in [0, 1)$ being level-dependent seam penalties.

---

### 3.2 Hierarchical Block-Matching (H-BM3D)

#### Hierarchical Similarity

$$\text{Sim}_H(B_1, B_2) = \sum_{k=0}^{L_B-1} \mu^k \cdot \text{Sim}(B_1^{(k)}, B_2^{(k)})$$

where $B^{(k)}$ is the restriction of block $B$ to level-$k$ sub-blocks, and $\mu > 1$ emphasizes coarse-scale similarity.

#### Seam-Aware Aggregation

$$I'(p) = \frac{\sum_{B \ni p} w_B \cdot \tilde{I}_B(p) \cdot \Psi_B(p)}{\sum_{B \ni p} w_B \cdot \Psi_B(p)}$$

with boundary-aware weight:

$$\Psi_B(p) = \prod_{k} \min\left(1, \frac{\delta_k(p, \partial B_k)}{\tau_k}\right)$$

where $\delta_k(p, \partial B_k)$ is the distance from $p$ to the level-$k$ block boundary.

---

### 3.3 Hierarchical SRAD (H-SRAD)

SRAD evolves the image according to:

$$\frac{\partial I}{\partial t} = \text{div}(c(\nabla I) \nabla I)$$

#### Hierarchical Diffusion Coefficient

$$c_H(\mathbf{h}) = \prod_{k=0}^{L-1} c_k\left(|\nabla^{(k)} I|(\mathbf{h})\right)$$

where the scale-$k$ gradient is:

$$\nabla^{(k)} I(\mathbf{h}) = I(\mathbf{h}) - \langle I \rangle_{\text{siblings}(k, \mathbf{h})}$$

and $c_k(g) = \frac{1}{1 + (g/\kappa_k)^2}$ with scale-dependent threshold $\kappa_k$.

#### Seam-Aware Discretization

$$\Delta_{\text{seam}} I(p) = \sum_{q \in \mathcal{N}(p)} \omega_{pq} \cdot c_H(p, q) \cdot (I(q) - I(p))$$

where cross-boundary neighbors get boosted weight:

$$\omega_{pq} = 1 + \nu \cdot \mathbf{1}[\text{sibling level}(p, q) < L]$$

with $\nu > 0$ boosting cross-boundary diffusion.

---

## 4. Quantum Circuit Formulation

### 4.1 MHRQIB: Direct Intensity Access

In MHRQIB, intensity is encoded in $b$ qubits as:

$$|I(\mathbf{h})\rangle = |i_{b-1} i_{b-2} \ldots i_0\rangle \quad \text{where } I = \sum_{j=0}^{b-1} i_j \cdot 2^j$$

#### Local Mean Computation

$$\langle I_{\text{local}} \rangle = \frac{1}{|\mathcal{N}|} \sum_{\mathbf{h}' \in \mathcal{N}(\mathbf{h})} I(\mathbf{h}')$$

**Implementation:**
1. Apply Hadamard to last position qubit pair → creates superposition of 4 neighbors
2. Measure intensity qubits → collapses to one neighbor value (shot-based)
3. Repeat and average OR use phase estimation

#### Diffusion Update

$$I'(\mathbf{h}) = I(\mathbf{h}) + \epsilon \cdot c_H \cdot (I_{\text{local}} - I(\mathbf{h}))$$

Implemented as controlled $R_y(\theta)$ rotation where $\theta = f(\epsilon, c_H)$.

### 4.2 MHRQI: Continuous Intensity (Angle Encoding)

In MHRQI, intensity is encoded as:

$$|\psi(\mathbf{h})\rangle = \cos(\theta(\mathbf{h})/2)|0\rangle + \sin(\theta(\mathbf{h})/2)|1\rangle$$

where $\theta = 2\arcsin(\sqrt{I})$ and $P(|1\rangle) = I$.

#### Denoising via Angle Adjustment

$$\theta' = \theta + \epsilon \cdot c_H \cdot (\theta_{\text{local}} - \theta)$$

---

## 5. Seam-Free Implementation Guidelines

1. **Never process blocks independently**: Always include cross-boundary terms
2. **Use overlapping windows**: At each hierarchy level, use windows that span sibling boundaries
3. **Apply boundary boosting**: Increase diffusion strength at block boundaries via $\omega_{pq}$
4. **Multi-pass refinement**: Alternate between intra-block and inter-block smoothing
5. **Coarse-to-fine**: Start denoising at coarse levels (averaging siblings) before fine levels

---

## 6. Summary of Key Equations

| Concept | Equation |
|---------|----------|
| Hierarchical patch distance | $d_H(p, q) = \sum_{k} \lambda^k \|P_k(p) - P_k(q)\|^2$ |
| Seam penalty | $\Phi(p, q) = \prod_{k: \text{sibling}} (1 - \eta_k)$ |
| H-NLM weight | $w_H = \exp(-d_H/h^2) \cdot \Phi$ |
| Hierarchical diffusion coeff | $c_H = \prod_k c_k(\|\nabla^{(k)} I\|)$ |
| Cross-boundary boost | $\omega_{pq} = 1 + \nu \cdot \mathbf{1}[\text{sibling}]$ |

## 7. Robust Cycle Spinning (Engineering Approach)

### 7.1 Concept
Standard Cycle Spinning (Translation Invariant Denoising) averages the results of shifted denoising passes to remove Gibbs artifacts.
6426I_{\text{ensure}}(p) = \frac{1}{K} \sum_{i=1}^K T_{-\delta_i} [\mathcal{F}(T_{\delta_i}[I])](p)6426
This corresponds to the **Mean**, which reduces variance but can blur features and smear speckles (outliers) into the background.

### 7.2 Speckle Rejection via Median/Mode
To address "speckle" noise (isolated high-intensity pixels) and avoid blurring, we propose using a **Robust Estimator** for the aggregation step.
6426I_{\text{robust}}(p) = \text{Median} \left\{ T_{-\delta_i} [\mathcal{F}(T_{\delta_i}[I])](p) \right\}_{i=1}^K6426
Or ideally the **Mode** (most frequent value) for discrete data.

**Mathematical Justification:**
- **Speckle as Outlier**: A speckle pixel is often an outlier relative to its neighbors. In different grid shifts, the denoiser $\mathcal{F}$ may successfully suppress it in some shifts (where it falls in a favorable block context) but not others.
- **Mean vs Median**: The Mean will preserve the speckle energy, spreading it as a "blur". The Median will essentially "vote" on the pixel value. If the pixel is successfully cleaned in >50% of shifts, the Median will be clean.
- **Edge Preservation**: Median filters preserve edges better than mean filters.

This aligns with the intuition of "spinning pixels" and "bringing outliers to the mode".
