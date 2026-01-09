# Seam Theory for Hierarchical Quantum Image Representations

**Author**: Keno S. Jose  
**Project**: MHRQI - Multiscale Hierarchical Representation of Quantum Images

---

## Abstract

This document establishes the theoretical foundations for understanding and preventing **seam artifacts** in hierarchical quantum image representations. Seams arise when processing operates independently within hierarchical blocks without accounting for cross-boundary relationships. We derive necessary conditions for seam-free processing and prove how MHRQIB's hierarchical structure naturally enables seam-aware operations.

---

## 1. Definitions

### 1.1 Hierarchical Image Partition

An $N \times N$ image where $N = d^L$ is hierarchically partitioned into blocks at $L+1$ levels.

**Definition 1 (Level-$k$ Block):**
A level-$k$ block $B_k(\mathbf{p})$ is the set of all pixels sharing the first $k$ hierarchy digits with prefix $\mathbf{p} = (p_0, p_1, \ldots, p_{k-1})$:

$$B_k(\mathbf{p}) = \{ (r,c) : \mathbf{h}(r,c)[0:k] = \mathbf{p} \}$$

- Level 0: Entire image ($|B_0| = N^2$ pixels)
- Level $k$: $d^{2k}$ blocks, each containing $d^{2(L-k)}$ pixels
- Level $L$: Individual pixels ($|B_L| = 1$)

### 1.2 Adjacency Relations

**Definition 2 (Intra-Block Neighbors):**
Pixels $p_1, p_2$ are *intra-block neighbors at level $k$* if:
1. They belong to the same level-$k$ block: $\mathbf{h}(p_1)[0:k] = \mathbf{h}(p_2)[0:k]$
2. They are spatially adjacent in the image grid

**Definition 3 (k-Siblings):**
Pixels $p_1, p_2$ are *$k$-siblings* if:
1. They share the same level-$k$ parent: $\mathbf{h}(p_1)[0:k] = \mathbf{h}(p_2)[0:k]$
2. They belong to different level-$(k+1)$ blocks: $\mathbf{h}(p_1)[k] \neq \mathbf{h}(p_2)[k]$

**Definition 4 (Seam Boundary):**
The *level-$k$ seam boundary* $\partial B_k$ is the set of pixel pairs $(p_1, p_2)$ that are:
- Spatially adjacent in the image grid
- $k$-siblings (crossing a level-$k$ block boundary)

---

## 2. The Seam Problem

### 2.1 Seam Artifact Definition

**Definition 5 (Seam Artifact):**
Let $\mathcal{F}$ be an image processing operator. A *seam artifact* of magnitude $\delta$ occurs at boundary $(p_1, p_2) \in \partial B_k$ when:

$$|\mathcal{F}[I](p_1) - \mathcal{F}[I](p_2)| - |I(p_1) - I(p_2)| > \delta$$

The operator has **amplified** the discontinuity across the block boundary.

### 2.2 Origin of Seams in Hierarchical Processing

**Theorem 1 (Block-Independent Processing Creates Seams):**
Let $\mathcal{F}$ be an operator that processes each level-$k$ block independently:

$$\mathcal{F}[I](p) = \mathcal{F}_{B_k(p)}[I|_{B_k(p)}](p)$$

where $\mathcal{F}_{B}$ operates only on pixels within block $B$. Then for any non-trivial smoothing operator $\mathcal{F}$, seam artifacts will occur at $\partial B_k$ unless the original image already has constant intensity across each block boundary.

**Proof (Sketch):**
Consider adjacent pixels $p_1 \in B_1$ and $p_2 \in B_2$ where $B_1, B_2$ are sibling blocks. Let $\mu_1 = \text{mean}(I|_{B_1})$ and $\mu_2 = \text{mean}(I|_{B_2})$.

A smoothing operator pulls pixel values toward the local mean:
- $\mathcal{F}[I](p_1) \approx \alpha I(p_1) + (1-\alpha)\mu_1$
- $\mathcal{F}[I](p_2) \approx \alpha I(p_2) + (1-\alpha)\mu_2$

The processed discontinuity is:
$$|\mathcal{F}[I](p_1) - \mathcal{F}[I](p_2)| \approx |\alpha(I(p_1) - I(p_2)) + (1-\alpha)(\mu_1 - \mu_2)|$$

If $\mu_1 \neq \mu_2$, the boundary term $(1-\alpha)(\mu_1 - \mu_2)$ creates or amplifies discontinuity. $\square$

### 2.3 Seam Locations in MHRQIB

For $d = 2$ (qubit-based), seam boundaries occur at:

| Level | Seam Pattern | Spacing |
|-------|--------------|---------|
| 0 | None (global) | - |
| 1 | Image midlines | $N/2$ |
| 2 | Quadrant midlines | $N/4$ |
| $k$ | Block $(N/2^k)$ boundaries | $N/2^k$ |

---

## 3. Seam-Free Processing Conditions

### 3.1 Boundary Consistency Constraint

**Definition 6 (Seam-Free Operator):**
An operator $\mathcal{F}$ is *seam-free* if for all $(p_1, p_2) \in \partial B_k$:

$$|\mathcal{F}[I](p_1) - \mathcal{F}[I](p_2)| \leq \gamma_k \cdot |I(p_1) - I(p_2)| + \epsilon$$

where:
- $\gamma_k \leq 1$ is the smoothing factor at level $k$
- $\epsilon$ is a noise floor tolerance

### 3.2 Cross-Boundary Coupling Requirement

**Theorem 2 (Cross-Boundary Coupling):**
To achieve seam-free processing, the operator must couple information across sibling blocks. Specifically, for each $(p_1, p_2) \in \partial B_k$, the output $\mathcal{F}[I](p_1)$ must depend on pixels from $B_k(p_2)$ and vice versa.

**Corollary:**
Any operator with receptive field smaller than the block size at any level will create seams at that level.

### 3.3 Multi-Scale Consistency

**Definition 7 (Hierarchically Consistent Smoothing):**
A smoothing operator is *hierarchically consistent* if the smoothing strength $\alpha_k$ at each level satisfies:

$$\frac{\alpha_{k}}{\alpha_{k+1}} \geq \frac{d^2 - 1}{d^2}$$

This ensures that coarse-level smoothing is sufficient to blend the seams created by fine-level smoothing.

---

## 4. MHRQIB's Natural Seam-Aware Structure

### 4.1 Hierarchical Averaging via Hadamard

The key insight is that MHRQIB's encoding naturally enables multi-scale averaging through the **partial Hadamard transform**.

**Proposition 1 (Block Mean Computation):**
Applying $H$ to position qubits at level $k$ creates a superposition that encodes the mean intensity of the level-$k$ block:

$$H_k|\mathbf{h}\rangle = \frac{1}{\sqrt{d^2}} \sum_{\mathbf{h}' \in \text{children}(\mathbf{h}, k)} |\mathbf{h}'\rangle$$

Measuring the intensity register after this operation samples from the uniform mixture of the $d^2$ child block values—effectively computing a block average.

### 4.2 Coarse-to-Fine Hierarchy

**Proposition 2 (Level-$k$ Mean):**
The mean intensity of a level-$k$ block is:

$$\mu_k(\mathbf{p}) = \frac{1}{|B_k(\mathbf{p})|} \sum_{(r,c) \in B_k(\mathbf{p})} I(r,c) = \frac{1}{d^{2(L-k)}} \sum_{\text{descendants}} I$$

This can be computed by applying $H^{\otimes 2(L-k)}$ to all position qubits from level $k$ to $L-1$.

### 4.3 Cross-Boundary Information Flow

MHRQIB naturally couples sibling blocks because:
1. **Coarse qubits encode block membership**: Qubits $0, 1, \ldots, k-1$ determine which level-$k$ block
2. **Superposition mixes siblings**: $H$ on qubit $k$ mixes adjacent blocks
3. **Intensity is shared**: The same intensity register contributes to all block states

---

## 5. Seam-Aware Operations

### 5.1 Sibling Coupling Operator

**Definition 8 (Sibling Coupling Gate):**
The sibling coupling at level $k$ is:

$$S_k = H_{q_{y,k}} \otimes H_{q_{x,k}}$$

applying Hadamard to both position qubits at level $k$. This creates equal superposition over the 4 sibling quadrants (for $d=2$).

### 5.2 Controlled Cross-Boundary Mixing

To selectively smooth across boundaries while preserving edges within blocks:

$$C\text{-}S_k(\theta) = \exp(-i\theta \cdot X_{q_{y,k}}) \otimes \exp(-i\theta \cdot X_{q_{x,k}})$$

where $\theta$ controls the mixing strength. At $\theta = 0$, no mixing occurs. At $\theta = \pi/2$, full sibling mixing.

### 5.3 Hierarchical Weight Schedule

For seam-free denoising, apply mixing at each level with weights:

$$w_k = \begin{cases}
w_{\text{seam}} \cdot (k+1) / L & \text{(linear for boundaries)} \\
w_{\text{smooth}} \cdot ((k+1)/L)^2 & \text{(quadratic for noise)}
\end{cases}$$

where $w_{\text{seam}} \ll w_{\text{smooth}}$ ensures gentle boundary blending without over-smoothing structure.

---

## 6. Mathematical Proofs

### 6.1 Proof: Hadamard Creates Block Superposition

**Lemma 1:**
For a pixel at HCV $\mathbf{h}$, applying $H$ to position qubit $q_{y,k}$ yields:

$$H|\mathbf{h}\rangle = \frac{1}{\sqrt{2}}(|\mathbf{h}_{q_{y,k}=0}\rangle + |\mathbf{h}_{q_{y,k}=1}\rangle)$$

This mixes the current pixel with its vertical sibling at level $k$.

**Proof:**
$H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ and $H|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$.

For $\mathbf{h}$ with $q_{y,k} = b \in \{0,1\}$:
$$H_k|\mathbf{h}\rangle = |\mathbf{h}[0:k], H|b\rangle, \mathbf{h}[k+1:]\rangle = \frac{1}{\sqrt{2}}(|\mathbf{h}_{0}\rangle + (-1)^b|\mathbf{h}_{1}\rangle)$$

The phase $(-1)^b$ cancels in probability: $|H|\mathbf{h}\rangle|^2 = \frac{1}{2} + \frac{1}{2} = 1$. $\square$

### 6.2 Proof: RX Creates Partial Mixing

**Lemma 2:**
$R_X(\theta)$ on position qubit $q_k$ mixes siblings with strength $\sin^2(\theta/2)$:

$$P(\text{sibling swap}) = \sin^2(\theta/2)$$

**Proof:**
$R_X(\theta) = \cos(\theta/2)I - i\sin(\theta/2)X$

For $|0\rangle$:
$$R_X(\theta)|0\rangle = \cos(\theta/2)|0\rangle - i\sin(\theta/2)|1\rangle$$

Probability of measuring $|1\rangle$ (i.e., transitioning to sibling):
$$P(1) = |\langle 1|R_X(\theta)|0\rangle|^2 = \sin^2(\theta/2) \quad \square$$

---

## 7. Summary of Key Results

| Concept | Mathematical Statement |
|---------|----------------------|
| Seam artifact | $\|\mathcal{F}[I](p_1) - \mathcal{F}[I](p_2)\| \gg \|I(p_1) - I(p_2)\|$ for $(p_1, p_2) \in \partial B_k$ |
| k-siblings | $\mathbf{h}(p_1)[0:k] = \mathbf{h}(p_2)[0:k]$ and $\mathbf{h}(p_1)[k] \neq \mathbf{h}(p_2)[k]$ |
| Seam-free condition | Operator must couple information across sibling blocks |
| Block averaging | $H^{\otimes 2}$ on level-$k$ qubits → mean of 4 children |
| Sibling mixing | $R_X(\theta)$ swaps siblings with probability $\sin^2(\theta/2)$ |
| Hierarchical consistency | $\alpha_k / \alpha_{k+1} \geq (d^2-1)/d^2$ |

---

## 8. References

1. Hierarchical Denoising Math (internal doc)
2. NEQR Quantum Image Representation
3. Multi-Resolution Analysis in Signal Processing
