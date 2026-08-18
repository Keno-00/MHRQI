# Utils Module Reference

Documentation for `utils.py` - utility functions for encoding and reconstruction.

---

## Coordinate Encoding

### angle_map(img, bit_depth=8)

Convert grayscale image to RY rotation angles (for angle-based encoding).

---

### get_Lmax(N, d)

Calculate hierarchy depth for image size N with dimension d.

**Example:** `get_Lmax(256, 2)` → 8

---

### get_subdiv_size(k, N, d)

Get subdivision size at hierarchy level k.

---

### compute_register(r, c, d, sk_prev)

Compute position register values (qy, qx) for pixel (r, c).

---

### compose_rc(hcv, d=2)

Reconstruct pixel coordinates (r, c) from Hierarchical Coordinate Vector.

**Parameters:**
- `hcv`: List of position qubit values [qy_0, qx_0, qy_1, qx_1, ...]
- `d`: Dimension (default: 2)

---

## Image Reconstruction

### mhrqi_bins_to_image(bins, hierarchy_matrix, d, image_shape, bias_stats=None, original_img=None)

Reconstruct image from measurement bins with confidence-weighted smoothing.

**Algorithm:**
1. Build baseline image from bin intensity averages
2. Compute confidence map from bias statistics (hit/miss ratio)
3. For low-confidence pixels, blend with median of high-confidence neighbors
4. Return smoothed image

**Parameters:**
- `bins`: Measurement bin dictionary from circuit
- `bias_stats`: (optional) Bias qubit statistics for confidence weighting
- `original_img`: (optional) Use as source instead of baseline reconstruction
