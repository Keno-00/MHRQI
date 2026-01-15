// MHRQI CORE JS

// --- MERMAID INIT ---
if (window.mermaid) {
    mermaid.initialize({
        startOnLoad: true,
        theme: 'base',
        themeVariables: {
            primaryColor: '#ffffff',
            primaryBorderColor: '#000000',
            primaryTextColor: '#000000',
            lineColor: '#000000',
            secondaryColor: '#ffffff',
            tertiaryColor: '#ffffff',
            edgeLabelBackground: '#ffffff'
        }
    });
}

// --- NAVIGATION ---
document.addEventListener('DOMContentLoaded', () => {
    const nav = document.querySelector('.main-nav');
    const header = document.getElementById('mainHeader');

    if (nav) {
        const links = nav.querySelectorAll('a');
        const currentPath = window.location.pathname.split('/').pop() || 'index.html';
        links.forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }

    // Shrinking Header Logic
    if (header) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 100) {
                header.classList.add('shrunk');
            } else {
                header.classList.remove('shrunk');
            }
        });
    }
});

// --- EXPLORER LOGIC ---
const MHRQI_Explorer = (function () {
    let refs = {};
    let state = {
        N: 8, D: 2, maxL: 3, zoomLevel: 0,
        collapseTimeout: null
    };

    function init(N, d) {
        // Find elements
        refs.grid = document.getElementById('pixelGrid');
        refs.gridOverlay = document.getElementById('gridOverlay');
        refs.isoContainer = document.getElementById('isoContainer');
        refs.levelViewport = document.getElementById('levelViewport');
        refs.levelIndicator = document.getElementById('levelIndicator');
        refs.mathContent = document.getElementById('mathContent');
        refs.defaultMsg = document.getElementById('defaultMsg');
        refs.errorMsg = document.getElementById('errorMsg');

        if (!refs.grid) return; // Not on this page

        const L = Math.floor(Math.log(N) / Math.log(d));
        if (Math.pow(d, L) !== N) {
            if (refs.errorMsg) refs.errorMsg.innerText = `N=${N} invalid for d=${d}`;
            return;
        }
        if (refs.errorMsg) refs.errorMsg.innerText = "";

        state.N = N; state.D = d; state.maxL = L; state.zoomLevel = 0;

        // Reset
        if (refs.mathContent) refs.mathContent.style.display = 'none';
        if (refs.defaultMsg) refs.defaultMsg.style.display = 'block';
        refs.grid.innerHTML = '';
        if (refs.gridOverlay) refs.grid.appendChild(refs.gridOverlay);

        // Grid
        refs.grid.style.gridTemplateColumns = `repeat(${N}, 1fr)`;
        for (let i = 0; i < N * N; i++) {
            const div = document.createElement('div');
            div.className = 'pixel';
            div.dataset.x = i % N;
            div.dataset.y = Math.floor(i / N);

            div.onmousedown = (e) => { e.preventDefault(); triggerExplode(); selectPixel(div.dataset.x, div.dataset.y); };
            div.onmouseup = triggerCollapse;
            div.onmouseleave = triggerCollapse;
            div.onclick = () => selectPixel(div.dataset.x, div.dataset.y);

            refs.grid.appendChild(div);
        }

        renderAllLevelsStack();
        updateActiveLayer(0);
        initScrollListener();

        // Hooks for architectural explorers (if they exist)
        updateArchExplorers([]);
    }

    function initScrollListener() {
        if (!refs.levelViewport) return;
        refs.levelViewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            triggerExplode();

            const delta = Math.sign(e.deltaY);
            if (delta > 0) {
                if (state.zoomLevel < state.maxL - 1) state.zoomLevel++;
            } else {
                if (state.zoomLevel > 0) state.zoomLevel--;
            }

            updateActiveLayer(state.zoomLevel);
            triggerCollapse();
        }, { passive: false });
    }

    function triggerExplode() {
        if (!refs.isoContainer) return;
        refs.isoContainer.classList.remove('mode-flat');
        refs.isoContainer.classList.add('mode-iso');
        if (state.collapseTimeout) clearTimeout(state.collapseTimeout);
    }

    function triggerCollapse() {
        if (!refs.isoContainer) return;
        if (state.collapseTimeout) clearTimeout(state.collapseTimeout);
        state.collapseTimeout = setTimeout(() => {
            refs.isoContainer.classList.remove('mode-iso');
            refs.isoContainer.classList.add('mode-flat');
        }, 300);
    }

    function renderAllLevelsStack() {
        if (!refs.isoContainer) return;
        refs.isoContainer.innerHTML = '';
        const fixedSize = 240;

        for (let k = 1; k <= state.maxL; k++) {
            const layer = document.createElement('div');
            layer.className = 'iso-layer';

            // CENTERING + Z-STACK
            const zPos = (state.maxL - k) * 60;
            layer.style.transform = `translate(-50%, -50%) translateZ(${zPos}px)`;

            layer.dataset.level = k - 1;
            layer.style.width = layer.style.height = `${fixedSize}px`;

            const label = document.createElement('div');
            label.className = 'layer-label';
            label.innerText = `L${k - 1}`;
            layer.appendChild(label);

            const nodesPerRow = Math.pow(state.D, k);
            layer.style.display = 'grid';
            layer.style.gridTemplateColumns = `repeat(${nodesPerRow}, 1fr)`;

            for (let j = 0; j < nodesPerRow * nodesPerRow; j++) {
                const node = document.createElement('div');
                node.className = 'node';
                node.style.width = node.style.height = '100%';
                node.dataset.level = k - 1;
                node.dataset.index = j;
                node.style.transformStyle = 'preserve-3d';
                node.onclick = (e) => { e.stopPropagation(); selectNode(k - 1, j); };
                layer.appendChild(node);
            }
            refs.isoContainer.appendChild(layer);
        }
    }

    function updateActiveLayer(level) {
        state.zoomLevel = level;
        if (refs.levelIndicator) refs.levelIndicator.innerText = `LEVEL ${level} (Zoom: ${level + 1}/${state.maxL})`;
        if (refs.isoContainer) {
            refs.isoContainer.querySelectorAll('.iso-layer').forEach(l => {
                if (parseInt(l.dataset.level) === level) l.classList.add('active-layer');
                else l.classList.remove('active-layer');
            });
        }
        updateGridOverlay(level + 1);
    }

    function updateGridOverlay(level) {
        if (!refs.gridOverlay) return;
        refs.gridOverlay.innerHTML = '';
        if (level === 0) return;
        const numBlocks = Math.pow(state.D, level);
        const step = 100 / numBlocks;
        for (let i = 1; i < numBlocks; i++) {
            let v = document.createElement('div'); v.className = 'grid-line'; v.style.left = `${i * step}%`; v.style.top = '0'; v.style.bottom = '0'; v.style.width = '1px';
            let h = document.createElement('div'); h.className = 'grid-line'; h.style.top = `${i * step}%`; h.style.left = '0'; h.style.right = '0'; h.style.height = '1px';
            refs.gridOverlay.appendChild(v); refs.gridOverlay.appendChild(h);
        }
    }

    function selectPixel(x, y) {
        x = parseInt(x); y = parseInt(y);
        clearHighlights();
        const p = refs.grid.querySelector(`.pixel[data-x="${x}"][data-y="${y}"]`);
        if (p) p.classList.add('active');
        const qv = calculateQVector(x, y);
        showMath(x, y, qv);
        highlightPathInStack(x, y);
        updateArchExplorers(qv);
    }

    function selectNode(level, index) {
        clearHighlights();
        if (state.zoomLevel !== level) {
            state.zoomLevel = level;
            updateActiveLayer(level);
            triggerCollapse(); // Collapse iso view to flat zoom
        }
        const layer = refs.isoContainer.querySelector(`.iso-layer[data-level="${level}"]`);
        const node = layer.querySelector(`.node[data-index="${index}"]`);
        if (node) node.classList.add('active');
        highlightGridRegion(level, index);

        const s_k = state.N / Math.pow(state.D, level + 1);
        const gridW = Math.pow(state.D, level + 1);
        const sx = (index % gridW) * s_k;
        const sy = Math.floor(index / gridW) * s_k;
        showMath(sx, sy, calculateQVector(sx, sy));
    }

    function highlightGridRegion(level, index) {
        const s_k = state.N / Math.pow(state.D, level + 1);
        const gridW = Math.pow(state.D, level + 1);
        const sx = (index % gridW) * s_k;
        const sy = Math.floor(index / gridW) * s_k;
        for (let y = sy; y < sy + s_k; y++) {
            for (let x = sx; x < sx + s_k; x++) {
                const p = refs.grid.querySelector(`.pixel[data-x="${x}"][data-y="${y}"]`);
                if (p) p.classList.add('active');
            }
        }
    }

    function highlightPathInStack(px, py) {
        if (!refs.isoContainer) return;
        for (let k = 1; k <= state.maxL; k++) {
            const sk = state.N / Math.pow(state.D, k);
            const gw = Math.pow(state.D, k);
            const idx = Math.floor(py / sk) * gw + Math.floor(px / sk);
            const layer = refs.isoContainer.querySelector(`.iso-layer[data-level="${k - 1}"]`);
            if (layer) {
                const node = layer.querySelector(`.node[data-index="${idx}"]`);
                if (node) {
                    node.classList.add('active');
                    if (k < state.maxL) injectCubeFaces(node);
                }
            }
        }
    }

    function injectCubeFaces(node) {
        if (node.querySelector('.cube-face')) return;
        ['face-front', 'face-back', 'face-left', 'face-right'].forEach(f => {
            const div = document.createElement('div'); div.className = `cube-face ${f}`;
            node.appendChild(div);
        });
    }

    function clearHighlights() {
        document.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.cube-face').forEach(el => el.remove());
    }

    function calculateQVector(x, y) {
        const v = [];
        for (let k = 1; k <= state.maxL; k++) {
            const sp = state.N / Math.pow(state.D, k - 1);
            const sk = state.N / Math.pow(state.D, k);
            const qx = Math.floor((x % sp) * state.D / sp);
            const qy = Math.floor((y % sp) * state.D / sp);
            v.push({ k: k - 1, qx, qy, s_prev: sp, k_orig: k });
        }
        return v;
    }

    function showMath(x, y, Q) {
        if (!refs.mathContent) return;
        refs.defaultMsg.style.display = 'none';
        refs.mathContent.style.display = 'block';
        const vecVal = Q.map(q => `(${q.qx},${q.qy})`).join(', ');
        const ketStr = Q.map(q => `|${q.qx}\\rangle |${q.qy}\\rangle`).join(' \\otimes ');

        let h = `<div class="ui-text" style="font-size:0.9rem; margin-bottom:10px;">OBJECTIVE: PIXEL LOCALIZATION (${x}, ${y})</div>`;
        h += `<p style="font-size:0.8rem; margin-bottom:15px;">Decomposing coordinates through $L=${state.maxL}$ levels of recursive subdivision.</p>`;

        Q.forEach(q => {
            h += `<div class="wireframe-box" style="padding:15px; font-size:0.85rem; border-color: #ddd;">
                <div class="ui-text" style="font-size:0.75rem; border-bottom:1px solid #eee; padding-bottom:5px; margin-bottom:10px;">LEVEL ${q.k} ANALYSIS</div>
                
                <div style="margin-bottom:10px;">
                    <strong>1. Formula:</strong>
                    $$ Q_k^c = \\lfloor (c \\bmod s_{k-1}) \\cdot \\frac{d}{s_{k-1}} \\rfloor $$
                </div>
                
                <div style="margin-bottom:10px;">
                    <strong>2. Equation:</strong>
                    $$ Q_{${q.k}}^x = \\lfloor (${x} \\bmod ${q.s_prev}) \\cdot \\frac{${state.D}}{${q.s_prev}} \\rfloor $$
                    $$ Q_{${q.k}}^y = \\lfloor (${y} \\bmod ${q.s_prev}) \\cdot \\frac{${state.D}}{${q.s_prev}} \\rfloor $$
                </div>
                
                <div>
                    <strong>3. Solution:</strong>
                    $Q_{${q.k}} = (\\boxed{${q.qx}}, \\boxed{${q.qy}})$
                </div>
            </div>`;
        });

        h += `<div class="ui-text" style="font-size:0.85rem; margin-top:20px;">FINAL REPRESENTATION</div>`;
        h += `<div class="equation-box">
            <strong>HCV:</strong> $$ H(${x},${y}) = [${vecVal}] $$
            <strong>STATE:</strong> $$ |\\Psi_{pix}\\rangle = ${ketStr} $$
        </div>`;

        refs.mathContent.innerHTML = h;
        if (window.MathJax && window.MathJax.typeset) MathJax.typeset();
    }

    function updateArchExplorers(qv) {
        // These can be extended per page
        if (typeof updateCircuitExplorer === 'function') updateCircuitExplorer(qv);
        if (typeof updateHomogeneityExplorer === 'function') updateHomogeneityExplorer(qv);
        if (typeof updateMetricExplorer === 'function') updateMetricExplorer(qv);
    }

    // Exposed methods
    return { init, selectPixel, selectNode };
})();

// --- HOMOGENEITY EXPLORER (Denoising Page) ---
function updateHomogeneityExplorer(qv) {
    const el = document.getElementById('homogeneityExplorer');
    if (!el) return;

    if (!qv || qv.length === 0) {
        el.innerHTML = '<p style="font-family:var(--font-ui); font-size:0.8rem;">[SELECT PIXEL TO ANALYZE CORRELATION]</p>';
        return;
    }

    // Simulate Homogeneity Logic
    // In real code: is_block_homogeneous depends on gradient/noise
    // Here we simulate it based on coordinate "evenness" as a visual proxy
    const lastQ = qv[qv.length - 1];
    const isEdge = (lastQ.qx + lastQ.qy) % 2 !== 0; // Visual proxy for "edge"

    let h = `<div class="ui-text" style="font-size:0.8rem; margin-bottom:10px;">NEIGHBOR CORRELATION ANALYSIS</div>`;

    if (isEdge) {
        h += `
        <div style="background:#fff2f2; border:1px solid #ffcccc; padding:10px; border-radius:4px;">
            <strong style="color:#cc0000;">STRUCTURE DETECTED (LOW HOMOGENEITY)</strong>
            <p style="font-size:0.7rem; margin-top:5px;">
                Neighbor correlation is <strong>suppressed</strong> to preserve structural fidelity.
                Unitary diffusion is restricted to prevent blurring of sharp boundaries.
            </p>
            <div style="display:flex; justify-content:center; gap:5px; margin-top:10px;">
                <div style="width:20px; height:20px; background:#cc0000; opacity:0.8;"></div>
                <div style="width:20px; height:20px; background:#eee; border:1px dashed #ccc;"></div>
            </div>
        </div>`;
    } else {
        h += `
        <div style="background:#f2fff2; border:1px solid #ccffcc; padding:10px; border-radius:4px;">
            <strong style="color:#008800;">NOISE/UNIFORM REGION (HIGH HOMOGENEITY)</strong>
            <p style="font-size:0.7rem; margin-top:5px;">
                Neighbor correlation is <strong>active</strong>. Sibling pixels within the hierarchical block
                are averaged to suppress speckle and Gaussian noise.
            </p>
            <div style="display:flex; justify-content:center; gap:5px; margin-top:10px;">
                <div style="width:20px; height:20px; background:#008800; opacity:0.5;"></div>
                <div style="width:20px; height:20px; background:#008800; opacity:0.5;"></div>
                <div style="width:20px; height:20px; background:#008800; opacity:0.5;"></div>
            </div>
        </div>`;
    }

    h += `<p style="font-size:0.6rem; color:#888; margin-top:10px;">[BASED ON HIERARCHICAL COUPLING AT L${qv.length - 1}]</p>`;

    el.innerHTML = h;
}

// --- CIRCUIT EXPLORER (Encoder Page) ---
function updateCircuitExplorer(qv) {
    const el = document.getElementById('circuitExplorer');
    if (!el) return;

    if (!qv || qv.length === 0) {
        el.innerHTML = '<p style="font-family:var(--font-ui); font-size:0.8rem;">[SELECT PIXEL TO SEE GATES]</p>';
        return;
    }

    let h = `<div class="ui-text" style="font-size:0.8rem; margin-bottom:10px;">BASIS ENCODING SEQUENCE</div>`;
    h += `<div style="font-family:monospace; font-size:0.7rem; text-align:left; background:#eee; padding:10px; overflow-x:auto;">`;

    qv.forEach(q => {
        h += `QC.MCX([q_y_${q.k}, q_x_${q.k}], and_ancilla)<br>`;
    });

    h += `<span style="color:#000088;">// Intensity Upload (Basis)</span><br>`;
    h += `FOR bit IN 0..7:<br>&nbsp;&nbsp;IF intensity_bits[bit] == '1':<br>&nbsp;&nbsp;&nbsp;&nbsp;QC.CX(and_ancilla, intensity[bit])<br>`;

    qv.slice().reverse().forEach(q => {
        h += `QC.MCX([q_y_${q.k}, q_x_${q.k}], and_ancilla) <span style="color:#888;">(Uncompute)</span><br>`;
    });

    h += `</div>`;
    el.innerHTML = h;
}

// --- METRIC EXPLORER (Benchmark Page) ---
function updateMetricExplorer(qv) {
    const el = document.getElementById('metricExplorer');
    if (!el) return;

    let h = `<div class="ui-text" style="font-size:0.8rem; margin-bottom:15px; border-bottom: 2px solid #000; padding-bottom:5px;">LATEST BENCHMARK RESULTS [20260110_0646]</div>`;

    // helper for table rows
    const row = (name, v1, v2, rank, highlight = false) => `
        <tr style="${highlight ? 'background:#f0f7f0; font-weight:bold;' : 'color:#666;'}">
            <td style="padding:4px 8px; border:1px solid #eee;">${name}</td>
            <td style="padding:4px 8px; border:1px solid #eee; text-align:center;">${v1}</td>
            <td style="padding:4px 8px; border:1px solid #eee; text-align:center;">${v2}</td>
            <td style="padding:4px 8px; border:1px solid #eee; text-align:center;">${rank}</td>
        </tr>`;

    // Category 1: Full Reference
    h += `<div class="ui-text" style="font-size:0.7rem; margin: 15px 0 5px;">01 // FULL REFERENCE (Similarity)</div>
    <table style="width:100%; font-size:0.7rem; border-collapse: collapse; font-family:var(--font-ui); margin-bottom:20px;">
        <thead><tr style="background:#000; color:#fff; text-align:center;">
            <th style="padding:5px; text-align:left;">METHOD</th><th>FSIM ↑</th><th>SSIM ↑</th><th>RANK</th>
        </tr></thead>
        <tbody>
            ${row('BM3D', '1.000', '1.000', '#1')}
            ${row('NL-MEANS', '0.982', '0.999', '#2')}
            ${row('MHRQI', '0.895', '0.999', '#3', true)}
            ${row('SRAD', '0.887', '0.999', '#4')}
        </tbody>
    </table>`;

    // Category 2: No-Reference (Quality)
    h += `<div class="ui-text" style="font-size:0.7rem; margin: 15px 0 5px;">02 // NO-REFERENCE (Naturalness)</div>
    <table style="width:100%; font-size:0.7rem; border-collapse: collapse; font-family:var(--font-ui); margin-bottom:20px;">
        <thead><tr style="background:#000; color:#fff; text-align:center;">
            <th style="padding:5px; text-align:left;">METHOD</th><th>PIQE ↓</th><th>BRISQUE ↓</th><th>RANK</th>
        </tr></thead>
        <tbody>
            ${row('MHRQI', '39.61**', '11.71*', '#1', true)}
            ${row('NL-MEANS', '64.02', '21.85', '#2')}
            ${row('SRAD', '72.36', '42.46', '#3')}
            ${row('BM3D', '77.09', '34.23', '#4')}
        </tbody>
    </table>`;

    // Category 3: Speckle Metrics
    h += `<div class="ui-text" style="font-size:0.7rem; margin: 15px 0 5px;">03 // SPECKLE METRICS (Suppression)</div>
    <table style="width:100%; font-size:0.7rem; border-collapse: collapse; font-family:var(--font-ui); margin-bottom:10px;">
        <thead><tr style="background:#000; color:#fff; text-align:center;">
            <th style="padding:5px; text-align:left;">METHOD</th><th>SSI ↓</th><th>SMPI ↓</th><th>RANK</th>
        </tr></thead>
        <tbody>
            ${row('SRAD', '0.620', '0.847', '#1')}
            ${row('MHRQI', '0.703\u2020', '0.785', '#2', true)}
            ${row('BM3D', '0.785', '0.964', '#3')}
            ${row('NL-MEANS', '0.788', '0.978', '#4')}
        </tbody>
    </table>
    <p style="font-size:0.6rem; color:#888; margin-top:10px; font-style:italic;">
        ** p < 0.01 (Significantly better than ALL baselines).<br>
        * p < 0.05 (Significantly better than BM3D/SRAD).<br>
        \u2020 p < 0.01 (Significantly better than SRAD / Comparable to BM3D).
    </p>`;

    el.innerHTML = h;
}

// Initialize if elements are present
document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('pixelGrid');
    if (grid) {
        // Decide N based on page
        const isPreprocessing = window.location.pathname.includes('preprocessing.html');
        // If not preprocessing, we use a smaller 4x4 probe grid
        const N = isPreprocessing ? 8 : 4;
        const D = 2;
        MHRQI_Explorer.init(N, D);

        const regen = document.getElementById('regenBtn');
        if (regen) {
            regen.onclick = () => {
                const n = parseInt(document.getElementById('inputN').value);
                const d = parseInt(document.getElementById('inputD').value);
                MHRQI_Explorer.init(n, d);
            };
        }
    } else {
        // Static components (Benchmark page)
        updateMetricExplorer();
    }
});
