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
        collapseTimeout: null,
        scrollInitialized: false
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
        if (!refs.levelViewport || state.scrollInitialized) return;
        state.scrollInitialized = true;

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
                }
            }
        }
    }



    function clearHighlights() {
        document.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
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
                
                <div style="margin-bottom:10px;">
                    <strong>3. Solution:</strong>
                    $$ \\boxed{Q_{${q.k}}^x = ${q.qx}} $$
                    $$ \\boxed{Q_{${q.k}}^y = ${q.qy}} $$
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

// --- CONSISTENCY EXPLORER ---
async function updateHomogeneityExplorer(qv) {
    const el = document.getElementById('homogeneityExplorer');
    if (!el) return;

    if (!qv || qv.length === 0) {
        el.innerHTML = '<p style="font-family:var(--font-ui); font-size:0.8rem; text-align:center;">[SELECT PIXEL TO ANALYZE CONSISTENCY]</p>';
        return;
    }

    el.innerHTML = '<p style="font-family:var(--font-ui); font-size:0.8rem; text-align:center;">Analyzing hierarchical consistency...</p>';

    try {
        const lastQ = qv[qv.length - 1];
        const response = await fetch(`/api/circuit/denoiser?n=16&d=2`); // Demo scale
        const data = await response.json();

        let h = `<div class="ui-text" style="font-size:0.8rem; margin-bottom:10px;">HIERARCHICAL CONSISTENCY ANALYSIS</div>`;

        // Explain the logic based on the current codebase
        h += `
        <div style="background:#f9f9f9; border:1px solid #eee; padding:15px; border-radius:4px;">
            <p style="font-size:0.8rem; margin-bottom:10px;">
                Checking consistency for pixel at <strong>L${qv.length - 1}</strong>.
            </p>
            <div class="circuit-render" style="margin-bottom:15px; background:#fff; border:1px solid #ddd; padding:10px; min-height:100px;">
                ${renderCircuitSVG(data)}
            </div>
            <p style="font-size:0.75rem; color:#444;">
                <strong>Logic:</strong> The MSB of the intensity register is compared with the parent average using a controlled-RY gate. 
                A successful match marks the <code>bias</code> qubit, ensuring structural preservation.
            </p>
        </div>`;

        el.innerHTML = h;
    } catch (err) {
        el.innerHTML = `<p style="color:red; font-size:0.7rem;">Error fetching consistency data: ${err.message}</p>`;
    }
}

// --- CIRCUIT EXPLORER ---
async function updateCircuitExplorer(qv) {
    const el = document.getElementById('circuitExplorer');
    const expl = document.getElementById('circuitExplanation');
    const coords = document.getElementById('pixelCoords');
    if (!el) return;

    if (!qv || qv.length === 0) {
        el.innerHTML = '<p style="font-family:var(--font-ui); font-size:0.8rem; text-align:center;">[SELECT PIXEL TO RENDER CIRCUIT]</p>';
        if (expl) expl.style.display = 'none';
        return;
    }

    const lastQ = qv[qv.length - 1];
    if (coords) coords.innerText = `(${lastQ.qx}, ${lastQ.qy}) @ L${qv.length - 1}`;

    el.innerHTML = '<p style="font-family:var(--font-ui); font-size:0.8rem; text-align:center;">Generating qudit gates...</p>';

    try {
        const response = await fetch(`/api/circuit/encoder?x=${lastQ.qx}&y=${lastQ.qy}&n=16`);
        const data = await response.json();

        el.innerHTML = renderCircuitSVG(data);
        if (expl) {
            expl.style.display = 'block';
            expl.innerHTML = `<strong>Basis Encoder Pipeline:</strong> Multi-controlled X gates address qudit paths. Qubits <code>qy${qv.length - 1}</code> and <code>qx${qv.length - 1}</code> act as controls for the ancilla-driven intensity flip.`;
        }
    } catch (err) {
        el.innerHTML = `<p style="color:red; font-size:0.7rem;">Error fetching circuit data: ${err.message}</p>`;
    }
}

function renderCircuitSVG(data) {
    if (data.error) return `<p style="color:orange;">${data.error}</p>`;

    const numQubits = data.qubits.length;
    const wireSpacing = 30;
    const gateStart = 80;
    const gateWidth = 35;
    const gateSpacing = 10;

    // Group gates into time slices (parallel gates share same column)
    const timeSlices = [];
    const qubitLastSlice = {};  // Track last slice each qubit was used

    data.gates.forEach((g) => {
        // Find earliest slot where none of this gate's qubits are already used
        let slotIdx = 0;
        for (const qi of g.qubits) {
            if (qubitLastSlice[qi] !== undefined) {
                slotIdx = Math.max(slotIdx, qubitLastSlice[qi] + 1);
            }
        }

        // Create slot if needed
        while (timeSlices.length <= slotIdx) {
            timeSlices.push([]);
        }

        // Add gate to this slot
        timeSlices[slotIdx].push(g);

        // Mark qubits as used in this slot
        for (const qi of g.qubits) {
            qubitLastSlice[qi] = slotIdx;
        }
    });

    const width = gateStart + timeSlices.length * (gateWidth + gateSpacing) + 40;
    const height = numQubits * wireSpacing + 30;

    let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" 
               xmlns="http://www.w3.org/2000/svg" style="display:block; margin:0 auto; background:#fff;">`;

    // Draw horizontal wires
    data.qubits.forEach((q, i) => {
        const y = i * wireSpacing + 20;
        svg += `<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="#ccc" stroke-width="1" />`;
        svg += `<text x="5" y="${y + 4}" font-family="monospace" font-size="10" fill="#333">${q.register || `q${i}`}</text>`;
    });

    // Draw gates by time slice
    timeSlices.forEach((slice, slotIdx) => {
        const x = gateStart + slotIdx * (gateWidth + gateSpacing);

        slice.forEach((g) => {
            const yPositions = g.qubits.map(qi => qi * wireSpacing + 20);
            const yMin = Math.min(...yPositions);
            const yMax = Math.max(...yPositions);
            const xCenter = x + gateWidth / 2;

            if (g.name === 'mcx' || g.name === 'cx' || g.name === 'ccx') {
                const targetIdx = g.qubits[g.qubits.length - 1];
                const targetY = targetIdx * wireSpacing + 20;

                if (g.qubits.length > 1) {
                    svg += `<line x1="${xCenter}" y1="${yMin}" x2="${xCenter}" y2="${yMax}" stroke="#333" stroke-width="2" />`;
                }
                for (let j = 0; j < g.qubits.length - 1; j++) {
                    const ctrlY = g.qubits[j] * wireSpacing + 20;
                    svg += `<circle cx="${xCenter}" cy="${ctrlY}" r="5" fill="#333" />`;
                }
                svg += `<circle cx="${xCenter}" cy="${targetY}" r="10" stroke="#333" fill="white" stroke-width="2" />`;
                svg += `<line x1="${xCenter - 10}" y1="${targetY}" x2="${xCenter + 10}" y2="${targetY}" stroke="#333" stroke-width="2" />`;
                svg += `<line x1="${xCenter}" y1="${targetY - 10}" x2="${xCenter}" y2="${targetY + 10}" stroke="#333" stroke-width="2" />`;
            }
            else if (g.name === 'h') {
                const targetY = g.qubits[0] * wireSpacing + 20;
                svg += `<rect x="${x}" y="${targetY - 12}" width="${gateWidth}" height="24" fill="#e8f4ff" stroke="#333" stroke-width="1.5" rx="2"/>`;
                svg += `<text x="${xCenter}" y="${targetY + 4}" font-family="monospace" font-size="12" font-weight="bold" text-anchor="middle" fill="#333">H</text>`;
            }
            else if (g.name === 'x') {
                const targetY = g.qubits[0] * wireSpacing + 20;
                svg += `<circle cx="${xCenter}" cy="${targetY}" r="10" stroke="#333" fill="white" stroke-width="2" />`;
                svg += `<line x1="${xCenter - 10}" y1="${targetY}" x2="${xCenter + 10}" y2="${targetY}" stroke="#333" stroke-width="2" />`;
                svg += `<line x1="${xCenter}" y1="${targetY - 10}" x2="${xCenter}" y2="${targetY + 10}" stroke="#333" stroke-width="2" />`;
            }
            else if (g.name === 'cry' || g.name === 'ry') {
                const targetIdx = g.qubits[g.qubits.length - 1];
                const targetY = targetIdx * wireSpacing + 20;
                if (g.qubits.length > 1) {
                    const ctrlY = g.qubits[0] * wireSpacing + 20;
                    svg += `<line x1="${xCenter}" y1="${yMin}" x2="${xCenter}" y2="${yMax}" stroke="#333" stroke-width="2" />`;
                    svg += `<circle cx="${xCenter}" cy="${ctrlY}" r="5" fill="#333" />`;
                }
                svg += `<rect x="${x}" y="${targetY - 12}" width="${gateWidth}" height="24" fill="#fff0e8" stroke="#333" stroke-width="1.5" rx="2"/>`;
                svg += `<text x="${xCenter}" y="${targetY + 4}" font-family="monospace" font-size="10" text-anchor="middle" fill="#333">${g.name.toUpperCase()}</text>`;
            }
            else {
                const targetY = g.qubits[0] * wireSpacing + 20;
                svg += `<rect x="${x}" y="${targetY - 12}" width="${gateWidth}" height="24" fill="#f5f5f5" stroke="#333" stroke-width="1.5" rx="2"/>`;
                svg += `<text x="${xCenter}" y="${targetY + 4}" font-family="monospace" font-size="10" text-anchor="middle" fill="#333">${g.name.toUpperCase()}</text>`;
            }
        });
    });

    svg += `</svg>`;
    // The wrapper has fixed max-width via CSS in HTML - SVG scrolls inside
    return `<div style="overflow-x:auto; overflow-y:hidden; white-space:nowrap;">${svg}</div>`;
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

// --- SUPERPOSITION ANIMATION (Encoder Page) ---
const SuperpositionAnimation = (function () {
    let state = {
        isPlaying: false,
        phase: 'idle', // 'idle', 'classical', 'quantum'
        animationId: null,
        timeoutIds: []
    };

    const GRID_SIZE = 8;
    const LEVELS = 3; // log2(8) = 3
    const COLORS = {
        '00': '#ef5350', // red
        '01': '#66bb6a', // green
        '10': '#42a5f5', // blue
        '11': '#ab47bc'  // purple
    };

    function init() {
        const viewport = document.getElementById('superpositionViewport');
        if (!viewport) return;

        resetAnimation();
    }

    function resetAnimation() {
        // Clear any running animations
        state.timeoutIds.forEach(id => clearTimeout(id));
        state.timeoutIds = [];
        state.isPlaying = false;
        state.phase = 'idle';

        // Reset UI
        const phaseLabel = document.getElementById('phaseLabel');
        const leftPanel = document.getElementById('leftPanel');
        const coordinateList = document.getElementById('coordinateList');
        const qubitPanel = document.getElementById('qubitPanel');
        const animGrid = document.getElementById('animGrid');
        const pixelIdentity = document.getElementById('pixelIdentity');

        if (phaseLabel) { phaseLabel.style.opacity = '0'; phaseLabel.textContent = ''; }
        if (coordinateList) { coordinateList.style.display = 'block'; coordinateList.innerHTML = ''; }
        if (qubitPanel) { qubitPanel.style.display = 'none'; qubitPanel.innerHTML = ''; }
        if (animGrid) { animGrid.innerHTML = ''; animGrid.style.gridTemplateColumns = ''; }
        if (pixelIdentity) { pixelIdentity.style.opacity = '0'; pixelIdentity.innerHTML = ''; }

        updatePlayButton(false);
    }

    function updatePlayButton(isPlaying) {
        const btn = document.getElementById('animPlayBtn');
        if (btn) btn.textContent = isPlaying ? '⏸ PAUSE' : '▶ PLAY';
    }

    async function runAnimation() {
        state.isPlaying = true;
        updatePlayButton(true);

        while (state.isPlaying) {
            await runClassicalPhase();
            if (!state.isPlaying) break;
            await delay(1000);

            await runQuantumPhase();
            if (!state.isPlaying) break;
            await delay(2000);
        }
    }

    async function runClassicalPhase() {
        state.phase = 'classical';
        const phaseLabel = document.getElementById('phaseLabel');
        const coordinateList = document.getElementById('coordinateList');
        const qubitPanel = document.getElementById('qubitPanel');
        const animGrid = document.getElementById('animGrid');

        // Setup UI
        if (coordinateList) { coordinateList.style.display = 'block'; coordinateList.innerHTML = ''; }
        if (qubitPanel) { qubitPanel.style.display = 'none'; }

        // Fade in label
        if (phaseLabel) {
            phaseLabel.textContent = 'CLASSICAL';
            phaseLabel.style.color = '#666';
            phaseLabel.style.opacity = '1';
        }
        await delay(500);

        // Initialize grid as single square
        if (animGrid) {
            animGrid.style.gridTemplateColumns = '1fr';
            animGrid.innerHTML = '<div class="anim-cell" style="width:200px; height:200px;"></div>';
        }
        await delay(500);

        // For demo, show first 8 pixels with decreasing delay
        const pixelsToShow = 16;
        for (let p = 0; p < pixelsToShow && state.isPlaying; p++) {
            const y = Math.floor(p / GRID_SIZE);
            const x = p % GRID_SIZE;
            const baseDelay = Math.max(100, 800 - p * 50); // Start slow, get faster

            // Level 1: Show 2x2
            await showSubdivision(2, x, y, 0);
            if (!state.isPlaying) return;
            await delay(baseDelay);

            // Level 2: Show 4x4
            await showSubdivision(4, x, y, 1);
            if (!state.isPlaying) return;
            await delay(baseDelay);

            // Level 3: Show 8x8
            await showSubdivision(8, x, y, 2);
            if (!state.isPlaying) return;
            await delay(baseDelay);

            // Add to coordinate list (interleaved format: y₀x₀,y₁x₁,y₂x₂)
            const yBin = y.toString(2).padStart(3, '0');
            const xBin = x.toString(2).padStart(3, '0');
            const interleaved = `${yBin[0]}${xBin[0]},${yBin[1]}${xBin[1]},${yBin[2]}${xBin[2]}`;
            addCoordinateToList(interleaved);

            await delay(baseDelay / 2);
        }

        // Show ellipsis
        addCoordinateToList('...');
        await delay(500);

        // Fade out
        if (phaseLabel) phaseLabel.style.opacity = '0';
        await delay(500);
    }

    async function showSubdivision(size, targetX, targetY, level) {
        const animGrid = document.getElementById('animGrid');
        if (!animGrid) return;

        const cellSize = Math.floor(200 / size);
        animGrid.style.gridTemplateColumns = `repeat(${size}, ${cellSize}px)`;
        animGrid.innerHTML = '';

        const scale = GRID_SIZE / size;
        const highlightX = Math.floor(targetX / scale);
        const highlightY = Math.floor(targetY / scale);

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const cell = document.createElement('div');
                cell.className = 'anim-cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;

                if (row === highlightY && col === highlightX) {
                    cell.classList.add('highlighted');
                }
                animGrid.appendChild(cell);
            }
        }

        // Update pixel identity display
        updatePixelIdentity(targetX, targetY, level);
    }

    function updatePixelIdentity(x, y, level) {
        const display = document.getElementById('pixelIdentity');
        if (!display) return;

        // Format as interleaved pairs: y₀x₀,y₁x₁,y₂x₂
        const yBin = y.toString(2).padStart(3, '0');
        const xBin = x.toString(2).padStart(3, '0');
        const interleaved = `${yBin[0]}${xBin[0]},${yBin[1]}${xBin[1]},${yBin[2]}${xBin[2]}`;

        display.innerHTML = `
            <div class="ui-text" style="font-size:0.65rem; margin-bottom:5px;">CURRENT PIXEL</div>
            <div style="font-family:monospace; font-size:1.1rem; font-weight:bold;">(${y}, ${x})</div>
            <div style="font-family:monospace; font-size:0.85rem; color:#333; margin-top:8px;">
                ${interleaved}
            </div>
            <div style="font-size:0.65rem; color:#888; margin-top:3px;">y₀x₀,y₁x₁,y₂x₂</div>
            <div style="font-size:0.7rem; color:#999; margin-top:8px;">Level ${level + 1}/${LEVELS}</div>
        `;
        display.style.opacity = '1';
    }

    function addCoordinateToList(text) {
        const list = document.getElementById('coordinateList');
        if (!list) return;
        const item = document.createElement('div');
        item.className = 'coord-item';
        item.textContent = text;
        list.appendChild(item);
        list.scrollTop = list.scrollHeight;
    }

    async function runQuantumPhase() {
        state.phase = 'quantum';
        const phaseLabel = document.getElementById('phaseLabel');
        const coordinateList = document.getElementById('coordinateList');
        const qubitPanel = document.getElementById('qubitPanel');
        const animGrid = document.getElementById('animGrid');

        // Setup UI
        if (coordinateList) { coordinateList.style.display = 'none'; coordinateList.innerHTML = ''; }
        if (qubitPanel) { qubitPanel.style.display = 'flex'; }
        const pixelIdentity = document.getElementById('pixelIdentity');
        if (pixelIdentity) { pixelIdentity.style.opacity = '0'; }

        // Fade in label (same gray color as classical)
        if (phaseLabel) {
            phaseLabel.textContent = 'QUANTUM';
            phaseLabel.style.color = '#666';
            phaseLabel.style.opacity = '1';
        }
        await delay(500);

        // Render qubits (y0,x0,y1,x1,y2,x2)
        renderQubits();
        await delay(500);

        // Initialize grid as single square
        if (animGrid) {
            animGrid.style.gridTemplateColumns = '1fr';
            animGrid.innerHTML = '<div class="anim-cell" style="width:200px; height:200px;"></div>';
        }
        await delay(500);

        // Level 1: 2x2, all 4 groups move simultaneously
        await quantumSubdivision(2, 0);
        if (!state.isPlaying) return;
        await delay(1000);

        // Level 2: 4x4, groups of 4 cells each
        await quantumSubdivision(4, 1);
        if (!state.isPlaying) return;
        await delay(1000);

        // Level 3: 8x8, groups of 16 cells each
        await quantumSubdivision(8, 2);
        if (!state.isPlaying) return;
        await delay(1500);

        // Fade out
        if (phaseLabel) phaseLabel.style.opacity = '0';
        await delay(500);
    }

    function renderQubits() {
        const panel = document.getElementById('qubitPanel');
        if (!panel) return;

        const labels = ['y₀', 'x₀', 'y₁', 'x₁', 'y₂', 'x₂'];
        let html = '<div class="ui-text" style="font-size:0.7rem; margin-bottom:10px;">QUBITS</div>';

        labels.forEach((label, i) => {
            html += `
                <div class="qubit-row" data-qubit="${i}">
                    <span class="qubit-label-text">${label}</span>
                    <div class="qubit-kets">
                        <span class="qubit-ket" data-val="0">|0⟩</span>
                        <span class="qubit-ket" data-val="1">|1⟩</span>
                    </div>
                </div>`;
        });
        panel.innerHTML = html;
    }

    async function quantumSubdivision(size, level) {
        const animGrid = document.getElementById('animGrid');
        if (!animGrid) return;

        const cellSize = Math.floor(200 / size);
        animGrid.style.gridTemplateColumns = `repeat(${size}, ${cellSize}px)`;
        animGrid.innerHTML = '';

        // Grayscale shades for groups
        const GRAY_SHADES = {
            '00': '#e8e8e8',
            '01': '#c8c8c8',
            '10': '#a8a8a8',
            '11': '#888888'
        };

        // Create cells with group data (checkerboard pattern using LSB)
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const cell = document.createElement('div');
                cell.className = 'anim-cell quantum-cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;

                // Group based on position within 2x2 blocks (LSB)
                const yBit = row & 1;
                const xBit = col & 1;
                const group = `${yBit}${xBit}`;
                cell.setAttribute('data-group', group);
                cell.style.background = '#e0e0e0';

                animGrid.appendChild(cell);
            }
        }

        // Animate each group sequentially: 00, 01, 10, 11
        for (const group of ['00', '01', '10', '11']) {
            if (!state.isPlaying) return;

            // Highlight this group's cells with BLACK background
            const cells = animGrid.querySelectorAll(`[data-group="${group}"]`);
            cells.forEach(c => {
                c.style.background = '#000';
                c.style.border = '2px solid #000';
            });

            // Show group label on qubits
            highlightQubitsForGroup(level, group);
            await delay(400);

            // Revert to grayscale shade
            cells.forEach(c => {
                c.style.background = GRAY_SHADES[group];
                c.style.border = '1px solid #999';
            });
        }

        // Final state: all groups shown with their shades
        highlightQubitsForLevel(level);
    }

    function highlightQubitsForGroup(level, group) {
        const panel = document.getElementById('qubitPanel');
        if (!panel) return;

        // Clear all highlights
        panel.querySelectorAll('.qubit-ket').forEach(k => {
            k.classList.remove('active-0', 'active-1', 'active-both');
        });

        const yBit = group[0];
        const xBit = group[1];
        const yQubitIdx = level * 2;
        const xQubitIdx = level * 2 + 1;

        // Highlight the specific ket for each qubit
        [yQubitIdx, xQubitIdx].forEach((qi, i) => {
            const bit = i === 0 ? yBit : xBit;
            const row = panel.querySelector(`.qubit-row[data-qubit="${qi}"]`);
            if (row) {
                const ket = row.querySelector(`.qubit-ket[data-val="${bit}"]`);
                if (ket) ket.classList.add(`active-${bit}`);
            }
        });
    }

    function highlightQubitsForLevel(level) {
        const panel = document.getElementById('qubitPanel');
        if (!panel) return;

        // Clear all highlights
        panel.querySelectorAll('.qubit-ket').forEach(k => {
            k.classList.remove('active-0', 'active-1', 'active-both');
        });

        // Highlight both kets for qubits at this level (showing superposition)
        // Level 0 = y0, x0 (indices 0, 1)
        // Level 1 = y1, x1 (indices 2, 3)
        // Level 2 = y2, x2 (indices 4, 5)
        const qubitIndices = [level * 2, level * 2 + 1];
        qubitIndices.forEach(qi => {
            const row = panel.querySelector(`.qubit-row[data-qubit="${qi}"]`);
            if (row) {
                row.querySelectorAll('.qubit-ket').forEach(k => {
                    k.classList.add('active-both');
                });
            }
        });
    }

    function delay(ms) {
        return new Promise(resolve => {
            const id = setTimeout(resolve, ms);
            state.timeoutIds.push(id);
        });
    }

    function toggle() {
        if (state.isPlaying) {
            stop();
        } else {
            runAnimation();
        }
    }

    function stop() {
        state.isPlaying = false;
        state.timeoutIds.forEach(id => clearTimeout(id));
        state.timeoutIds = [];
        updatePlayButton(false);
    }

    function restart() {
        stop();
        resetAnimation();
        runAnimation();
    }

    return { init, toggle, restart };
})();

// Global functions for button onclick
function toggleSuperposAnimation() { SuperpositionAnimation.toggle(); }
function restartSuperposAnimation() { SuperpositionAnimation.restart(); }

// Initialize if elements are present
document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('pixelGrid');
    if (grid) {
        const isPreprocessing = window.location.pathname.includes('preprocessing.html');
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
        updateMetricExplorer();
    }

    // Superposition Animation (Encoder Page)
    SuperpositionAnimation.init();

    // Simulation Live Demo
    const runSimBtn = document.getElementById('runSimBtn');
    if (runSimBtn) {
        runSimBtn.onclick = async () => {
            const status = document.getElementById('simStatus');
            const grid = document.getElementById('simGrid');
            const output = document.getElementById('simOutput');
            const dataEl = document.getElementById('simData');

            status.innerText = 'STATUS: SAMPLING...';
            runSimBtn.disabled = true;

            try {
                const res = await fetch('/api/simulate');
                const data = await res.json();

                grid.innerHTML = '';
                data.bins.forEach((b, i) => {
                    const div = document.createElement('div');
                    div.style.width = '100%';
                    div.style.height = '25px';
                    const intensity = Math.floor(b.hits / (b.hits + b.misses) * 255);
                    div.style.background = `rgb(${intensity}, ${intensity}, ${intensity})`;
                    div.title = `Hits: ${b.hits}, Misses: ${b.misses}`;
                    grid.appendChild(div);
                });

                output.style.display = 'block';
                dataEl.innerText = JSON.stringify(data.bins.slice(0, 10), null, 2) + '\n... (truncated)';
                status.innerText = 'STATUS: COMPLETE';
            } catch (err) {
                status.innerText = 'STATUS: ERROR';
                console.error(err);
            } finally {
                runSimBtn.disabled = false;
            }
        };
    }

    // Retrieval Demo
    const fetchRetrievalBtn = document.getElementById('fetchRetrievalBtn');
    if (fetchRetrievalBtn) {
        fetchRetrievalBtn.onclick = async () => {
            const dataEl = document.getElementById('retrievalData');
            dataEl.innerHTML = '<p style="font-size:0.8rem;">Fetching statistical mapping...</p>';

            try {
                const res = await fetch('/api/retrieval');
                const data = await res.json();
                let h = `<p style="font-size:0.8rem; margin-bottom:10px;"><strong>Formula:</strong> ${data.formula}</p>`;
                h += `<p style="font-size:0.8rem;"><strong>Example Bin:</strong> ${data.example_bin.hits} hits / ${data.example_bin.max_trials} trials &rarr; <span style="color:green;">${data.result_intensity} intensity</span></p>`;
                h += `<div style="margin-top:15px; display:flex; gap:10px;">`;
                data.mapping.forEach(m => {
                    h += `<div style="font-size:0.65rem; padding:5px; border:1px solid #ddd; background:#fff;">${m.label}<br>[${m.range[0]}-${m.range[1]}]</div>`;
                });
                h += `</div>`;
                dataEl.innerHTML = h;
            } catch (err) {
                dataEl.innerHTML = `<p style="color:red; font-size:0.7rem;">Error: ${err.message}</p>`;
            }
        };
    }
});
