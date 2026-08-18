import os
import sys
import json
import math
import uuid
import numpy as np
import cv2
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from http.cookies import SimpleCookie

# Add parent directory to path so we can import circuit.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Qiskit and circuit imports
try:
    from qiskit import QuantumCircuit, QuantumRegister
    import circuit as mhrqi_circuit
    import utils
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"Warning: Could not import circuit module: {e}")

HOST = "0.0.0.0"
PORT = 21191
WEB_ROOT = os.path.join(os.path.dirname(__file__), "site")
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

# Available images (1 per category)
AVAILABLE_IMAGES = {
    "cnv": "cnv1.jpeg",
    "dme": "dme1.jpeg",
    "drusen": "drusen1.jpeg",
    "normal": "normal1.jpeg"
}

# Session storage (in-memory for demo)
SESSIONS = {}


def get_or_create_session(cookie_header):
    """Get or create a session from cookie header."""
    session_id = None
    if cookie_header:
        cookie = SimpleCookie()
        cookie.load(cookie_header)
        if 'mhrqi_session' in cookie:
            session_id = cookie['mhrqi_session'].value
    
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    
    # Create new session
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "selected_image": "drusen",
        "selected_scale": 8
    }
    return session_id, SESSIONS[session_id]


def preprocess_image(img_path, n):
    """Load and preprocess image for circuit generation."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (n, n))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    normalized = np.clip(img.astype(np.float64) / 255.0, 0.0, 1.0)
    return normalized


def build_hierarchy_matrix(n, d):
    """Build hierarchy matrix for image of size n×n."""
    L_max = utils.get_Lmax(n, d)
    sk = []
    for L in range(0, L_max):
        sk.append(n if L == 0 else utils.get_subdiv_size(L, n, d))
    
    hierarchy_matrix = []
    for r, c in np.ndindex(n, n):
        hcv = []
        for k in sk:
            sub_hcv = utils.compute_register(r, c, d, k)
            hcv.extend(sub_hcv)
        hierarchy_matrix.append(hcv)
    
    return hierarchy_matrix, L_max


def circuit_to_json(qc):
    """Parses a Qiskit circuit into a JSON structure for 2D rendering."""
    if qc is None:
        return {"error": "Circuit unavailable"}
        
    gates = []
    qubit_map = {q: i for i, q in enumerate(qc.qubits)}
    
    for circuit_instr in qc.data:
        instr = circuit_instr.operation
        qargs = circuit_instr.qubits
        name = instr.name
        indices = [qubit_map[q] for q in qargs]
        params = [float(p) if isinstance(p, (float, int, np.float64)) else str(p) for p in instr.params]
        
        gates.append({
            "name": name,
            "qubits": indices,
            "params": params
        })
    
    # Build qubit info with proper labels
    qubits_info = []
    for i, q in enumerate(qc.qubits):
        try:
            reg = getattr(q, 'register', None) or getattr(q, '_register', None)
            if reg is not None:
                reg_name = reg.name
                if reg.size > 1:
                    idx_in_reg = list(reg).index(q)
                    label = f"{reg_name}[{idx_in_reg}]"
                else:
                    label = reg_name
            else:
                label = f"q{i}"
        except:
            label = f"q{i}"
        qubits_info.append({"index": i, "register": label})

    return {
        "qubits": qubits_info,
        "gates": gates
    }


class MHRQIHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed_path = urlparse(path)
        rel_path = parsed_path.path.lstrip('/')
        if not rel_path:
            rel_path = 'index.html'
        return os.path.join(WEB_ROOT, rel_path)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Serve images from resources
        if parsed_path.path.startswith('/resources/'):
            self.serve_resource(parsed_path.path)
        elif parsed_path.path.startswith('/api/'):
            self.handle_api(parsed_path)
        else:
            full_path = self.translate_path(self.path)
            if os.path.isdir(full_path):
                full_path = os.path.join(full_path, 'index.html')
            
            if os.path.exists(full_path):
                super().do_GET()
            else:
                self.send_error(404, "File Not Found")

    def serve_resource(self, path):
        """Serve image files from resources directory."""
        rel_path = path.replace('/resources/', '', 1)
        full_path = os.path.join(RESOURCES_DIR, rel_path)
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            self.send_response(200)
            if full_path.endswith('.jpeg') or full_path.endswith('.jpg'):
                self.send_header('Content-Type', 'image/jpeg')
            elif full_path.endswith('.png'):
                self.send_header('Content-Type', 'image/png')
            self.end_headers()
            with open(full_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "Resource Not Found")

    def handle_api(self, parsed_path):
        query = parse_qs(parsed_path.query)
        path = parsed_path.path
        
        # Get session
        cookie_header = self.headers.get('Cookie')
        session_id, session = get_or_create_session(cookie_header)
        
        response_data = {}
        
        try:
            if path == '/api/images':
                # List available images
                response_data = {
                    "images": [
                        {"id": "cnv", "name": "CNV", "file": AVAILABLE_IMAGES["cnv"]},
                        {"id": "dme", "name": "DME", "file": AVAILABLE_IMAGES["dme"]},
                        {"id": "drusen", "name": "Drusen", "file": AVAILABLE_IMAGES["drusen"]},
                        {"id": "normal", "name": "Normal", "file": AVAILABLE_IMAGES["normal"]}
                    ],
                    "scales": [8, 16],
                    "session": {
                        "selected_image": session.get("selected_image", "drusen"),
                        "selected_scale": session.get("selected_scale", 8)
                    }
                }

            elif path == '/api/encoder/circuit':
                image_id = query.get('image', [session.get('selected_image', 'drusen')])[0]
                n = int(query.get('n', [session.get('selected_scale', 8)])[0])
                d = 2
                
                # Update session
                session['selected_image'] = image_id
                session['selected_scale'] = n
                
                if image_id not in AVAILABLE_IMAGES:
                    response_data = {"error": f"Unknown image: {image_id}"}
                elif not QISKIT_AVAILABLE:
                    response_data = {"error": "Qiskit not available"}
                else:
                    # Load and preprocess image
                    img_path = os.path.join(RESOURCES_DIR, AVAILABLE_IMAGES[image_id])
                    img = preprocess_image(img_path, n)
                    
                    # Build hierarchy matrix
                    hierarchy_matrix, L_max = build_hierarchy_matrix(n, d)
                    
                    # Create circuit (ONLY CIRCUIT, NO SIMULATION)
                    qc, pos_regs, intensity_reg, bias = mhrqi_circuit.MHRQI_init(d, L_max)
                    qc = mhrqi_circuit.MHRQI_upload(qc, pos_regs, intensity_reg, d, hierarchy_matrix, img)
                    
                    response_data = circuit_to_json(qc)
                    response_data["info"] = {
                        "image": image_id,
                        "scale": n,
                        "L_max": L_max,
                        "total_gates": len(response_data["gates"]),
                        "total_qubits": len(response_data["qubits"])
                    }

            elif path == '/api/denoiser/circuit':
                image_id = query.get('image', [session.get('selected_image', 'drusen')])[0]
                n = int(query.get('n', [session.get('selected_scale', 8)])[0])
                d = 2
                
                # Update session
                session['selected_image'] = image_id
                session['selected_scale'] = n
                
                if image_id not in AVAILABLE_IMAGES:
                    response_data = {"error": f"Unknown image: {image_id}"}
                elif not QISKIT_AVAILABLE:
                    response_data = {"error": "Qiskit not available"}
                else:
                    # Load and preprocess image
                    img_path = os.path.join(RESOURCES_DIR, AVAILABLE_IMAGES[image_id])
                    img = preprocess_image(img_path, n)
                    
                    # Build hierarchy matrix
                    hierarchy_matrix, L_max = build_hierarchy_matrix(n, d)
                    
                    # Create circuit (ONLY CIRCUIT, NO SIMULATION)
                    qc, pos_regs, intensity_reg, bias = mhrqi_circuit.MHRQI_init(d, L_max)
                    qc = mhrqi_circuit.MHRQI_upload(qc, pos_regs, intensity_reg, d, hierarchy_matrix, img)
                    qc, denoise_qc = mhrqi_circuit.DENOISER(qc, pos_regs, intensity_reg, bias)
                    
                    # Use only the denoiser circuit (not the full encoder+denoiser)
                    response_data = circuit_to_json(denoise_qc)
                    response_data["info"] = {
                        "image": image_id,
                        "scale": n,
                        "L_max": L_max,
                        "total_gates": len(response_data["gates"]),
                        "total_qubits": len(response_data["qubits"]),
                        "includes_denoiser": True
                    }

            elif path == '/api/simulate':
                # Lazy-loaded simulation (only if user explicitly requests)
                n = int(query.get('n', [8])[0])
                if n > 16:
                    response_data = {"error": "Scale too large for demo simulation (max 16)"}
                else:
                    # Placeholder - actual simulation would go here
                    response_data = {
                        "status": "not_implemented",
                        "message": "Simulation endpoint for future use"
                    }

            elif path == '/api/retrieval':
                response_data = {
                    "algorithm": {
                        "step1": {"name": "Outcome Interpretation", "formula": "outcome=0 → preserve, outcome=1 → smooth"},
                        "step2": {"name": "Confidence Calculation", "formula": "confidence = hits / (hits + misses)"},
                        "step3": {"name": "Trusted Neighbor Selection", "formula": "trusted if neighbor_confidence <= 0.7"},
                        "step4": {"name": "Blending", "formula": "I_final = conf × I_original + (1-conf) × median(trusted_neighbors)"}
                    },
                    "threshold": 0.7
                }
            
            else:
                self.send_error(404, "API Endpoint Not Found")
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            # Set session cookie
            self.send_header('Set-Cookie', f'mhrqi_session={session_id}; Path=/; HttpOnly')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Internal Server Error: {str(e)}")


def main():
    httpd = HTTPServer((HOST, PORT), MHRQIHandler)
    print(f"Serving MHRQI API and site on {HOST}:{PORT}")
    print(f"circuit.py available: {QISKIT_AVAILABLE}")
    print(f"Resources directory: {RESOURCES_DIR}")
    print(f"Available images: {list(AVAILABLE_IMAGES.keys())}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
