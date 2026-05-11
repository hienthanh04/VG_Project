"""
BACKEND — Van Gogh Restoration API (3 periods)
================================================
Trả về kết quả phục hồi từ cả 3 thời kỳ song song để người dùng so sánh.

Endpoints:
  POST /api/restore   — nhận ảnh + deg_type, trả về 3 kết quả (Arles/Paris/Netherlands)
  GET  /api/models    — danh sách models đang available
  GET  /api/health    — health check

Cách chạy:
  pip install flask flask-cors torch torchvision pillow opencv-python scikit-image
  python app.py
"""

import os, io, base64, random
import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))
from step2_train_pix2pix import Generator

app    = Flask(__name__)
CORS(app)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

MODELS_CONFIG = {
    "arles": {
        "grayscale": "./output_pix2pix/arles/grayscale/checkpoints/generator_best.pth",
        "blur": "./output_pix2pix/arles/blur/checkpoints/generator_best.pth",
        "blur_noise": "./output_pix2pix/arles/blur_noise/checkpoints/generator_best.pth",
        "gray_blur_noise": "./output_pix2pix/arles/gray_blur_noise/checkpoints/generator_best.pth",
    },
    "paris": {
        "grayscale": "./output_pix2pix/paris/grayscale/checkpoints/generator_best.pth",
        "blur": "./output_pix2pix/paris/blur/checkpoints/generator_best.pth",
        "blur_noise": "./output_pix2pix/paris/blur_noise/checkpoints/generator_best.pth",
        "gray_blur_noise": "./output_pix2pix/paris/gray_blur_noise/checkpoints/generator_best.pth",
    },
    "netherlands": {
        "grayscale": "./output_pix2pix/netherlands/grayscale/checkpoints/generator_best.pth",
        "blur": "./output_pix2pix/netherlands/blur/checkpoints/generator_best.pth",
        "blur_noise": "./output_pix2pix/netherlands/blur_noise/checkpoints/generator_best.pth",
        "gray_blur_noise": "./output_pix2pix/netherlands/gray_blur_noise/checkpoints/generator_best.pth",
    },
}

def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        G = Generator().to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        if "G" in state: state = state["G"]
        G.load_state_dict(state)
        G.eval()
        return G
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return None

print(f"\n[Server] Device: {DEVICE}")
print("[Server] Loading models...")
LOADED_MODELS = {}
for period, deg_map in MODELS_CONFIG.items():
    LOADED_MODELS[period] = {}
    for deg_type, path in deg_map.items():
        G = load_model(path)
        LOADED_MODELS[period][deg_type] = G
        print(f"  {period}/{deg_type}: {'OK' if G else 'not found'}")
print("[Server] Ready!\n")

PERIOD_INFO = {
    "arles": {
        "name": "Giai đoạn Arles", "years": "1888–1889",
        "location": "Arles, Provence, Pháp",
        "color_style": "Vàng và xanh bão hòa cao, nắng miền Nam",
        "description": "Thời kỳ sáng tác prolific nhất — hơn 200 tác phẩm trong 14 tháng. Đặc trưng bởi màu vàng rực rỡ và nét cọ dày, mạnh mẽ.",
        "notable": ["Sunflowers", "The Yellow House", "Café Terrace at Night"],
        "palette": ["#E8B84B", "#2A4A8C", "#8B4513"],
    },
    "paris": {
        "name": "Giai đoạn Paris", "years": "1886–1888",
        "location": "Paris, Pháp",
        "color_style": "Palette đa dạng, ảnh hưởng Ấn tượng Pháp",
        "description": "Tiếp xúc với trường phái Ấn tượng Pháp. Màu sắc sáng và đa dạng hơn. Nhiều self-portrait và cảnh quán café Paris.",
        "notable": ["Self-Portrait 1887", "Père Tanguy", "Italian Woman"],
        "palette": ["#9B59B6", "#27AE60", "#E74C3C"],
    },
    "netherlands": {
        "name": "Giai đoạn Hà Lan", "years": "1880–1886",
        "location": "Nuenen & The Hague, Hà Lan",
        "color_style": "Tối tăm, nâu đất, phản ánh cuộc sống nông dân",
        "description": "Giai đoạn đầu với tone màu tối. Chủ đề: cuộc sống nông dân và thợ mỏ. Ảnh hưởng Rembrandt và Millet.",
        "notable": ["The Potato Eaters", "The Weaver", "Head of a Peasant Woman"],
        "palette": ["#5D4037", "#827717", "#1A1A2E"],
    },
}

PAINTINGS_DB = {
    "arles": [
        {"title": "Sunflowers", "year": 1888, "desc": "Biểu tượng giai đoạn Arles — màu vàng đặc trưng."},
        {"title": "The Bedroom", "year": 1888, "desc": "Phòng ngủ tại Ngôi nhà vàng."},
        {"title": "Café Terrace at Night", "year": 1888, "desc": "Cảnh đêm với ánh đèn vàng."},
        {"title": "Almond Blossom", "year": 1890, "desc": "Vẽ tặng cháu trai, ảnh hưởng nghệ thuật Nhật Bản."},
    ],
    "paris": [
        {"title": "Self-Portrait with Grey Felt Hat", "year": 1887, "desc": "Self-portrait tiêu biểu giai đoạn Paris."},
        {"title": "Père Tanguy", "year": 1887, "desc": "Chân dung người bán màu yêu thích."},
        {"title": "Italian Woman", "year": 1887, "desc": "Chân dung với màu sắc rực rỡ."},
        {"title": "Le Moulin de la Galette", "year": 1886, "desc": "Cảnh quán cà phê Paris."},
    ],
    "netherlands": [
        {"title": "The Potato Eaters", "year": 1885, "desc": "Kiệt tác giai đoạn Hà Lan."},
        {"title": "The Weaver", "year": 1884, "desc": "Series khung dệt — cuộc sống công nhân."},
        {"title": "Head of a Peasant Woman", "year": 1885, "desc": "Chân dung nông dân Hà Lan."},
        {"title": "Old Church Tower at Nuenen", "year": 1885, "desc": "Tháp nhà thờ cổ u tối."},
    ],
}

def preprocess(img_pil):
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    return tf(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

def postprocess(tensor):
    img = (tensor.squeeze(0).cpu()*0.5+0.5).clamp(0,1)
    return Image.fromarray((img.permute(1,2,0).numpy()*255).astype(np.uint8))

def to_base64(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def compute_metrics(a, b):
    try:
        from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
        import warnings; warnings.filterwarnings("ignore")
        ar = np.array(a.resize((IMG_SIZE,IMG_SIZE)))
        br = np.array(b.resize((IMG_SIZE,IMG_SIZE)))
        return {"ssim": round(float(ssim(ar,br,channel_axis=2,data_range=255)),4),
                "psnr": round(float(psnr(ar,br,data_range=255)),2)}
    except: return {"ssim":None,"psnr":None}

@app.route("/api/health")
def health():
    available = {p:{d:(G is not None) for d,G in dm.items()} for p,dm in LOADED_MODELS.items()}
    return jsonify({"status":"ok","device":str(DEVICE),"models":available})

@app.route("/api/models")
def list_models():
    return jsonify({p:{"info":PERIOD_INFO[p],"deg_types":{d:(G is not None) for d,G in dm.items()}}
                    for p,dm in LOADED_MODELS.items()})

@app.route("/api/restore", methods=["POST"])
def restore():
    if "image" not in request.files:
        return jsonify({"error":"Thiếu field 'image'"}), 400
    file     = request.files["image"]
    deg_type = request.form.get("deg_type","grayscale")
    if deg_type not in ["grayscale","blur","blur_noise","gray_blur_noise"]:
        return jsonify({"error":f"deg_type không hợp lệ: {deg_type}"}), 400
    try:
        input_pil = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error":f"Không đọc được ảnh: {e}"}), 400

    orig_size = input_pil.size
    results = {}
    for period in ["arles","paris","netherlands"]:
        G = LOADED_MODELS[period].get(deg_type)
        if G is not None:
            with torch.no_grad():
                restored = postprocess(G(preprocess(input_pil)))
            results[period] = {
                "available":       True,
                "restored_b64":    to_base64(restored),
                "metrics":         compute_metrics(input_pil, restored),
                "period_info":     PERIOD_INFO[period],
                "sample_painting": random.choice(PAINTINGS_DB[period]),
            }
        else:
            results[period] = {
                "available":   False,
                "period_info": PERIOD_INFO[period],
                "message":     f"Model {period}/{deg_type} chưa train",
            }
    return jsonify({"input_size":list(orig_size),"deg_type":deg_type,"results":results})

if __name__ == "__main__":
    n = sum(1 for dm in LOADED_MODELS.values() for G in dm.values() if G)
    print(f"{'='*48}\n  Van Gogh Restoration — 3 Periods\n  http://localhost:5000  |  Models: {n}/12\n{'='*48}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
