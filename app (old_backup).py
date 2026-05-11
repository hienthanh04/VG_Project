"""
BACKEND — Flask API cho website phục hồi tranh Van Gogh
========================================================
Endpoints:
  POST /api/restore   — nhận ảnh, trả về ảnh phục hồi + thông tin tranh
  GET  /api/health    — kiểm tra server

Cách chạy:
  pip install flask flask-cors torch torchvision pillow opencv-python
  python app.py
  
Server chạy tại: http://localhost:5000
"""

import os
import io
import base64
import random
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Import Generator từ step2 ────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from step2_train_pix2pix import Generator

app = Flask(__name__)
CORS(app)

# ── Cấu hình ─────────────────────────────────────────────────────────────────

# Đường dẫn đến model tốt nhất — đổi nếu cần
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "grayscale": "./models/generator_grayscale.pth",
    "blur": "./models/generator_blur.pth",
    "blur_noise": "./models/generator_blur_noise.pth",
    "gray_blur_noise": "./models/generator_gray_blur_noise.pth",
}

MODELS = {}

def load_generator(model_path):
    if not os.path.exists(model_path):
        print(f"[Server] WARNING: Không tìm thấy model: {model_path}")
        return None

    model = Generator().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)

    if "G" in state:
        state = state["G"]

    model.load_state_dict(state)
    model.eval()
    print(f"[Server] Loaded model OK: {model_path}")
    return model

print(f"[Server] Device: {DEVICE}")

for mode, path in MODEL_PATHS.items():
    MODELS[mode] = load_generator(path)
# ── Database thông tin tranh Van Gogh ────────────────────────────────────────

VANGOGH_INFO = {
    "starry_night": {
        "title":    "The Starry Night",
        "year":     1889,
        "location": "Saint-Rémy-de-Provence, Pháp",
        "period":   "Saint-Rémy",
        "museum":   "Museum of Modern Art (MoMA), New York",
        "description": (
            "Một trong những tác phẩm nổi tiếng nhất của Van Gogh, "
            "vẽ từ cửa sổ phòng trú của ông tại bệnh viện Saint-Paul-de-Mausole. "
            "Bầu trời đêm xoáy cuộn với những ngôi sao rực rỡ là đặc trưng "
            "nét cọ năng động của giai đoạn Saint-Rémy."
        ),
        "style_traits": ["Xoáy cuộn năng động", "Màu xanh đậm và vàng tương phản", "Nét cọ dày, rõ hướng"],
    },
    "sunflowers": {
        "title":    "Sunflowers (Hoa hướng dương)",
        "year":     1888,
        "location": "Arles, Pháp",
        "period":   "Arles",
        "museum":   "National Gallery, London",
        "description": (
            "Vẽ tại Arles để trang trí phòng dành cho Paul Gauguin. "
            "Van Gogh đã tạo ra một loạt 7 bức tranh hoa hướng dương — "
            "biểu tượng cho niềm vui và sự biết ơn. "
            "Màu vàng đặc trưng phản ánh ánh nắng miền Nam nước Pháp."
        ),
        "style_traits": ["Vàng nóng đặc trưng Arles", "Nét cọ tròn, dày", "Bố cục đơn giản mạnh mẽ"],
    },
    "almond_blossom": {
        "title":    "Almond Blossom (Hoa hạnh nhân)",
        "year":     1890,
        "location": "Saint-Rémy-de-Provence, Pháp",
        "period":   "Saint-Rémy",
        "museum":   "Van Gogh Museum, Amsterdam",
        "description": (
            "Vẽ để tặng cho cháu trai mới sinh — Theo van Gogh con. "
            "Lấy cảm hứng từ tranh khắc gỗ Nhật Bản với hoa nở trên nền xanh lam. "
            "Là một trong những tác phẩm tươi sáng và lạc quan nhất của ông."
        ),
        "style_traits": ["Nền xanh lam yên tĩnh", "Cành hoa tinh tế", "Ảnh hưởng nghệ thuật Nhật Bản"],
    },
    "irises": {
        "title":    "Irises (Hoa diên vĩ)",
        "year":     1889,
        "location": "Saint-Rémy-de-Provence, Pháp",
        "period":   "Saint-Rémy",
        "museum":   "J. Paul Getty Museum, Los Angeles",
        "description": (
            "Vẽ trong tuần đầu tiên tại bệnh viện Saint-Paul-de-Mausole, "
            "khi Van Gogh có thể ra ngoài vẽ trong vườn. "
            "Màu xanh tím rực rỡ của hoa diên vĩ tương phản với nền cam vàng "
            "là ví dụ điển hình về cách dùng màu bổ trợ của ông."
        ),
        "style_traits": ["Màu bổ trợ xanh-cam", "Nét cọ uốn lượn", "Chi tiết hoa cỏ tỉ mỉ"],
    },
    "cafe_terrace": {
        "title":    "Café Terrace at Night",
        "year":     1888,
        "location": "Arles, Pháp",
        "period":   "Arles",
        "museum":   "Kröller-Müller Museum, Hà Lan",
        "description": (
            "Một trong những bức tranh đầu tiên của Van Gogh vẽ ban đêm "
            "ngoài trời, không dùng màu đen thuần túy. "
            "Ánh đèn vàng của quán café tương phản với bầu trời xanh đêm "
            "đầy sao là đặc trưng phong cách Arles."
        ),
        "style_traits": ["Ánh sáng nhân tạo vs thiên nhiên", "Vàng-xanh đêm đặc trưng", "Cảnh đường phố sống động"],
    },
    "bedroom_arles": {
        "title":    "Bedroom in Arles (Phòng ngủ tại Arles)",
        "year":     1888,
        "location": "Arles, Pháp",
        "period":   "Arles",
        "museum":   "Van Gogh Museum, Amsterdam",
        "description": (
            "Vẽ phòng ngủ của chính ông tại Ngôi nhà vàng ở Arles. "
            "Van Gogh cố ý dùng màu sắc mạnh để tạo cảm giác yên bình, nghỉ ngơi. "
            "Không gian bị bóp méo nhẹ tạo chiều sâu đặc trưng."
        ),
        "style_traits": ["Màu sắc mạnh, tươi sáng", "Góc nhìn bị bóp méo nhẹ", "Bố cục đơn giản, thân mật"],
    },
    "wheatfield_crows": {
        "title":    "Wheatfield with Crows",
        "year":     1890,
        "location": "Auvers-sur-Oise, Pháp",
        "period":   "Auvers",
        "museum":   "Van Gogh Museum, Amsterdam",
        "description": (
            "Một trong những tác phẩm cuối cùng của Van Gogh, "
            "vẽ chỉ vài tuần trước khi ông qua đời. "
            "Bầu trời bão tố đen xanh, đàn quạ loạn bay trên cánh đồng lúa mì vàng "
            "thường được diễn giải là biểu hiện tâm trạng bất an."
        ),
        "style_traits": ["Bầu trời bão tố dữ dội", "Tương phản vàng-đen-xanh", "Nét cọ mạnh, hỗn loạn"],
    },
    "farmhouse": {
        "title":    "Farmhouse in a Wheat Field",
        "year":     1888,
        "location": "Arles, Pháp",
        "period":   "Arles",
        "museum":   "Van Gogh Museum, Amsterdam",
        "description": (
            "Phong cảnh nông thôn Arles với những cánh đồng lúa mì vàng óng. "
            "Thể hiện rõ tình yêu của Van Gogh với thiên nhiên miền Nam nước Pháp "
            "và kỹ thuật dùng màu vàng đặc trưng giai đoạn Arles."
        ),
        "style_traits": ["Vàng lúa mì nổi bật", "Bầu trời xanh trong", "Nét cọ ngang, nhịp nhàng"],
    },
    "default": {
        "title":    "Tác phẩm Van Gogh",
        "year":     1888,
        "location": "Arles / Saint-Rémy, Pháp",
        "period":   "Arles",
        "museum":   "Van Gogh Museum, Amsterdam",
        "description": (
            "Tác phẩm thuộc giai đoạn Arles (1888–1889) — thời kỳ sáng tác "
            "prolific nhất của Van Gogh. Đặc trưng bởi màu vàng và xanh dương "
            "bão hòa cao, nét cọ dày và có hướng rõ ràng."
        ),
        "style_traits": ["Màu vàng-xanh đặc trưng Arles", "Nét cọ dày, năng động", "Độ bão hòa màu cao"],
    },
}

def get_painting_info(filename: str) -> dict:
    """Tìm thông tin tranh dựa vào tên file, fallback về default."""
    fname = filename.lower().replace(" ", "_").replace("-", "_")
    for key in VANGOGH_INFO:
        if key in fname and key != "default":
            return VANGOGH_INFO[key]
    # Random một tác phẩm thật thay vì luôn trả default
    real_paintings = [k for k in VANGOGH_INFO if k != "default"]
    random_key = random.choice(real_paintings)
    return VANGOGH_INFO[random_key]

# ── Xử lý ảnh ────────────────────────────────────────────────────────────────

def preprocess(img_pil: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    return transform(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)


def postprocess(tensor: torch.Tensor) -> Image.Image:
    img = tensor.squeeze(0).cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


def tensor_to_base64(img_pil: Image.Image) -> str:
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_basic_metrics(original_pil: Image.Image,
                           restored_pil: Image.Image) -> dict:
    """Tính SSIM đơn giản để hiển thị cho người dùng."""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import warnings; warnings.filterwarnings("ignore")

    orig = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE)))
    rest = np.array(restored_pil.resize((IMG_SIZE, IMG_SIZE)))
    try:
        s = ssim(orig, rest, channel_axis=2, data_range=255)
        p = psnr(orig, rest, data_range=255)
        return {"ssim": round(float(s), 4), "psnr": round(float(p), 2)}
    except Exception:
        return {"ssim": None, "psnr": None}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            mode: {
                "loaded": model is not None,
                "path": MODEL_PATHS[mode]
            }
            for mode, model in MODELS.items()
        }
    })


@app.route("/api/restore", methods=["POST"])
def restore():
    # ── Validate input ──
    if "image" not in request.files:
        return jsonify({"error": "Thiếu file ảnh. Gửi dưới dạng multipart/form-data với key 'image'"}), 400

    file     = request.files["image"]
    filename = file.filename or "unknown.jpg"
    mode = request.form.get("mode", "gray_blur_noise")

    if mode not in MODEL_PATHS:
        return jsonify({"error": f"Mode không hợp lệ: {mode}"}), 400

    G = MODELS.get(mode)
    if not file.content_type.startswith("image/"):
        return jsonify({"error": "File phải là ảnh (jpg, png, webp...)"}), 400

    try:
        input_pil = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Không đọc được ảnh: {str(e)}"}), 400

    # ── Chạy model ──
    if G is not None:
        with torch.no_grad():
            x       = preprocess(input_pil)
            out     = G(x)
            restored_pil = postprocess(out)
        metrics = compute_basic_metrics(input_pil, restored_pil)
    else:
        # Demo mode: trả về ảnh gốc nếu chưa có model
        restored_pil = input_pil.copy()
        metrics = {"ssim": None, "psnr": None, "note": "Demo mode — model chưa load"}

    # ── Resize về kích thước gốc ──
    orig_w, orig_h = input_pil.size
    restored_pil = restored_pil.resize((orig_w, orig_h), Image.LANCZOS)

    # ── Thông tin tranh ──
    painting_info = get_painting_info(filename)

    return jsonify({
        "restored_image": tensor_to_base64(restored_pil),
        "input_size":     [orig_w, orig_h],
        "metrics":        metrics,
        "painting_info":  painting_info,
        "selected_mode": mode,
    })


# ── Chạy server ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Van Gogh Restoration API")
    print(f"  http://localhost:5000")
    print(f"  Device: {DEVICE}")
    loaded_count = sum(1 for model in MODELS.values() if model is not None)
    print(f"  Models loaded: {loaded_count}/{len(MODELS)}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
