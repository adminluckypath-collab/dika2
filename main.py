import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import json
import joblib
import cv2
import numpy as np
import numpy.random._pickle as np_random_pickle
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor, ViTModel
from skimage.feature import graycomatrix, graycoprops


def patch_numpy_bit_generator_pickle():
    original_ctor = np_random_pickle.__bit_generator_ctor

    def compatible_ctor(bit_generator_name="MT19937"):
        if not isinstance(bit_generator_name, str) and hasattr(bit_generator_name, "__name__"):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    np_random_pickle.__bit_generator_ctor = compatible_ctor


patch_numpy_bit_generator_pickle()

def get_cors_origins():
    origins = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


# Initialize FastAPI app
app = FastAPI(
    title="Apple Quality Detection API",
    description="API for detecting whether an apple is fresh or rotten using ViT and Gradient Boosting.",
    version="1.0.0"
)

# Enable CORS for frontend integration.
# Set CORS_ORIGINS on Railway to your Vercel URL, for example:
# https://apple-quality-web.vercel.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models and configurations
gb_model = None
scaler = None
vit_processor = None
vit_model = None
apple_classifier = None
vit_config = None
device = None
label_map_rev = None
APPLE_LABEL_KEYWORDS = ("apple", "granny smith")

@app.on_event("startup")
async def load_models():
    """
    Load all necessary models and configurations when the FastAPI application starts.
    This avoids reloading them for every request, improving performance.
    """
    global gb_model, scaler, vit_processor, vit_model, apple_classifier, vit_config, device, label_map_rev

    # Base directory where this script and model files are located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load configuration
    config_path = os.path.join(base_dir, "vit_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        vit_config = json.load(f)

    label_map_rev = {v: k for k, v in vit_config["label_mapping"].items()}

    # Load scikit-learn models (Scaler and Gradient Boosting)
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    gb_model_path = os.path.join(base_dir, "gb_model.pkl")

    try:
        scaler = joblib.load(scaler_path)
        gb_model = joblib.load(gb_model_path)
        print("Scaler and Gradient Boosting models loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler or GB model: {str(e)}")

    # Load ViT model from HuggingFace
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = vit_config["vit_model_name"]

    try:
        vit_processor = ViTImageProcessor.from_pretrained(model_name)
        vit_model = ViTModel.from_pretrained(model_name)
        apple_classifier = ViTForImageClassification.from_pretrained(model_name)
        vit_model.to(device)
        apple_classifier.to(device)
        vit_model.eval()
        apple_classifier.eval()
        print(f"ViT Model '{model_name}' loaded on {device}.")
    except Exception as e:
        raise RuntimeError(f"Failed to load ViT model from HuggingFace: {str(e)}")


def check_image_is_apple(img_resized: np.ndarray) -> dict:
    """
    Use the ImageNet head from the same ViT checkpoint as a lightweight guard.
    The quality model only knows fresh/rotten, so this prevents confident
    non-apple images from being forced into one of those two classes.
    """
    inputs = vit_processor(
        images=[img_resized],
        do_resize=False,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = apple_classifier(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    top_count = min(10, probabilities.shape[0])
    top_probabilities, top_indices = torch.topk(probabilities, k=top_count)
    labels = []

    for probability, index in zip(top_probabilities.cpu().tolist(), top_indices.cpu().tolist()):
        label = apple_classifier.config.id2label[index]
        labels.append({
            "label": label,
            "confidence": round(float(probability), 4),
        })

    is_apple = any(
        keyword in item["label"].lower()
        for item in labels
        for keyword in APPLE_LABEL_KEYWORDS
    )
    top_confidence = labels[0]["confidence"] if labels else 0.0

    return {
        "is_apple": is_apple,
        "confidence": top_confidence,
        "top_label": labels[0]["label"] if labels else "unknown",
        "top_labels": labels[:5],
    }


def ensure_rgb_3channel(img_rgb: np.ndarray) -> np.ndarray:
    """
    Pastikan gambar selalu memiliki 3 channel (RGB), uint8.
    Menangani kasus grayscale (2D) atau RGBA (4 channel) yang bisa lolos
    dari cv2.imdecode/cvtColor tergantung format file sumber.
    """
    if img_rgb is None:
        raise ValueError("Image array is None.")

    if img_rgb.ndim == 2:
        # Grayscale -> RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        # RGBA -> drop alpha channel
        img_rgb = img_rgb[:, :, :3]
    elif img_rgb.ndim == 3 and img_rgb.shape[2] == 1:
        # Single channel diberikan sebagai 3D -> RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {img_rgb.shape}")

    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    return np.ascontiguousarray(img_rgb)


def extract_texture_features(image_rgb_norm):
    """
    Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
    Expects normalized [0, 1] RGB image.
    """
    img_uint8 = (image_rgb_norm * 255).astype(np.uint8) if image_rgb_norm.dtype != np.uint8 else image_rgb_norm
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    return [contrast, dissimilarity, homogeneity, energy, correlation]


def extract_color_features(image_rgb_norm):
    """
    Extract color features (Mean and Std deviation of RGB channels).
    Expects normalized [0, 1] RGB image.
    """
    means = image_rgb_norm.mean(axis=(0, 1))
    stds = image_rgb_norm.std(axis=(0, 1))
    return list(means) + list(stds)


@app.post("/predict")
async def predict_apple(file: UploadFile = File(...)):
    """
    Endpoint to predict whether an apple is 'fresh' or 'rotten'.
    Accepts an image file upload.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read the uploaded image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Convert to RGB and enforce a consistent 3-channel uint8 array.
        # This is the key fix: some PNG/WebP/etc. files can still end up
        # with an unexpected channel count after imdecode/cvtColor on
        # certain OpenCV builds, which breaks batching inside the ViT
        # image processor with "Unable to create tensor..." errors.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = ensure_rgb_3channel(img_rgb)

        # Resize before creating a batch so every image has an identical
        # H x W x C shape. The padding hint from Transformers is generic;
        # ViT image batches need fixed spatial dimensions, not text padding.
        img_size = int(vit_config["image_size"])
        img_resized = cv2.resize(
            img_rgb,
            (img_size, img_size),
            interpolation=cv2.INTER_AREA,
        )
        img_resized = ensure_rgb_3channel(img_resized)

        apple_check = check_image_is_apple(img_resized)
        if not apple_check["is_apple"]:
            return {
                "success": True,
                "prediction": "not_apple",
                "message": "Ini bukan apel.",
                "confidence": apple_check["confidence"],
                "apple_check": apple_check,
                "filename": file.filename
            }

        # 1. Extract ViT Features
        inputs = vit_processor(
            images=[img_resized],
            do_resize=False,
            return_tensors="np",
        )
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None or pixel_values.shape != (1, 3, img_size, img_size):
            shape = None if pixel_values is None else tuple(pixel_values.shape)
            raise ValueError(f"Unexpected ViT input tensor shape: {shape}")

        # Converting through Python lists also supports older PyTorch builds
        # compiled against NumPy 1.x when NumPy 2.x is installed.
        inputs = {
            key: torch.tensor(value.tolist(), dtype=torch.float32, device=device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = vit_model(**inputs)

        cls_feat = np.asarray(
            outputs.last_hidden_state[:, 0, :].cpu().tolist(),
            dtype=np.float32,
        )
        cls_feat_scaled = scaler.transform(cls_feat)

        # 2. Extract Texture & Color Features (Requires manually resized and normalized [0, 1] image)
        img_norm = img_resized / 255.0

        texture_feat = np.array([extract_texture_features(img_norm)])
        color_feat = np.array([extract_color_features(img_norm)])

        # 3. Combine Features and Predict
        combined = np.concatenate([cls_feat_scaled, texture_feat, color_feat], axis=1)
        pred = gb_model.predict(combined)[0]

        result = label_map_rev.get(pred, "unknown")

        # Optional: Get probabilities
        probabilities = gb_model.predict_proba(combined)[0]
        prob_dict = {label_map_rev[i]: round(float(prob), 4) for i, prob in enumerate(probabilities)}

        return {
            "success": True,
            "prediction": result,
            "confidence": prob_dict[result],
            "probabilities": prob_dict,
            "apple_check": apple_check,
            "filename": file.filename
        }

    except HTTPException:
        # Re-raise FastAPI HTTPExceptions as-is (don't wrap them in 500)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "Apple Quality Detection API is running.",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # When running directly `python main.py`, start uvicorn.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
