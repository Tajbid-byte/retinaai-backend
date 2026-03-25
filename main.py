"""
RetinaAI — FastAPI Backend
Diabetic Retinopathy Screening — CNN-CatBoost Ensemble Simulation
Deploy on: Render.com / Railway / HuggingFace Spaces (FREE)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import hashlib
import io
import math
from PIL import Image
import uvicorn

app = FastAPI(
    title="RetinaAI DR Screening API",
    description="CNN-CatBoost Ensemble Diabetic Retinopathy Detection",
    version="1.0.0"
)

# Allow your Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel URL in production e.g. ["https://your-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GRADE METADATA ──────────────────────────────────────────────────────────
GRADES = {
    0: {
        "name": "No DR",
        "tag": "NORMAL",
        "color": "#22d98a",
        "urgency": "routine",
        "recommendation": (
            "No signs of diabetic retinopathy detected. Annual dilated eye examination "
            "recommended. Maintain optimal glycemic control (HbA1c < 7%), blood pressure "
            "< 130/80 mmHg, and lipid management. Continue current management plan."
        ),
        "findings": (
            "Clear fundus with no visible microaneurysms, hemorrhages, exudates, or "
            "neovascularization. Optic disc and macula appear normal."
        ),
    },
    1: {
        "name": "Mild NPDR",
        "tag": "NPDR",
        "color": "#ffb340",
        "urgency": "routine",
        "recommendation": (
            "Mild non-proliferative diabetic retinopathy detected. Annual monitoring "
            "recommended. Optimize HbA1c, blood pressure, and lipid levels. No retinal "
            "treatment required at this stage. Patient education on symptom awareness advised."
        ),
        "findings": (
            "Presence of at least one microaneurysm. No other retinopathy signs beyond "
            "mild microaneurysm formation in the posterior pole."
        ),
    },
    2: {
        "name": "Moderate NPDR",
        "tag": "NPDR",
        "color": "#ff9944",
        "urgency": "moderate",
        "recommendation": (
            "Moderate non-proliferative diabetic retinopathy detected. Ophthalmology "
            "referral within 3-6 months. Biannual retinal examination advised. Systemic "
            "risk factor optimization is critical. Assess for diabetic macular edema."
        ),
        "findings": (
            "Multiple microaneurysms, dot and blot hemorrhages, hard exudates, and/or "
            "cotton-wool spots observed. Changes more than mild but less than severe NPDR."
        ),
    },
    3: {
        "name": "Severe NPDR",
        "tag": "NPDR",
        "color": "#ff4466",
        "urgency": "urgent",
        "recommendation": (
            "⚠️ Severe NPDR detected — urgent ophthalmology referral required within 4 weeks. "
            "High risk (>50%) of progression to proliferative DR within 1 year. "
            "Pan-retinal photocoagulation may be indicated. Immediate systemic optimization."
        ),
        "findings": (
            "Severe hemorrhages and microaneurysms in all 4 quadrants, venous beading in "
            "≥2 quadrants, and/or intraretinal microvascular abnormalities (IRMA). "
            "4-2-1 rule criteria met."
        ),
    },
    4: {
        "name": "Proliferative DR",
        "tag": "PDR",
        "color": "#9966ff",
        "urgency": "emergency",
        "recommendation": (
            "🚨 Proliferative diabetic retinopathy — IMMEDIATE treatment required. "
            "Urgent vitreoretinal surgery consultation. Pan-retinal photocoagulation (PRP) "
            "or anti-VEGF therapy indicated. High risk of vision loss without intervention. "
            "Refer to retinal specialist within 1 week."
        ),
        "findings": (
            "Neovascularization of the disc (NVD) or elsewhere (NVE) detected. "
            "Possible vitreous hemorrhage, tractional retinal detachment, or fibrovascular "
            "proliferation. Advanced proliferative changes present."
        ),
    },
}

MODELS = ["EfficientNetV2M", "InceptionResNetV2", "Xception", "EfficientNetB3", "DenseNet169"]

# ── IMAGE FEATURE EXTRACTION ─────────────────────────────────────────────────
def extract_image_features(image: Image.Image) -> dict:
    """
    Extract real pixel-level features from the uploaded image.
    These features drive the grade prediction — different images give different results.
    """
    # Resize to standard
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # 1. Overall brightness (dark fundus images typical in DR)
    brightness = float(np.mean(arr))

    # 2. Red channel dominance (hemorrhages are red/dark-red)
    red_dominance = float(np.mean(r) - np.mean(b))

    # 3. Green channel (fundus vessels most visible in green)
    green_mean = float(np.mean(g))

    # 4. Contrast / std deviation (lesions create local contrast)
    contrast = float(np.std(arr))

    # 5. Bright spot ratio — exudates and optic disc are bright
    bright_mask = arr > 0.75
    bright_ratio = float(np.mean(bright_mask))

    # 6. Dark spot ratio — hemorrhages, microaneurysms are dark
    dark_mask = arr < 0.12
    dark_ratio = float(np.mean(dark_mask))

    # 7. Local variance in patches (lesion texture irregularity)
    patch_size = 28
    patch_vars = []
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            patch = g[i:i+patch_size, j:j+patch_size]
            patch_vars.append(float(np.var(patch)))
    local_variance = float(np.mean(patch_vars))

    # 8. Edge density (vessel and lesion boundaries)
    gy = np.abs(np.diff(g, axis=0)).mean()
    gx = np.abs(np.diff(g, axis=1)).mean()
    edge_density = float((gy + gx) / 2)

    # 9. Center-periphery ratio (DR lesions spread peripherally)
    cx, cy = 112, 112
    r_inner = 40
    center_mask = np.zeros((224,224), bool)
    for i in range(224):
        for j in range(0, 224, 4):  # sample for speed
            if (i-cx)**2 + (j-cy)**2 < r_inner**2:
                center_mask[i,j] = True
    center_bright = float(np.mean(arr[center_mask])) if center_mask.any() else brightness

    # 10. Color entropy proxy
    hist_r = np.histogram(r, bins=32)[0] / (224*224)
    entropy_r = float(-np.sum(hist_r * np.log(hist_r + 1e-9)))

    return {
        "brightness": brightness,
        "red_dominance": red_dominance,
        "green_mean": green_mean,
        "contrast": contrast,
        "bright_ratio": bright_ratio,
        "dark_ratio": dark_ratio,
        "local_variance": local_variance,
        "edge_density": edge_density,
        "center_bright": center_bright,
        "entropy_r": entropy_r,
    }


def features_to_grade(feats: dict, seed: int) -> np.ndarray:
    """
    Convert extracted image features into a grade probability vector.
    Deterministic given the same image — different images → different results.
    Mimics a trained CNN ensemble's softmax output behavior.
    """
    rng = np.random.RandomState(seed % (2**31))

    b   = feats["brightness"]
    rd  = feats["red_dominance"]
    gm  = feats["green_mean"]
    ct  = feats["contrast"]
    br  = feats["bright_ratio"]
    dr  = feats["dark_ratio"]
    lv  = feats["local_variance"]
    ed  = feats["edge_density"]
    ent = feats["entropy_r"]

    # ── Grade scoring function ──
    # Each grade has a "score" based on pathology-relevant features.
    # Inspired by actual DR grading criteria.

    # Grade 0 (No DR): bright, low red dominance, low dark spots, uniform
    s0 = (
        1.5 * b
        - 3.0 * dr
        - 2.0 * rd
        + 1.0 * gm
        - 2.0 * lv
        - 1.5 * ed
        + 0.5
    )

    # Grade 1 (Mild): slight darkness, minimal lesions
    s1 = (
        - abs(b - 0.35) * 2
        + rd * 2.0
        + dr * 3.0
        - lv * 1.5
        + 0.3
    )

    # Grade 2 (Moderate): more dark spots, higher variance, moderate red
    s2 = (
        dr * 5.0
        + rd * 3.0
        + lv * 4.0
        + ed * 2.0
        - b * 1.5
        + ct * 2.0
        - 0.5
    )

    # Grade 3 (Severe): high dark ratio, very high variance, high edge density
    s3 = (
        dr * 8.0
        + rd * 4.0
        + lv * 6.0
        + ed * 4.0
        - b * 2.0
        + ct * 3.0
        - 1.5
    )

    # Grade 4 (PDR): extreme features, highest irregularity
    s4 = (
        dr * 10.0
        + rd * 5.0
        + lv * 8.0
        + ed * 6.0
        - b * 3.0
        + ct * 4.0
        + ent * 0.5
        - 3.0
    )

    scores = np.array([s0, s1, s2, s3, s4])

    # Softmax
    scores = scores - scores.max()
    exp_s = np.exp(scores)
    probs = exp_s / exp_s.sum()

    # Add tiny per-model noise for realism (models don't agree perfectly)
    noise = rng.dirichlet(np.ones(5) * 20)  # very small noise
    probs = 0.88 * probs + 0.12 * noise
    probs = probs / probs.sum()

    return probs


def simulate_ensemble(feats: dict, image_hash: int) -> dict:
    """
    Simulate 5 CNN models each producing slightly different probability vectors.
    CatBoost meta-learner averages with learned weights.
    Returns final ensemble prediction.
    """
    # Each model uses a different seed → different noise → realistic disagreement
    model_seeds = [image_hash, image_hash+7, image_hash+13, image_hash+31, image_hash+97]

    model_probs = {}
    catboost_weights = [0.28, 0.24, 0.20, 0.16, 0.12]  # EfficientNetV2M weighted highest

    stacked_probs = np.zeros(5)

    for i, model_name in enumerate(MODELS):
        probs = features_to_grade(feats, model_seeds[i])
        model_probs[model_name] = probs
        stacked_probs += catboost_weights[i] * probs

    stacked_probs /= stacked_probs.sum()

    predicted_grade = int(np.argmax(stacked_probs))
    confidence = float(stacked_probs[predicted_grade])

    # Per-model top-1 confidence for the predicted grade
    model_scores = {
        name: float(model_probs[name][predicted_grade])
        for name in MODELS
    }

    # LIME region importance (normalized feature contributions per grade)
    lime_weights = {
        "microaneurysms": float(feats["dark_ratio"] * 5),
        "hemorrhages":    float(feats["red_dominance"] * 4 + feats["dark_ratio"] * 3),
        "exudates":       float(feats["bright_ratio"] * 3),
        "vessels":        float(feats["edge_density"] * 4),
        "neovascularization": float(feats["local_variance"] * 5),
    }
    total = sum(lime_weights.values()) + 1e-9
    lime_weights = {k: round(v / total, 4) for k, v in lime_weights.items()}

    return {
        "grade": predicted_grade,
        "grade_name": GRADES[predicted_grade]["name"],
        "grade_tag": GRADES[predicted_grade]["tag"],
        "grade_color": GRADES[predicted_grade]["color"],
        "urgency": GRADES[predicted_grade]["urgency"],
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in stacked_probs],
        "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
        "findings": GRADES[predicted_grade]["findings"],
        "recommendation": GRADES[predicted_grade]["recommendation"],
        "lime_weights": lime_weights,
        "image_features": {
            "brightness": round(feats["brightness"], 4),
            "contrast": round(feats["contrast"], 4),
            "dark_lesion_ratio": round(feats["dark_ratio"], 4),
            "red_channel_dominance": round(feats["red_dominance"], 4),
            "edge_density": round(feats["edge_density"], 4),
        },
        "model_used": "CNN-CatBoost Stacking Ensemble (Simulated — Weights Pending Training)",
        "disclaimer": (
            "This is a research demonstration. Results are based on image feature analysis "
            "and simulated ensemble behavior. Not for clinical use. Replace model_probs with "
            "your trained .h5 weights for real clinical-grade predictions."
        ),
    }


# ── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "RetinaAI DR Screening API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Upload fundus image → get DR grade prediction",
            "GET /health":   "Health check",
            "GET /docs":     "Auto-generated API docs (FastAPI Swagger UI)",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": False, "mode": "simulation"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file (JPG, PNG, WEBP, etc.)")

    # Read image bytes
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="Image too large. Please use an image under 20MB.")

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a valid image file.")

    # Deterministic hash — same image always gives same result
    image_hash = int(hashlib.md5(contents).hexdigest(), 16)

    # Extract real pixel features
    feats = extract_image_features(image)

    # Run ensemble simulation
    result = simulate_ensemble(feats, image_hash)

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)