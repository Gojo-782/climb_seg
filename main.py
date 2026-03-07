"""
FastAPI server for hold/volume segmentation using the same Detectron2 model
as in hold-segmentation.ipynb. Returns bounding boxes, segmentations (polygons/masks),
and optional visualization.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Detectron2 imports (after FastAPI so startup is quick before heavy imports)
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# ---------------------------------------------------------------------------
# Paths: same config/weights as the notebook (run from climb_seg or project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LATER_DIR = os.path.join(_SCRIPT_DIR, "..", "Later")
CONFIG_FILE = os.path.join(LATER_DIR, "experiment_config.yml")
MODEL_PATH = os.path.join(LATER_DIR, "model_complete.pth")

CLASS_NAMES = ["hold", "volume"]

# Global predictor (loaded once at startup)
predictor = None
metadata = None


def load_model():
    """Load Detectron2 model exactly like hold-segmentation.ipynb."""
    global predictor, metadata
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_weights = checkpoint["model_state_dict"]
        fd, temp_weights_path = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        try:
            torch.save(model_weights, temp_weights_path)
            weights_path = temp_weights_path
        except Exception:
            if os.path.exists(temp_weights_path):
                os.remove(temp_weights_path)
            raise
    else:
        temp_weights_path = None
        weights_path = MODEL_PATH

    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    MetadataCatalog.get("meta").thing_classes = CLASS_NAMES
    metadata = MetadataCatalog.get("meta")
    predictor = DefaultPredictor(cfg)

    if temp_weights_path and os.path.exists(temp_weights_path):
        try:
            os.remove(temp_weights_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hold segmentation API",
    description="Run Detectron2 hold/volume segmentation (same model as hold-segmentation.ipynb)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    load_model()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    """Single detection with bbox and segmentation (polygon + optional mask image)."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    score: float
    polygon: List[List[float]]  # [[x,y], ...] contour in image coordinates
    mask_base64: Optional[str] = None  # PNG of the mask (same size as image)


class PredictResponse(BaseModel):
    detections: List[Detection]
    image_with_boxes_base64: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """Convert binary mask (H,W) to polygon [[x,y], ...] using largest contour."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in points.tolist()]


def _mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask (H,W) as PNG base64."""
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _run_predict_on_image(
    img: np.ndarray, include_mask_images: bool = True
) -> tuple:
    """Run predictor on BGR image; return (detections with segmentations, vis_image_rgb or None)."""
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    has_masks = instances.has("pred_masks")
    masks = instances.pred_masks.numpy() if has_masks else None

    detections: List[Detection] = []
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = box.tolist()
        class_name = (
            CLASS_NAMES[int(cls)] if 0 <= int(cls) < len(CLASS_NAMES) else "unknown"
        )
        polygon: List[List[float]] = []
        mask_base64: Optional[str] = None
        if has_masks and masks is not None:
            mask = masks[i]
            polygon = _mask_to_polygon(mask)
            if include_mask_images:
                mask_base64 = _mask_to_base64(mask)
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                class_name=class_name,
                score=float(score),
                polygon=polygon,
                mask_base64=mask_base64,
            )
        )

    vis_img = None
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out_vis = v.draw_instance_predictions(instances)
    vis_img = out_vis.get_image()
    return detections, vis_img


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


TEST_IMAGE_PATH = os.path.join(LATER_DIR, "test_2.jpeg")


@app.get("/predict/test-image", response_model=PredictResponse)
def predict_test_image(include_mask_images: bool = True):
    """
    Run the model on the built-in test image (test_2.jpeg). Returns detections
    with bounding boxes and segmentations (polygon + optional mask image).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not os.path.exists(TEST_IMAGE_PATH):
        raise HTTPException(status_code=404, detail="Test image not found")
    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read test image")
    detections, vis_img = _run_predict_on_image(img, include_mask_images=include_mask_images)
    image_with_boxes_base64 = None
    if vis_img is not None:
        _, buf = cv2.imencode(".png", vis_img[:, :, ::-1])
        image_with_boxes_base64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return PredictResponse(
        detections=detections,
        image_with_boxes_base64=image_with_boxes_base64,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    include_visualization: bool = True,
    include_mask_images: bool = True,
):
    """
    Upload an image; run the model and return detections with bounding boxes
    and segmentations (polygon + optional mask image per detection), and
    optionally a base64 visualization image with boxes/masks drawn.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    detections, vis_img = _run_predict_on_image(
        img, include_mask_images=include_mask_images
    )
    image_with_boxes_base64 = None
    if include_visualization and vis_img is not None:
        _, buf = cv2.imencode(".png", vis_img[:, :, ::-1])
        image_with_boxes_base64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return PredictResponse(
        detections=detections,
        image_with_boxes_base64=image_with_boxes_base64,
    )
