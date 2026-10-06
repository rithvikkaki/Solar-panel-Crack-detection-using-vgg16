"""
SolarSentinel AI - Backend API Integration Tests
Tests:
- Health endpoint status (now reporting active model)
- Real production inference via /api/v1/inspect
- Real Grad-CAM generation
- Demo status report
- Demo visualization generation
- File validation (unsupported extension, corrupt image, empty file)
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient with application lifespan context."""
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    """Verify root information endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "SolarSentinel AI"
    assert "docs_url" in data


def test_health_endpoint(client):
    """Verify health endpoint accurately reflects model status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "SolarSentinel AI"
    assert data["model_loaded"] is True
    assert data["model_mode"] == "REAL_MODEL"
    assert isinstance(data["verified_classes"], list)
    assert len(data["verified_classes"]) == 6


def test_demo_status_endpoint(client):
    """Verify demo status reports current capability honestly."""
    response = client.get("/api/v1/demo/status")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "REAL_MODEL"
    assert data["model_available"] is True


def test_real_inspect_endpoint(client):
    """
    Verify production inference endpoint on an actual image.
    Validates:
    - 200 OK status
    - Real prediction & confidence
    - Probability distribution summing to 1.0
    - Real Grad-CAM base64 images (original, heatmap, overlay)
    """
    img = Image.new("RGB", (244, 244), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/inspect",
        files={"file": ("test_panel.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_condition"] in data["all_classes"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["class_probabilities"]) == 6

    # Verify probability distribution sums to 1.0
    prob_sum = sum(data["class_probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-2

    # Verify Grad-CAM
    assert "explainability" in data
    assert "images" in data["explainability"]
    assert data["explainability"]["images"]["original"].startswith("data:image/png;base64,")
    assert data["explainability"]["images"]["heatmap"].startswith("data:image/png;base64,")
    assert data["explainability"]["images"]["overlay"].startswith("data:image/png;base64,")
    assert "Grad-CAM visualizes image regions" in data["explainability"]["disclaimer"]

    # Verify health assessment
    assert 0 <= data["health_assessment"]["health_score"] <= 100
    assert data["health_assessment"]["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert data["model_metadata"]["is_demo_preview"] is False


def test_demo_inspect_endpoint(client):
    """Verify demo preview endpoint generates complete labeled payload."""
    img = Image.new("RGB", (120, 120), color=(80, 120, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/v1/demo/inspect",
        files={"file": ("demo_panel.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["inspection_id"].startswith("demo-")
    assert "DEMO VISUALIZATION" in data["explainability"]["disclaimer"]
    assert data["model_metadata"]["is_demo_preview"] is True


def test_inspect_unsupported_extension(client):
    """Verify rejection of non-image file extensions."""
    buf = io.BytesIO(b"Hello world text file")
    response = client.post(
        "/api/v1/demo/inspect",
        files={"file": ("malicious.txt", buf, "text/plain")}
    )
    assert response.status_code == 422
    assert "Unsupported file extension" in response.json()["detail"]


def test_inspect_corrupt_image(client):
    """Verify rejection of corrupted image bytes with .jpg extension."""
    buf = io.BytesIO(b"RANDOM_CORRUPT_BYTES_NOT_AN_IMAGE")
    response = client.post(
        "/api/v1/demo/inspect",
        files={"file": ("corrupt.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 422
    assert "Corrupted or non-decodable" in response.json()["detail"]


def test_inspect_empty_file(client):
    """Verify rejection of empty file (0 bytes)."""
    buf = io.BytesIO(b"")
    response = client.post(
        "/api/v1/demo/inspect",
        files={"file": ("empty.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 422
    assert "empty (0 bytes)" in response.json()["detail"]


def test_ready_endpoint(client):
    """Verify /api/v1/ready returns readiness and operational mode."""
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["model_loaded"] is True
    assert data["mode"] == "REAL_MODEL"
    assert "model_path" in data
    assert "timestamp" in data


def test_inspect_png_image(client):
    """Verify PNG images are properly accepted and processed."""
    img = Image.new("RGB", (244, 244), color=(80, 140, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("test.png", buf, "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["predicted_condition"] in response.json()["all_classes"]


def test_inspect_webp_image(client):
    """Verify WebP images are properly accepted and processed."""
    img = Image.new("RGB", (200, 200), color=(120, 180, 90))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    buf.seek(0)
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("test.webp", buf, "image/webp")}
    )
    assert response.status_code == 200
    assert response.json()["predicted_condition"] in response.json()["all_classes"]


def test_inspect_tiny_image(client):
    """Verify extremely small images (8x8) do not crash the service."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("tiny.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["predicted_condition"] in response.json()["all_classes"]


def test_inspect_large_image(client):
    """Verify high resolution images (1500x1500) are accepted and resized."""
    img = Image.new("RGB", (1500, 1500), color=(40, 70, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("large.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["predicted_condition"] in response.json()["all_classes"]


@pytest.mark.parametrize("class_name, sample_file", [
    ("Bird-drop", "Bird (1).jpeg"),
    ("Clean", "Clean (1).jpeg"),
    ("Dusty", "Dust (1).jpg"),
    ("Electrical-damage", "Electrical (1).jpg"),
    ("Physical-Damage", "Physical (1).jpg"),
    ("Snow-Covered", "Snow (1).jpg"),
])
def test_real_dataset_image_inference(client, class_name, sample_file):
    """
    Executes real inference on an actual dataset sample for each class.
    SCIENTIFIC HONESTY: Asserts HTTP 200, valid schema, probability sum ~ 1, and Grad-CAM generation.
    Does NOT assert prediction must match directory label (which would be scientifically dishonest).
    """
    import os
    img_path = os.path.join("dataset", "Faulty_solar_panel", class_name, sample_file)
    if not os.path.exists(img_path):
        pytest.skip(f"Dataset file {img_path} not found")

    with open(img_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/inspect",
        files={"file": (sample_file, io.BytesIO(file_bytes), "image/jpeg")}
    )
    assert response.status_code == 200, f"Failed on {class_name}: {response.text}"
    data = response.json()

    # Schema assertions
    assert data["predicted_condition"] in data["all_classes"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["class_probabilities"]) == 6
    prob_sum = sum(data["class_probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.02

    # Grad-CAM checks
    xai = data["explainability"]
    assert xai["images"]["original"].startswith("data:image/png;base64,")
    assert xai["images"]["heatmap"].startswith("data:image/png;base64,")
    assert xai["images"]["overlay"].startswith("data:image/png;base64,")
    assert "Grad-CAM visualizes image regions" in xai["disclaimer"]

    # Health assessment check
    assert 0 <= data["health_assessment"]["health_score"] <= 100
    assert data["health_assessment"]["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    match = (data["predicted_condition"] == class_name)
    print(
        f"\n[DATASET TEST] Directory: {class_name:<18} | "
        f"Predicted: {data['predicted_condition']:<18} | "
        f"Confidence: {data['confidence']:.4f} | "
        f"Match: {'YES' if match else 'NO'}"
    )


def test_split_manifest_and_audit():
    """Verifies that split_manifest_70_15_15.json and split_audit.json exist and show no leakage."""
    import json
    import os
    assert os.path.exists("ml/metadata/split_manifest_70_15_15.json")
    assert os.path.exists("ml/metadata/split_audit.json")

    with open("ml/metadata/split_audit.json", "r") as f:
        audit = json.load(f)

    assert audit["path_overlap_count"] == 0
    assert audit["hash_leakage_groups_count"] == 0
    assert audit["data_leakage_detected"] is False


def test_confidence_level_policy(client):
    """Verify confidence level classification in inspection responses."""
    img = Image.new("RGB", (244, 244), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/inspect",
        files={"file": ("test_panel.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "confidence_level" in data
    conf = data["confidence"]
    level = data["confidence_level"]
    if conf >= 0.85:
        assert level == "HIGH"
    elif conf >= 0.60:
        assert level == "MODERATE"
    else:
        assert level == "LOW"


def test_model_metrics_endpoint(client):
    """Verify GET /api/v1/model/metrics returns production model identity and metrics."""
    response = client.get("/api/v1/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["production_model"]["model_id"] in [
        "vgg16_baseline_production",
        "vgg16_head_b_photometric_production",
        "vgg16_dual_head_ensemble_production",
        "vgg16_res384_block5_production"
    ]
    assert data["production_model"]["architecture"] == "VGG16"
    assert "block5_conv3" in data["production_model"]["gradcam_target_layer"]
    
    # Verify numeric evaluation metrics
    assert "test_evaluation" in data
    assert isinstance(data["test_evaluation"]["total_test_samples"], int)
    assert data["test_evaluation"]["total_test_samples"] == 131
    assert "overall" in data["test_evaluation"]
    assert isinstance(data["test_evaluation"]["overall"]["accuracy"], float)
    assert 0.80 <= data["test_evaluation"]["overall"]["accuracy"] <= 0.90
    assert len(data["test_evaluation"]["per_class"]) == 6

    # Verify benchmark latency presence
    assert "performance_benchmark" in data
    assert "total_pipeline_latency_ms" in data["performance_benchmark"]


def test_demo_sample_images_endpoint(client):
    """Verify GET /api/v1/demo/sample-images and individual download."""
    response = client.get("/api/v1/demo/sample-images")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 6
    assert len(data["samples"]) == 6

    # Test downloading the first sample
    first_sample_id = data["samples"][0]["id"]
    dl_resp = client.get(f"/api/v1/demo/sample-images/{first_sample_id}")
    assert dl_resp.status_code == 200
    assert dl_resp.headers["content-type"] in ["image/jpeg", "image/png"]




