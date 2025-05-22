# Re-Id_pipeline

A lightweight Python-based system for detecting, tracking, and re-identifying people across videos. It combines object detection, multi-object tracking, and person re-identification to:

- Track people in videos using YOLO and DeepSORT
- Match identities across different video files
- Detect appearance changes (e.g. picked up or dropped an object)

🧠 Ideal for surveillance, anomaly detection, or multi-camera monitoring use cases.

## 🔧 How It Works

1. **Detection**: Uses YOLOv8n to detect people in video frames.
2. **Tracking**: Uses DeepSORT to assign consistent IDs to each person throughout the video.
3. **Re-Identification**: Extracts appearance embeddings using Torchreid with the `osnet_x1_0` model and compares them via cosine similarity to identify individuals across videos.
4. **Anomaly Detection**: Tracks appearance changes of individuals by comparing historical embedding vectors.

## 🛠️ Models & Hyperparameters

- 🔍 Detection: `yolov8n.pt` (Ultralytics YOLOv8)
- 🧭 Tracker: DeepSORT with:
  - `max_age=50`, `n_init=5`, `max_cosine_distance=0.2`, `nn_budget=100`, `max_iou_distance=0.5`
- 👤 Re-ID Model: Torchreid `osnet_x1_0` with pretrained weights (`osnet_x1_weights_market1501.pth`)

> Note: You may need to tune thresholds and tracker settings depending on your specific video quality, camera angles, and scene complexity.

## 🚀 API Endpoints

- `POST /videos`: Upload a new video for processing.
- `POST /process_video`: Run the detection + tracking + re-identification pipeline.
- `GET /persons`: Retrieve all tracked person IDs.
- `GET /persons/<person_id>/anomalies`: View detected appearance changes for a specific person.

## 📁 Project Structure

- YOLO + DeepSORT for detection and tracking
- Torchreid for appearance embedding
- Flask backend to serve the pipeline and data
- In-memory database (`person_db`, `embedding_dict`) for storing re-id history and anomalies
