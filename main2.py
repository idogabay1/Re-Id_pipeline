import random
from flask import Flask, request, jsonify
from uuid import uuid4
import os
from werkzeug.utils import secure_filename
from collections import deque
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torchreid
from PIL import Image

# ========== Globals ==========
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Simulated "Database" (for demo; use SQLite or similar in production)

# ------------- databases -------------
person_db = {}
"""{
    0:{ # 0 is the system person id. each person has a unique id witch is the same across all videos.
        "appearances":[
            {
                "video_id":"vid1",
                "track_data":[{"start_point":3,"end_point":7,"track_id":2,"bbox":[10,5,100,102]}, {"start_point":20,"end_point":70,"track_id":7,"bbox":[13,6,105,111]}] #[[entry, exit], [entry, exit]]
            },
            {
                "video_id":"vid2",
                "track_data":[{"start_point":1,"end_point":5,"track_id":7,"bbox":[65,89,123,432]}, {"start_point":60,"end_point":70,"track_id":5,"bbox":[65,75,230,302]}]
            }
            ],
        "anomalies":[
            {
                "frame_number":8,
                "event":"appearance_changed",
                "video_id":"vid1"
            }
            ]
        }
    } """

e_v = {} #{"44dd":{5:deque(maxlen = 20),7,deque(maxlen = 20)}}



# ------------- models -------------
yolo_model = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=50,
    n_init=5,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    nn_budget=100,
    override_track_class=None,
    max_iou_distance=0.5,
)

def load_torchreid_model():
    model = torchreid.models.build_model(
        name='osnet_x1_0', 
        num_classes=1000, 
        pretrained=True,
        use_gpu=True
    )
    torchreid.utils.load_pretrained_weights(model, 'osnet_x1_weights_market1501.pth')
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load image transform function
    transform = torchreid.data.transforms.build_transforms(height=384, width=192, is_train=False)
    return model, transform


reid_model, reid_transform = load_torchreid_model()
reid_model.eval()
reid_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# for cosine similarity - [-1,1] higher is more similar
NEW_PERSON_THRESHOLD = 0.8
APPEARANCE_CHANGE_THRESHOLD = 0.85
EMB_HISTORY_LEN = 20
LEFT_FRAME_THRESHOLD = 5
COMBINE_PEOPLE_THRESHOLD = 0.82



# ========== API Routes ==========
# usage example: curl -X POST http://localhost:8000/videos "file=video.mp4" 

@app.route("/videos", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported video format"}), 415

    video_id = str(uuid4())
    filename = f"{video_id}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # TODO: start your video processing pipeline here
    return jsonify({"video_id": video_id, "status": "uploaded"})

def allowed_file(filename):
    allowed_extensions = {"mp4", "avi", "mov"}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj



@app.route("/persons", methods=["GET"])
def get_all_persons():
    output = []
    for person_id, data in person_db.items():
        appearances = data.get("appearances", [])
        appearances = convert_numpy(appearances)
        output.append({
            "person_id": person_id,
            "appearances": appearances
        })
    return jsonify(output)


@app.route("/persons/<person_id>/anomalies", methods=["GET"])
def get_person_anomalies(person_id):
    if int(person_id) not in person_db:
        return jsonify({"error": "Person not found"}), 404

    anomalies = person_db[int(person_id)].get("anomalies", [])
    anomalies = convert_numpy(anomalies)
    return jsonify({
        "person_id": person_id,
        "anomalies": anomalies
    })

# ---------- video processing functions ----------
next_person_id = [1]  # Use list for mutability


def detect_people_yolo(video_path):
    """
    Runs YOLO on each frame of the video and detects people.
    Returns a list of detections per frame.
    Each detection: [frame_idx, x1, y1, x2, y2, conf]
    """
    detections = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # h, w = frame.shape[:2]
        results = yolo_model.predict(source=frame, classes=[0], conf=0.3, verbose=False)[0]
        results = results.boxes
        boxes = results.xyxy.cpu().numpy()
        for box,res in zip(boxes,results):
            x1, y1, x2, y2 = box#.xyxy[0].cpu().numpy()
            conf = res.conf[0].item()
            detections.append([frame_idx, int(x1), int(y1), int(x2), int(y2), conf])

        frame_idx += 1

    cap.release()
    return detections

def get_detections_for_frame(detections, frame_number):
    """
    Returns only the detections for a specific frame number.
    
    Each detection format: [frame_idx, x1, y1, x2, y2, confidence]
    """
    return [det for det in detections if det[0] == frame_number]


def sanitize_bbox(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width - 1))
    y2 = max(0, min(int(y2), frame_height - 1))
    
    if x2 <= x1 or y2 <= y1:
        return None  # invalid bbox
    return [x1, y1, x2, y2]



def run_deepsort_tracking(video_path, all_detections):
    """
    Apply Deep SORT to track persons using YOLO detections.

    :param video_path: str, path to video file
    :param all_detections: list of detections per frame from YOLO
    :return: list of tracks per frame
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    tracked_results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = get_detections_for_frame(all_detections, frame_idx)
        # Format: [frame_num, x1, y1, x2, y2, confidence], ... ]
        # continue if no detections
        if not detections:
            tracked_results.append([])
            frame_idx += 1
            continue
        frame_height, frame_width = frame.shape[:2]
        dets_for_sort = []

        for det in detections:
            # only keep dets with confidence > 0.3
            if det[5] > 0.3:
                bbox = sanitize_bbox(det[1:5], frame_width, frame_height)
                if bbox is None:
                    print(f"Skipping invalid bbox: {det}")
                    continue
                conf = float(det[5])
                class_name = "person"
                dets_for_sort.append(([*bbox], conf, class_name))
        # Update DeepSort
        actice_tracks = tracker.update_tracks(dets_for_sort, frame=frame)
        frame_tracks = []
        frame_bboxes = []
        for track in actice_tracks:
            if track.original_ltwh is None:
                continue
            else:
                x1_original, y1_original, x2_original, y2_original = track.original_ltwh
                x1 = max(0, min(int(x1_original), frame_width - 1))
                x2 = max(0, min(int(x2_original), frame_width - 1))
                y1 = max(0, min(int(y1_original), frame_height - 1))
                y2 = max(0, min(int(y2_original), frame_height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                if not track.is_confirmed(): 
                    continue
                    
                track_id = track.track_id
                ltrb = track.original_ltwh
                
                frame_tracks.append({
                    "track_id": track_id,
                    "bbox": ltrb
                })
                bbox1 = [int(x1), int(y1), int(x2), int(y2)]
                collision = False
                for bbox2 in frame_bboxes:
                    iou = find_iou(bbox1, bbox2)
                    if iou > 0.3:
                        collision = True
                        break
                    

                frame_bboxes.append(bbox1)
                crop = frame[int(y1):int(y2),int(x1):int(x2), :]
                emb = extract_embedding_torchreid(crop)
                video_id = os.path.basename(video_path)
                frame_number = frame_idx
                update_identity_and_database(emb, video_id, frame_number,track_id,ltrb,collision)

        tracked_results.append(frame_tracks)
        frame_idx += 1

    cap.release()
    return tracked_results


def find_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 < x2 and y1 < y2:
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
    return 0.0

def combine_similar_tracks_by_avg_embedding():
    global person_db, e_v
    for e_v_video_id1, data1 in e_v.items():
        for e_v_video_id2, data2 in e_v.items():
            for e_v_person_id1, emb_list1 in data1.items():
                for e_v_person_id2, emb_list2 in data2.items():
                    if e_v_person_id1 == e_v_person_id2:
                        continue
                    # Compare embeddings
                    emb_1 = np.mean(list(e_v[e_v_video_id1][e_v_person_id1]), axis=0)
                    emb_2 = np.mean(list(e_v[e_v_video_id2][e_v_person_id2]), axis=0)
                    sim = cosine_sim(emb_1, emb_2)
                    if sim > COMBINE_PEOPLE_THRESHOLD:
                        # Combine tracks
                        person_db[e_v_person_id1]["appearances"].extend(person_db[e_v_person_id2]["appearances"])
                        person_db[e_v_person_id1]["anomalies"].extend(person_db[e_v_person_id2]["anomalies"])
                        del person_db[e_v_person_id2]
                        del e_v[e_v_video_id2][e_v_person_id2]
                        # print(f"Combined person {e_v_person_id1} and {e_v_person_id2} with similarity {sim}")
                        return

# -------- embedding vector extraction --------

def extract_embedding_torchreid(crop_img: np.ndarray):
    """
    crop_img: a NumPy image array in BGR format (OpenCV default)
    returns: a 1D NumPy embedding vector
    """
    # Convert to RGB
    rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

    # Apply transform (resizing, normalization)
    rgb_img_pil = Image.fromarray(rgb_img)
    img_tensor = reid_transform[1](rgb_img_pil).unsqueeze(0)  # shape (1, C, H, W)
    img_tensor = img_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

    # Extract features
    with torch.no_grad():
        features = reid_model(img_tensor)
        features = features.cpu().numpy().flatten()

    return features







def cosine_sim(a, b):
    return 1 - cosine(a, b)


def update_identity_and_database(embedding, video_id, frame_number, track_id,bbox_x1y1x2y2,collision):
    global e_v, person_db

    best_similarity = -2
    matched_person_id = -1

    # check if the video_id is in the database
    if video_id not in e_v:
        e_v[video_id] = {}

    # check if the the database is empty    
    if len(e_v) == 0:
        new_id = 0
        e_v[video_id][new_id] = deque([embedding], maxlen=EMB_HISTORY_LEN)
        person_db[new_id] = {
            "appearances": [{
                "video_id": video_id,
                "track_data": [{"start_point":frame_number, "end_point": frame_number, "track_id":track_id,"bbox":bbox_x1y1x2y2}],
            }],
            "anomalies": []
        }
        return new_id, True # return because first appearance
    
    # Check if the person is already in the database in active tracking
    updated = False
    get_out = False
    for person_id in person_db:
        if get_out:
            break
        for appearance in person_db[person_id]["appearances"]:
            if appearance["video_id"] == video_id:
                last_track_id = appearance["track_data"][-1]["track_id"]
                if last_track_id == track_id:
                    last_frame_number = appearance["track_data"][-1]["end_point"]
                    if last_frame_number == frame_number: # person already in the same frame
                        get_out = True
                        break
                    if appearance["track_data"][-1]["end_point"] >= frame_number - LEFT_FRAME_THRESHOLD: # new appearance
                        # Update existing person
                        appearance["track_data"].append({"start_point":frame_number, "end_point": frame_number, "track_id":track_id,"bbox":bbox_x1y1x2y2})
                        updated = True
                        matched_person_id = person_id
                        break
                    else:
                        appearance["track_data"][-1]["end_point"] = frame_number
                        updated = True
                        matched_person_id = person_id
                        break

    # find the most similar person if no active tracking found
    if not updated:
        if video_id not in e_v:
            e_v[video_id] = {}
        else:
            for e_v_person_id, emb_list in e_v[video_id].items():
                recent_embs = list(emb_list)[-20:]  # use last 20 or fewer
                avg_emb = np.mean(recent_embs, axis=0)
                normalized_emb = avg_emb / np.linalg.norm(avg_emb)
                embedding = embedding / np.linalg.norm(embedding)
                sim = cosine_sim(normalized_emb, embedding)
                if sim > best_similarity: # lower similarity means more similar
                    # check if the person is in the same frame and continue if it does
                    current_appearance = person_db[e_v_person_id]["appearances"]
                    if video_id in [appearance["video_id"] for appearance in person_db[person_id]["appearances"]]:
                        for app in current_appearance:
                            if app["video_id"] == video_id:
                                last_frame_number = app["track_data"][-1]["end_point"]
                                if frame_number == last_frame_number:
                                    continue

                    best_similarity = sim
                    matched_person_id = person_id
        
        # check if the person is new to the system
        if NEW_PERSON_THRESHOLD > best_similarity:
            # New person
            new_id = max(person_db.keys(), default=0) + 1
            e_v[video_id][new_id] = deque([embedding], maxlen=EMB_HISTORY_LEN)
            
            person_db[new_id] = {
                "appearances": [{
                    "video_id": video_id,
                    "track_data": [{"start_point":frame_number, "end_point": frame_number, "track_id":track_id,"bbox":bbox_x1y1x2y2}],
                }],
                "anomalies": []
            }
            return new_id, True  # True means new person

        # person exists but not in active tracking
       

    # Get average of history and compare again to current
    avg_emb = np.mean(list(e_v[video_id][matched_person_id])[-20:], axis=0)
    sim = cosine_sim(avg_emb, embedding)
    # find out is there is high IOU of the bbox to another bbox
    if not collision: 
        e_v[video_id][matched_person_id].append(embedding)

    # Appearance change detection
    # has to be: sim > NEW_PERSON_THRESHOLD
    if sim < APPEARANCE_CHANGE_THRESHOLD:
        if "anomalies" not in person_db[matched_person_id]:
            person_db[matched_person_id]["anomalies"] = []
        # Add anomaly event
        person_db[matched_person_id]["anomalies"].append({
            "frame_number": frame_number,
            "event": "appearance_changed",
            "video_id": video_id
        })


    if not updated:
        # Add new timestamp to appearance record
        if "appearances" not in person_db[matched_person_id]:
            person_db[matched_person_id]["appearances"] = []
        appearances = person_db[matched_person_id]["appearances"]
        for system_id in appearances:
            if system_id["video_id"] == video_id:
                last_appearance = system_id["track_data"][-1]
                if last_appearance["track_id"] == track_id:
                    if last_appearance["end_point"] >= frame_number - LEFT_FRAME_THRESHOLD:
                        last_appearance["end_point"] = frame_number
                    else:
                        system_id["track_data"].append({"start_point":frame_number, "end_point":frame_number, "track_id":track_id, "bbox":bbox_x1y1x2y2})                
                else:
                    system_id["track_data"].append({"start_point":frame_number, "end_point":frame_number, "track_id":track_id, "bbox":bbox_x1y1x2y2})
            else: # in case of new video
                appearances.append({
                    "video_id": video_id,
                    "track_data":[{"start_point":frame_number, "end_point":frame_number, "track_id":track_id, "bbox":bbox_x1y1x2y2}]
                })
    
    return matched_person_id, False  # False means existing person


    # return matched_person_id, False  # False means existing person


# go through all person_db and add event "entered_to_frame" and "exited_from_frame"
def add_enter_exit_events():
    global person_db
    for person_id, data in person_db.items():
        appearances = data.get("appearances", [])
        for appearance in appearances:
            # video_id = appearance["video_id"]
            len_track_data = len(appearance["track_data"])
            for i,track_data in enumerate(appearance["track_data"]):
                frame_number = track_data["start_point"], track_data["end_point"]
                
                if i < len_track_data - 1:
                    next_frame_number = appearance["track_data"][i+1]["start_point"], appearance["track_data"][i]["end_point"]
                    entry = next_frame_number[0]
                    exit_point = frame_number[1]
                    if entry - exit_point > LEFT_FRAME_THRESHOLD:
                        if "anomalies" not in person_db[person_id]:
                            person_db[person_id]["anomalies"] = []
                        person_db[person_id]["anomalies"].append({
                            "frame_number": exit_point,
                            "event": "exited_from_frame",
                            "video_id": appearance["video_id"]
                        })
                        person_db[person_id]["anomalies"].append({
                            "frame_number": entry,
                            "event": "entered_to_frame",
                            "video_id": appearance["video_id"]
                        })
            # add end_point to the last entry
            if len(appearance["track_data"]) > 0:
                try:
                    last_entry = appearance["track_data"][-1]
                except:
                    print(f"Error: {track_data}")
                if "anomalies" not in person_db[person_id]:
                    person_db[person_id]["anomalies"] = []
                person_db[person_id]["anomalies"].append({
                    "frame_number": last_entry["end_point"],
                    "event": "exited_from_frame",
                    "video_id": appearance["video_id"]
                })
        
                
            
                



@app.route("/process_video", methods=["POST"])
def main():
    video_id = request.form.get("video_id")
    if not video_id:
        return jsonify({"error": "Missing video_id"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    # Run your processing pipeline here
    all_detections = detect_people_yolo(video_path)
    tracked_results = run_deepsort_tracking(video_path, all_detections)
    add_enter_exit_events()
    combine_similar_tracks_by_avg_embedding()
    print_tracks_ontop_video(video_path)
    return jsonify({"status": "processing complete"})#, "tracked_results": tracked_results})



def print_tracks_ontop_video(video_path):
    global person_db
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    output_path = os.path.join(UPLOAD_FOLDER, f"tracked_{os.path.basename(video_path)}")
    first_frame = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame:
            first_frame = False
            # Get the width and height of the frame
            frame_height, frame_width = frame.shape[:2]
            # Create a VideoWriter object to save the output video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
        # Draw tracked results
        for person_id, data in person_db.items():
            for appearance in data["appearances"]:
                if appearance["video_id"] == os.path.basename(video_path):
                    for track_data in appearance["track_data"]:
                        if track_data["start_point"] <= frame_idx <= track_data["end_point"]:
                            # Draw bounding box
                            if "bbox" in track_data:
                                bbox = track_data["bbox"]
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = bbox
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    cv2.putText(frame, f"ID: {person_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    # return                                  
                            else:
                                # Fallback to using the track_id
                                if "track_id" in track_data:
                                    person_id = track_data["track_id"]
                                    if "bbox" in track_data:
                                        bbox = track_data["bbox"]
                                        if len(bbox) == 4:
                                            x1, y1, x2, y2 = bbox
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Save the video
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

    # Save the video with tracked results

    
    



# ========== Entry Point ==========
if __name__ == "__main__":
    use_service = True
    if use_service:
        app.run(host="0.0.0.0", port=8000, debug=True)
        # main()
    else:
        video_path = "videos/vid1.mp4"
        # simulate_process_video(video_id="vid1")
        # simulate_process_video(video_id="vid2")

        run_deepsort_tracking(video_path,detect_people_yolo(video_path))
        add_enter_exit_events()

