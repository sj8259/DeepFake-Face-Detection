import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from mtcnn import MTCNN
import os
import tempfile
import logging

app = Flask(__name__)
CORS(app)

# Configuration
app.config['DEBUG'] = True
logging.basicConfig(level=logging.DEBUG)

FRAME_SIZE = (224, 224)
SEQ_LENGTH = 30
MIN_FACE_PIXELS = 50  # Minimum face size in pixels
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Load models with enhanced error handling
try:
    model = tf.keras.models.load_model("op.h5")
    yolo_net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    mtcnn = MTCNN()  # No unsupported parameters
    logging.info("All models loaded successfully")
except Exception as e:
    logging.critical(f"Model initialization failed: {str(e)}")
    raise RuntimeError(f"Model loading error: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(frame):
    """Hybrid YOLOv3 + MTCNN face detection with fallback"""
    try:
        height, width = frame.shape[:2]
        faces = []

        # YOLOv3 Person Detection
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())

        person_boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.4:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x1 = int(centerX - (w / 2))
                    y1 = int(centerY - (h / 2))
                    x2, y2 = x1 + w, y1 + h
                    person_boxes.append((x1, y1, x2, y2))

        # Fallback to full-frame MTCNN
        if not person_boxes:
            mtcnn_results = mtcnn.detect_faces(frame)
            for res in mtcnn_results:
                if res['confidence'] > 0.85:
                    x, y, w, h = res['box']
                    faces.append((x, y, x + w, y + h))
        else:
            for (x1, y1, x2, y2) in person_boxes:
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue
                mtcnn_results = mtcnn.detect_faces(person_roi)
                for res in mtcnn_results:
                    if res['confidence'] > 0.85:
                        fx, fy, fw, fh = res['box']
                        faces.append((
                            x1 + fx,
                            y1 + fy,
                            x1 + fx + fw,
                            y1 + fy + fh
                        ))

        # Manual size filtering
        valid_faces = [
            (x, y, x2, y2) for (x, y, x2, y2) in faces
            if (x2 - x) >= MIN_FACE_PIXELS and 
               (y2 - y) >= MIN_FACE_PIXELS and 
               x2 <= width and y2 <= height
        ]
        
        return valid_faces

    except Exception as e:
        logging.error(f"Face detection error: {str(e)}")
        return []

def process_video(video_path):
    """Enhanced video processing pipeline"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, SEQ_LENGTH, dtype=int)
        
        valid_frames_count = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Skipping invalid frame at index {idx}")
                continue

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detect_faces(frame_rgb)
                
                if faces:
                    # Extract largest face
                    x1, y1, x2, y2 = max(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
                    face_region = frame_rgb[max(0,y1):min(frame.shape[0],y2),
                                            max(0,x1):min(frame.shape[1],x2)]
                    
                    if face_region.size > 0:
                        resized_face = cv2.resize(face_region, FRAME_SIZE)
                        preprocessed_face = tf.keras.applications.xception.preprocess_input(resized_face)
                        frames.append(preprocessed_face)
                        valid_frames_count += 1

            except Exception as e:
                logging.error(f"Frame {idx} processing error: {str(e)}")
                continue

        if valid_frames_count < 5:  # Require at least 5 valid faces
            raise RuntimeError(f"Only {valid_frames_count} valid faces detected")

        # Pad sequence
        while len(frames) < SEQ_LENGTH:
            frames.append(np.zeros((*FRAME_SIZE, 3), dtype=np.float32))

        return np.array(frames[:SEQ_LENGTH])

    except Exception as e:
        logging.error(f"Video processing failed: {str(e)}")
        raise
    finally:
        cap.release()

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Analysis endpoint with complete error handling"""
    if 'video' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No video file provided",
            "result": None,
            "confidence": 0.0
        }), 400

    try:
        video_file = request.files['video']
        if not allowed_file(video_file.filename):
            return jsonify({
                "status": "error",
                "message": "Invalid file type",
                "result": None,
                "confidence": 0.0
            }), 400

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            logging.info(f"Processing video: {tmp.name}")
            
            faces_array = process_video(tmp.name)
            
            if faces_array.shape != (SEQ_LENGTH, FRAME_SIZE[0], FRAME_SIZE[1], 3):
                raise ValueError("Invalid processed frames shape")

            # Make predictions
            predictions = [model.predict(frame[np.newaxis,...], verbose=0)[0][0] 
                          for frame in faces_array]
            avg_prediction = np.mean(predictions)
            
            # Confidence calibration
            calibrated_conf = 1 / (1 + np.exp(-(avg_prediction - 0.7)/0.15))
            threshold = 0.85  # Stricter threshold for real videos
            
            return jsonify({
                "status": "success",
                "message": "Analysis complete",
                "result": "Real" if calibrated_conf > threshold else "Fake",
                "confidence": float(np.clip(calibrated_conf, 0, 1)),
                "metrics": {
                    "processed_frames": len(predictions),
                    "average_confidence": float(avg_prediction)
                }
            })

    except RuntimeError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "result": None,
            "confidence": 0.0
        }), 400
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "result": None,
            "confidence": 0.0
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
