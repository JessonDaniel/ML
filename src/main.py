'''from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os, cv2, base64, logging, shutil
from ultralytics import YOLO
import numpy as np
import google.generativeai as genai
from collections import Counter
import asyncio, time, torch, colorsys, tempfile, sys
import pandas as pd
import torch.backends.cudnn as cudnn
from PIL import Image

# Add the parent directory of 'deep_sort' to sys.path
sys.path.append((os.path.join(os.path.dirname(__file__), '..')))

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker
deep_sort_weights = '../deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=80)

genai.configure(api_key="key")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)

# Create FastAPI instance
app = FastAPI()

# Enable CORS (Adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection details
MONGO_DETAILS = "mongodb+srv://jessondanielj:adTpKhQfdpLX3EuC@cluester1.4wkmq.mongodb.net/?retryWrites=true&w=majority&appName=Cluester1"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client['LLM']  # Database name 'Cluster1'
videos_collection = database['test']  # Collection name 'videos'
# Temporary collection for storing inferences
inferences_collection = database['temp_inferences']  # Collection name 'temp_inferences'

# Directory to save uploaded files
UPLOAD_DIRECTORY = "./uploaded_videos"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# YOLO model initialization (used for both file upload and stream processing)
model = YOLO('yolov8n.pt')

# Global variables for tracking persons, running, weapons, and obstruction
unique_track_ids = set()
track_labels = {}
track_times = {}
track_positions = {}
running_counters = {}
obstruction_frames = 0
obstruction_blocked = False
fps = 30  # Assuming 30 FPS for camera feed
running_threshold = 5  # Define your threshold for running detection (pixels per second)
alert_person_ids = []
inference_store = []
current_person_count = 0
current_running_status = "No Running Detected"
current_weapon_status = "No Weapon Detected"
current_camera_view_status = "Unobstructed"

# Pydantic model to define the video data structure
class VideoData(BaseModel):
    filename: str
    analysis: str

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define function to clear temporary inferences (optional)
def clear_temp_inferences():
    global inference_store
    inference_store.clear()

def process_video(file_path: str) -> str:
    try:
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        class_counter = Counter()
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    class_counter[class_name] += 1
                    detections.append({
                        "frame": frame_count,
                        "class_name": class_name,
                        "confidence": float(box.conf[0])
                    })

            frame_count += 1

        cap.release()

        unique_classes = list(class_counter.keys())
        person_detected = 'person' in class_counter
        weapon_detected = 'weapon' in class_counter  # Change 'weapons' to 'weapon'

        # Trigger alert if a weapon is detected
        if weapon_detected:
            main_theme = "A weapon has been detected in the video!"
        elif person_detected:
            main_theme = "This video predominantly features people."
        else:
            main_theme = "No significant detection of persons or weapons."

        analysis_summary = f"Video Analysis Report:<br>" \
                           f"Detected Classes: {', '.join(unique_classes)}.<br>"\
                           f"Main Theme: {main_theme}<br>" 

        return analysis_summary, detections

    except Exception as e:
     return f"An error occurred: {str(e)}", []


# Endpoint to handle video file uploads
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        analysis_result, detections = process_video(file_path)

        video_data = {
            "filename": file.filename,
            "analysis": analysis_result,
            "detections": detections
        }
        await videos_collection.insert_one(video_data)

        return JSONResponse(status_code=200, content={"result": analysis_result})

    except Exception as e:
        logging.error(f"Error in upload-video: {e}")
        return JSONResponse(status_code=500, content={"error": "Error processing video."})

# Endpoint to retrieve all saved video data
@app.get("/videos")
async def get_videos():
    try:
        videos = await videos_collection.find().to_list(1000)  # Adjust the limit as needed
        for video in videos:
            video['_id'] = str(video['_id'])  # Convert ObjectId to string
        return JSONResponse(status_code=200, content={"videos": videos})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def resize_frame_to_16_9(frame, target_width=1920, target_height=1080):
    # Resize the frame to fit the 16:9 aspect ratio
    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return frame
        
# Optional endpoint to clear temporary inferences
@app.delete("/clear-inferences")
async def clear_inferences():
    try:
        await inferences_collection.delete_many({})  # Clear the collection
        return JSONResponse(status_code=200, content={"message": "Temporary inferences cleared."})
    except Exception as e:
        logging.error(f"Error clearing inferences: {e}")
        return JSONResponse(status_code=500, content={"error": "Error clearing inferences."})'''
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os, cv2, base64, logging, shutil
from ultralytics import YOLO
from typing import Dict, List
import json
import numpy as np
import google.generativeai as genai
from collections import Counter
import asyncio, time, torch, colorsys, tempfile, sys
import pandas as pd
import torch.backends.cudnn as cudnn
from PIL import Image

# Configure the Gemini API
genai.configure(api_key="AIzaSyBCk1rpZVqHZSmR9MGdtBpZV2uPQthAf3I")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Create FastAPI instance
app = FastAPI()

# Enable CORS (Adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection details
MONGO_DETAILS = "mongodb+srv://jessondanielj:adTpKhQfdpLX3EuC@cluester1.4wkmq.mongodb.net/?retryWrites=true&w=majority&appName=Cluester1"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client['LLM']
videos_collection = database['test']
inferences_collection = database['temp_inferences']

# Directory to save uploaded files
UPLOAD_DIRECTORY = "./uploaded_videos"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# YOLO model initialization
yolo_model = YOLO('yolov10m.pt')

# Pydantic model to define the video data structure
class VideoData(BaseModel):
    filename: str
    analysis: str
    detections: list

# Function to process video and return analysis summary
'''def process_video(file_path: str):
    try:
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        class_counter = Counter()
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    class_counter[class_name] += 1
                    detections.append({
                        "frame": frame_count,
                        "class_name": class_name,
                        "confidence": float(box.conf[0])
                    })

            frame_count += 1

        cap.release()

        unique_classes = list(class_counter.keys())
        person_detected = 'person' in class_counter
    
        weapon_detected = 'weapon' in class_counter

        if weapon_detected:
            main_theme = "A weapon has been detected in the video!"
        elif person_detected:
            main_theme = "This video predominantly features people."
        else:
            main_theme = "No significant detection of persons or weapons."

        analysis_summary = f"Video Analysis Report:" \
                           f"Detected Classes: {', '.join(unique_classes)}."\
                           f"Main Theme: {main_theme}" 

        return analysis_summary, detections

    except Exception as e:
        return f"An error occurred: {str(e)}", []
'''
async def process_video(file_path: str):
    try:
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        class_counter = Counter()
        detections = []
        batch_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for faster processing (optional)
            frame_resized = cv2.resize(frame, (640, 360))  # Adjust size as needed
            batch_frames.append(frame_resized)

            # Process in batches
            if len(batch_frames) >= 10:  # Process every 10 frames
                results = yolo_model(batch_frames)
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        class_counter[class_name] += 1
                        detections.append({
                            "frame": frame_count,
                            "class_name": class_name,
                            "confidence": float(box.conf[0])
                        })
                batch_frames.clear()  # Clear the batch

            frame_count += 1

        cap.release()

        unique_classes = list(class_counter.keys())
        main_theme = "This video features the following detected classes: " + ', '.join(unique_classes)

        analysis_summary = f"Video Analysis Report:<br>" \
                           f"Detected Classes: {', '.join(unique_classes)}.<br>"\
                           f"Main Theme: {main_theme}<br>"

        return analysis_summary, detections

    except Exception as e:
        return f"An error occurred: {str(e)}", []

# Endpoint to handle video file uploads
@app.post("/upload-video")
async def upload_video(file: UploadFile):
    try:
        # Save the uploaded file to a temporary location
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the video asynchronously
        analysis_result, detections = await process_video(file_path)

        return {
            "analysis_result": analysis_result,
            "detections": detections
        }
    except Exception as e:
        print(f"Error in upload-video: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Endpoint to handle user queries
@app.post("/ask-question")
async def ask_question(question: str):
    try:
        latest_video = await videos_collection.find_one(sort=[("_id", -1)])
        if not latest_video:
            return JSONResponse(status_code=404, content={"error": "No video data available."})

        detected_classes = ', '.join([d['class_name'] for d in latest_video['detections']])
        context = f"The video contains the following detected classes: {detected_classes}."

        # Generate a response from the Gemini API
        response = gemini_model.generate_content(f"{context} Now, {question}")
        return JSONResponse(status_code=200, content={"response": response.text})

    except Exception as e:
        logging.error(f"Error in ask-question: {e}")
        return JSONResponse(status_code=500, content={"error": "Error handling question."})

# Endpoint to retrieve all saved video data
@app.get("/videos")
async def get_videos():
    try:
        videos = await videos_collection.find().to_list(1000)
        for video in videos:
            video['_id'] = str(video['_id'])
        return JSONResponse(status_code=200, content={"videos": videos})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.chat_histories: Dict[str, List[Dict]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = []

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    def add_to_history(self, client_id: str, message: dict):
        if client_id in self.chat_histories:
            self.chat_histories[client_id].append(message)

manager = ConnectionManager()

# Add these new endpoints to your FastAPI app
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            # Get the latest video context
            latest_video = await videos_collection.find_one(sort=[("_id", -1)])
            detected_classes = ', '.join([d['class_name'] for d in latest_video['detections']]) if latest_video else ""
            context = f"The video contains the following detected classes: {detected_classes}."
            
            # Generate response using Gemini
            response = await generate_gemini_response(context, data_json['question'])
            
            # Save to chat history
            chat_entry = {
                "question": data_json['question'],
                "response": response,
                "timestamp": time.time()
            }
            manager.add_to_history(client_id, chat_entry)
            
            # Send response back to client
            await manager.send_message(
                json.dumps({"response": response, "history": manager.chat_histories[client_id]}),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def generate_gemini_response(context: str, question: str) -> str:
    try:
        response = gemini_model.generate_content(f"{context} Question: {question}")
        return response.text
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        return f"Error generating response: {str(e)}"

# Add this endpoint to get chat history
@app.get("/chat-history/{client_id}")
async def get_chat_history(client_id: str):
    if client_id in manager.chat_histories:
        return JSONResponse(status_code=200, content={"history": manager.chat_histories[client_id]})
    return JSONResponse(status_code=404, content={"error": "No chat history found"})