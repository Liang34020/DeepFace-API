from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

# from fastapi import Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware



import io
import cv2
import numpy as np
import insightface
from insightface.model_zoo import get_model

import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定 ["http://localhost:5500"]（本地開發）、正式網站網址
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)
import requests

MODEL_URL = "https://www.dropbox.com/scl/fi/68a9l8y9pe1f2iwwgi24q/inswapper_128.onnx?rlkey=598smki7wbygn1ukigocfxc10&st=3hev5ics&dl=1"
MODEL_PATH = "Models/inswapper_128.onnx"

# 自動建立 models 資料夾
os.makedirs("Models", exist_ok=True)

# 若模型不存在就下載
if not os.path.exists(MODEL_PATH):
    print("🔽 模型不存在，從 Dropbox 下載中...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ 模型下載完成")
    else:
        raise RuntimeError(f"❌ 無法下載模型：HTTP {response.status_code}")

model_path = "Models/inswapper_128.onnx"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the file exists.")


model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=-1)
swapper = get_model("Models/inswapper_128.onnx", providers=["CPUExecutionProvider"], download=False)

# # 前端資源
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-images/")
async def process_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_bytes = await file1.read()
    img2_bytes = await file2.read()
    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    faces1 = model.get(img1)
    faces2 = model.get(img2)
    if len(faces1) == 0 or len(faces2) == 0:
        return {"error": "找不到臉部，請使用正面臉照"}
    
    swapped = swapper.get(img1, faces1[0], faces2[0], paste_back=True)
    _, buffer = cv2.imencode('.png', swapped)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
