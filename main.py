from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

# from fastapi import Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

import io
import cv2
import numpy as np
import insightface
from insightface.model_zoo import get_model

import os

app = FastAPI()

model_path = "models/inswapper_128.onnx"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the file exists.")


model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)
swapper = get_model("models/inswapper_128.onnx", providers=["CPUExecutionProvider"], download=False)

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
