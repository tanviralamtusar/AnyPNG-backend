import io
import cv2
import numpy as np
import easyocr
from rembg import remove
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response

app = FastAPI(title="Pro Image Tools API")
security = HTTPBearer()

# 🛑 CONFIGURATION: Change this to a secure password!
SECRET_TOKEN = "my_super_secret_hostinger_token_123!"

# Initialize AI Models (Loaded on startup)
# FIXED: Added
reader = easyocr.Reader(, gpu=False)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x2.pb")
sr.setModel("edsr", 2)

# 🔒 SECURITY MIDDLEWARE
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid Security Token")
    return credentials.credentials

@app.get("/ping")
async def ping():
    return {"status": "success", "message": "API is Live!"}

# FIXED: Added
@app.post("/upscale", dependencies=)
async def upscale_image(image: UploadFile = File(...)):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    upscaled_img = sr.upsample(img)
    _, encoded_img = cv2.imencode('.png', upscaled_img)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

# FIXED: Added
@app.post("/remove-watermark", dependencies=)
async def remove_watermark(image: UploadFile = File(...)):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = reader.readtext(img)
    mask = np.zeros(img.shape, dtype=np.uint8)

    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl), int(tl))
        br = (int(br), int(br))
        cv2.rectangle(mask, tl, br, 255, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    _, encoded_img = cv2.imencode('.png', inpainted_img)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

# FIXED: Added
@app.post("/remove-background", dependencies=)
async def remove_background_api(image: UploadFile = File(...)):
    contents = await image.read()
    output_image_bytes = remove(contents) 
    return Response(content=output_image_bytes, media_type="image/png")
