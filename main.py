import io
import cv2
import numpy as np
import easyocr
import gc  # 🟢 NEW: Python's Garbage Collector
from rembg import remove, new_session  # 🟢 NEW: Imported new_session to manage RAM
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response

app = FastAPI(title="Pro Image Tools API")
security = HTTPBearer()

# 🛑 CONFIGURATION: Change this to your actual password!
SECRET_TOKEN = "my_super_secret_hostinger_token_123!"

print("🟢 Server started in LOW-RAM (Lazy-Loading) Mode! AI will sleep until needed.")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_TOKEN:
        print("❌ Unauthorized Access Attempted!")
        raise HTTPException(status_code=401, detail="Invalid Security Token")
    return credentials.credentials

@app.get("/ping")
async def ping():
    print("🏓 Ping endpoint hit by Chrome Extension!")
    return {"status": "success", "message": "API is Live in Low-RAM Mode!"}

@app.post("/upscale")
async def upscale_image(image: UploadFile = File(...), scale: int = Form(2), token: str = Depends(verify_token)):
    print(f"🚀 UPSCALING: Request received for {scale}x scaling.")
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # WAKE UP THE UPSCALER AI
    print("🚀 UPSCALING: Waking up OpenCV AI from hard drive into RAM...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    
    if scale == 4:
        sr.readModel("EDSR_x4.pb")
        sr.setModel("edsr", 4)
    else:
        sr.readModel("EDSR_x2.pb")
        sr.setModel("edsr", 2)
        
    print(f"🚀 UPSCALING: Processing through {scale}x EDSR model... Please wait.")
    upscaled_img = sr.upsample(img)
    _, encoded_img = cv2.imencode('.png', upscaled_img)
    
    # 🟢 KILL THE AI AND FREE THE RAM
    print("🧹 UPSCALING: Destroying AI from RAM to free memory...")
    del sr
    gc.collect()
    
    print("✅ UPSCALING: Success! Sending PNG back to Chrome.")
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.post("/remove-watermark")
async def remove_watermark(image: UploadFile = File(...), token: str = Depends(verify_token)):
    print("💧 WATERMARK: Request received.")
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # WAKE UP EASYOCR AI
    print("💧 WATERMARK: Waking up EasyOCR AI into RAM...")
    reader = easyocr.Reader(('en',), gpu=False)
    
    print("💧 WATERMARK: Scanning image for text...")
    results = reader.readtext(img)
    
    h, w, c = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    print(f"💧 WATERMARK: Found {len(results)} text blocks. Generating mask...")
    for (bbox, text, prob) in results:
        tl, tr, br, bl = bbox
        x_tl, y_tl = tl
        x_br, y_br = br
        
        pt1 = (int(x_tl), int(y_tl))
        pt2 = (int(x_br), int(y_br))
        cv2.rectangle(mask, pt1, pt2, 255, thickness=-1)

    print("💧 WATERMARK: Running AI Inpainting to erase pixels...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    _, encoded_img = cv2.imencode('.png', inpainted_img)
    
    # 🟢 KILL THE AI AND FREE THE RAM
    print("🧹 WATERMARK: Destroying AI from RAM to free memory...")
    del reader
    gc.collect()
    
    print("✅ WATERMARK: Success! Sending healed PNG back.")
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.post("/remove-background")
async def remove_background_api(image: UploadFile = File(...), token: str = Depends(verify_token)):
    print("✂️ BACKGROUND: Request received.")
    contents = await image.read()
    
    # WAKE UP U2-NET AI
    print("✂️ BACKGROUND: Waking up U2-Net AI into RAM...")
    session = new_session("u2net")
    
    print("✂️ BACKGROUND: Running U2-Net AI Model...")
    output_image_bytes = remove(contents, session=session) 
    
    # 🟢 KILL THE AI AND FREE THE RAM
    print("🧹 BACKGROUND: Destroying AI from RAM to free memory...")
    del session
    gc.collect()
    
    print("✅ BACKGROUND: Success! Sending transparent PNG back.")
    return Response(content=output_image_bytes, media_type="image/png")
