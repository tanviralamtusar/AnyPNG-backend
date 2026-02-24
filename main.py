import io
import cv2
import numpy as np
import gc  # 🟢 Python's Garbage Collector
from rembg import remove, new_session  
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response

app = FastAPI(title="AnyPNG API")
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

    # 1. NEW AUTO-MASKER (Finds thin watermark lines, ignores thick logos)
    print("💧 WATERMARK: Auto-detecting faint watermark lines...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Black-hat transform: Extracts dark/light thin lines on varying backgrounds
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    
    # Isolate the faint lines
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Expand the mask slightly so the AI covers the edges perfectly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 2. WAKE UP LAMA GENERATIVE AI (The Magic Part)
    print("💧 WATERMARK: Waking up LaMa Generative AI...")
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    
    lama = SimpleLama()
    
    # Convert formats for LaMa
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask).convert('L')
    
    print("💧 WATERMARK: AI is recreating the missing background...")
    result_img = lama(pil_img, pil_mask)
    
    # 🟢 KILL THE AI AND FREE THE RAM
    print("🧹 WATERMARK: Destroying AI from RAM to free memory...")
    del lama
    gc.collect()

    # Convert the finished PIL image back to bytes for download
    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    
    print("✅ WATERMARK: Success! Sending perfectly healed PNG back.")
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

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
