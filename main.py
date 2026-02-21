import io
import cv2
import numpy as np
import easyocr
from rembg import remove
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response

app = FastAPI(title="Pro Image Tools API")
security = HTTPBearer()

# 🛑 CONFIGURATION: Change this to your actual password!
SECRET_TOKEN = "my_super_secret_hostinger_token_123!"

print("🤖 Loading AI Models into memory... (This takes a few seconds)")
reader = easyocr.Reader(('en',), gpu=False)

# Load 2x Model
sr_x2 = cv2.dnn_superres.DnnSuperResImpl_create()
sr_x2.readModel("EDSR_x2.pb")
sr_x2.setModel("edsr", 2)

# Load 4x Model
sr_x4 = cv2.dnn_superres.DnnSuperResImpl_create()
sr_x4.readModel("EDSR_x4.pb")
sr_x4.setModel("edsr", 4)
print("✅ All AI Models Loaded Successfully!")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_TOKEN:
        print("❌ Unauthorized Access Attempted!")
        raise HTTPException(status_code=401, detail="Invalid Security Token")
    return credentials.credentials

@app.get("/ping")
async def ping():
    print("🏓 Ping endpoint hit by Chrome Extension!")
    return {"status": "success", "message": "API is Live!"}

@app.post("/upscale")
async def upscale_image(image: UploadFile = File(...), scale: int = Form(2), token: str = Depends(verify_token)):
    print(f"🚀 UPSCALING: Request received for {scale}x scaling.")
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if scale == 4:
        print("🚀 UPSCALING: Processing through 4x EDSR model... Please wait.")
        upscaled_img = sr_x4.upsample(img)
    else:
        print("🚀 UPSCALING: Processing through 2x EDSR model... Please wait.")
        upscaled_img = sr_x2.upsample(img)
        
    _, encoded_img = cv2.imencode('.png', upscaled_img)
    print("✅ UPSCALING: Success! Sending PNG back to Chrome.")
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.post("/remove-watermark")
async def remove_watermark(image: UploadFile = File(...), token: str = Depends(verify_token)):
    print("💧 WATERMARK: Request received.")
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    
    # IMREAD_COLOR guarantees 3 channels (Height, Width, Colors)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    print("💧 WATERMARK: Scanning image for text...")
    results = reader.readtext(img)
    
    # THE FIX: Safely unpack the exact height and width to create a flat 2D mask
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
    print("✅ WATERMARK: Success! Sending healed PNG back.")
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.post("/remove-background")
async def remove_background_api(image: UploadFile = File(...), token: str = Depends(verify_token)):
    print("✂️ BACKGROUND: Request received.")
    contents = await image.read()
    print("✂️ BACKGROUND: Running U2-Net AI Model...")
    output_image_bytes = remove(contents) 
    print("✅ BACKGROUND: Success! Sending transparent PNG back.")
    return Response(content=output_image_bytes, media_type="image/png")
