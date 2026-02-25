import io
import cv2
import numpy as np
import gc  # 🟢 Python's Garbage Collector
from rembg import remove, new_session  
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response

# 🟢 NEW: Import Google GenAI and PIL
from google import genai
from PIL import Image

app = FastAPI(title="AnyPNG API")
security = HTTPBearer()

# 🛑 CONFIGURATION
SECRET_TOKEN = "my_super_secret_hostinger_token_123!"
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # 👈 PASTE YOUR GOOGLE API KEY HERE

# Initialize Google Client
if GOOGLE_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    google_client = genai.Client(api_key=GOOGLE_API_KEY)

print("🟢 Server started in LOW-RAM Mode! Watermarks routed to Gemini Pro.")

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
    
    print("🧹 UPSCALING: Destroying AI from RAM to free memory...")
    del sr
    gc.collect()
    
    print("✅ UPSCALING: Success! Sending PNG back to Chrome.")
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.post("/remove-watermark")
async def remove_watermark(
    image: UploadFile = File(...), 
    prompt: str = Form("Remove all watermarks, text, and translucent lines like 'pngtree' or 'VectorStock'. Keep the logos, subject, and background completely untouched and perfectly preserved."),
    token: str = Depends(verify_token)
):
    print("💧 WATERMARK: Request received. Sending to Google Gemini Pro...")
    
    # 1. Read the uploaded image
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents))
    
    try:
        # 2. Ask Gemini to edit the image using Natural Language
        result = google_client.models.generate_images(
            model='gemini-3.1-pro-preview', # Use the model name from your AI Studio
            prompt=prompt,
            image=pil_img,
            output_format="png"
        )
        
        # 3. Get the edited image bytes back from Google
        output_bytes = result.generated_images[0].image.image_bytes
        
        print("✅ WATERMARK: Success! Google flawlessly removed the watermark.")
        return Response(content=output_bytes, media_type="image/png")
        
    except Exception as e:
        print(f"❌ WATERMARK ERROR: Google API Failed - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google API Error: {str(e)}")

@app.post("/remove-background")
async def remove_background_api(image: UploadFile = File(...), token: str = Depends(verify_token)):
    print("✂️ BACKGROUND: Request received.")
    contents = await image.read()
    
    print("✂️ BACKGROUND: Waking up U2-Net AI into RAM...")
    session = new_session("u2net")
    
    print("✂️ BACKGROUND: Running U2-Net AI Model...")
    output_image_bytes = remove(contents, session=session) 
    
    print("🧹 BACKGROUND: Destroying AI from RAM to free memory...")
    del session
    gc.collect()
    
    print("✅ BACKGROUND: Success! Sending transparent PNG back.")
    return Response(content=output_image_bytes, media_type="image/png")
