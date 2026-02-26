import io
import os
import cv2
import numpy as np
import gc  # 🟢 Python's Garbage Collector
from rembg import remove, new_session  
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response

# Google GenAI & PIL
from google import genai
from PIL import Image
from google.genai import types

# Supabase & Dotenv
from supabase import create_client, Client
from dotenv import load_dotenv

# 🟢 NEW: Load secret variables securely
load_dotenv()

app = FastAPI(title="AnyPNG SaaS API")
security = HTTPBearer()

# 🛑 CONFIGURATION (Securely pulled from Coolify Environment Variables!)
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my_super_secret_hostinger_token_123!") # Fallback for old tools
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Clients
if GOOGLE_API_KEY:
    google_client = genai.Client(api_key=GOOGLE_API_KEY)
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🟢 SaaS Server started! Watermarks routed to Gemini Pro. Upscaler/BG running locally.")

# --- AUTH 1: For Free Local Tools (Upscale & BG Remove) ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_TOKEN:
        print("❌ Unauthorized Access Attempted (Basic Token)!")
        raise HTTPException(status_code=401, detail="Invalid Security Token")
    return credentials.credentials

# --- AUTH 2: For SaaS Pro Tools (Watermark via Supabase Login) ---
async def verify_supabase_user(authorization: str = Header(...)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing login token")
    
    token = authorization.split(" ")[1]
    try:
        # Ask Supabase who this token belongs to
        user_res = supabase.auth.get_user(token)
        if not user_res.user: raise Exception()
        return user_res.user.id
    except:
        raise HTTPException(status_code=401, detail="Invalid session. Please login via the extension.")

# --- ENDPOINTS ---

@app.get("/ping")
async def ping():
    print("🏓 Ping endpoint hit!")
    return {"status": "success", "message": "API is Live!"}

@app.post("/upscale")
async def upscale_image(image: UploadFile = File(...), scale: int = Form(2), token: str = Depends(verify_token)):
    print(f"🚀 UPSCALING: Request received for {scale}x scaling.")
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    if scale == 4:
        sr.readModel("EDSR_x4.pb")
        sr.setModel("edsr", 4)
    else:
        sr.readModel("EDSR_x2.pb")
        sr.setModel("edsr", 2)
        
    upscaled_img = sr.upsample(img)
    _, encoded_img = cv2.imencode('.png', upscaled_img)
    
    del sr
    gc.collect()
    print("✅ UPSCALING: Success!")
    return Response(content=encoded_img.tobytes(), media_type="image/png")

@app.post("/remove-background")
async def remove_background_api(image: UploadFile = File(...), token: str = Depends(verify_token)):
    print("✂️ BACKGROUND: Request received.")
    contents = await image.read()
    
    session = new_session("u2net")
    output_image_bytes = remove(contents, session=session) 
    
    del session
    gc.collect()
    print("✅ BACKGROUND: Success!")
    return Response(content=output_image_bytes, media_type="image/png")

@app.post("/remove-watermark")
async def remove_watermark(
    image: UploadFile = File(...), 
    prompt: str = Form(...),
    user_id: str = Depends(verify_supabase_user) 
):
    print(f"💎 PRO AI: Watermark request from User {user_id}. Prompt: {prompt}")
    
    # 1. CHECK & DEDUCT CREDITS IN SUPABASE
    profile = supabase.table("profiles").select("credits").eq("id", user_id).execute()
    if not profile.data or profile.data[0]['credits'] <= 0:
        raise HTTPException(status_code=402, detail="Out of credits! Please buy more.")
    
    current_credits = profile.data[0]['credits']
    
    # Deduct 1 credit
    supabase.table("profiles").update({"credits": current_credits - 1}).eq("id", user_id).execute()
    print(f"💎 PRO AI: 1 Credit deducted. {current_credits - 1} remaining.")

    # 2. RUN GOOGLE GEMINI
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents))
    
    try:
        # 🟢 FIXED: Use generate_content and pass both the image and prompt in the contents list!
        result = google_client.models.generate_content(
            model='gemini-3.1-pro-preview', 
            contents=[pil_img, prompt]
        )
        
        # 3. Safely extract the generated image bytes from Gemini's response
        output_bytes = None
        if result.candidates and result.candidates[0].content.parts:
            for part in result.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    output_bytes = part.inline_data.data
                    break
        
        # If Gemini gets confused and sends text back instead of an image, catch it!
        if not output_bytes:
            raise Exception("Google returned a text response instead of an image.")
            
        print("✅ PRO AI: Success! Google flawlessly removed the watermark.")
        return Response(content=output_bytes, media_type="image/png")
        
    except Exception as e:
        print(f"❌ PRO AI ERROR: {str(e)}")
        # 🟢 Refund the credit so the user isn't cheated out of their money!
        supabase.table("profiles").update({"credits": current_credits}).eq("id", user_id).execute()
        raise HTTPException(status_code=500, detail="AI generation failed. Credit safely refunded.")
