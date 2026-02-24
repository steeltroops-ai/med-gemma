import uvicorn
import os

if __name__ == "__main__":
    # Get port from environment (Hugging Face Spaces uses 7860)
    port = int(os.environ.get("PORT", 7860))
    # Run uvicorn server pointing to the app in src.api.main
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
