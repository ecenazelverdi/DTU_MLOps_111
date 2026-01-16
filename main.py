from dtu_mlops_111.api import app

if __name__ == "__main__":
    # Run the API server
    # IMPORTANT: Run "uv pip install -e ." or "uv pip install ." before running this 
    # script or it might not able to find "dtu_mlops_111"
    import uvicorn
    uvicorn.run(app, host="[IP_ADDRESS]", port=8000)
