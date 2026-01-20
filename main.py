from dtu_mlops_111.api import app

if __name__ == "__main__":
    # Run the API server
    import uvicorn
    uvicorn.run(app, host="[IP_ADDRESS]", port=8000)
