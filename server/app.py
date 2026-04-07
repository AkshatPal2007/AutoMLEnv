import os
import sys
import uvicorn

# Add the project root so server/app.py can import main.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()