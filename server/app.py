import os
import sys
import uvicorn

# Add the parent directory to sys.path so we can import 'main' from the root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import app
except ImportError:
    # Fallback in case the script is run from the root directory
    from main import app


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()