The attached zip file contains the current backend and frontend code for the GenAI Internal Research Tool. To run the backend, please create your own .env file with LLM API credentials, then run:

cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

The frontend can be started from the frontend folder using your standard React toolchain (for example npm install and npm start), depending on your internal environment.