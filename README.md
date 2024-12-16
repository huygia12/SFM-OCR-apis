# SFM-OCR-apis

- Service that provides apis to scan KMA student forms</br>
- Currently, this service just works with these following applications:</br>
  [continue-study-application](ocr/base-form/continue-study-application/1.jpg)</br>
  [drop-out-school-application](ocr/base-form/drop-out-school-application/1.jpg)</br>
  [reissued-student-card-application](ocr/base-form/reissued-student-card-application/1.jpg)</br>
  [reissued-student-health-insurrance-application](ocr/base-form/reissued-student-health-insurrance-application/1.jpg)</br>

# Getting Started

1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate #linux distributions
venv\Scripts\activate #window
deactivate #deactivate virtual environment
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run server

```bash
uvicorn app.main:app --reload
```

4. Access
   - API document: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health check server: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/v1/health)

# Environment variables example:

- FRONTEND_HOST=http://localhost:3000
- ENVIRONMENT=local
- BACKEND_CORS_ORIGINS=http://localhost:4000,http://localhost:5173
