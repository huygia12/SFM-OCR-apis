# SFM-OCR-apis

Service that provides apis to scan KMA student forms

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
