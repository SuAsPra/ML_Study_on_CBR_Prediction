# CBR Prediction App - Hosting Package

This folder contains the clean deployable version of the CBR prediction project.

## Structure

```text
SOIL_ML_TO_HOST/
├── backend/
│   ├── server.py
│   ├── requirements.txt
│   └── runtime.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   ├── index.html
│   └── vite.config.js
├── models/
│   ├── final_cbr_model_with_pl.pkl
│   ├── imputer_with_pl.pkl
│   └── features_with_pl.pkl
└── render.yaml
```

## Local Run

Start backend:

```powershell
cd SOIL_ML_TO_HOST
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python backend\server.py
```

Backend runs at:

```text
http://localhost:8000
```

Start frontend in another terminal:

```powershell
cd SOIL_ML_TO_HOST\frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

## Render Backend Deployment

Create a new Render Web Service using this folder as the project root.

Build command:

```bash
pip install -r backend/requirements.txt
```

Start command:

```bash
python backend/server.py
```

Render will provide a backend URL such as:

```text
https://your-service-name.onrender.com
```

Check:

```text
https://your-service-name.onrender.com/health
```

## Vercel Frontend Deployment

Deploy the `frontend` folder on Vercel.

Settings:

```text
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
```

Add this Vercel environment variable:

```text
VITE_API_BASE_URL=https://your-render-backend-url.onrender.com
```

Then deploy.

## Required App Inputs

The deployed app requires:

- LL (%)
- PL (%)
- OMC (%)
- MDD (g/cm3)
- Gravel (%)
- Sand (%)
- Fines

Optional:

- PI
- AASHTO
- UCSCS
- Specific Gravity

