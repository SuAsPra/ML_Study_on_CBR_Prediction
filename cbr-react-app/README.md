# CBR React App

This app predicts CBR using the final saved model artifacts from the project root:

- `final_cbr_model_with_pl.pkl`
- `imputer_with_pl.pkl`
- `features_with_pl.pkl`

Compulsory inputs:

- `LL (%)`
- `PL (%)`
- `OMC (%)`
- `MDD (g/cm3)`
- `Gravel (%)`
- `Sand (%)`
- `Fines`

Optional context fields:

- `PI (%)`
- `AASHTO`
- `UCSCS`
- `SPECIFIC GRAVITY`

## 1) Start backend API

From the project root:

```powershell
.\venv\Scripts\python.exe .\cbr-react-app\backend\server.py
```

Backend runs at `http://localhost:8000`.

## 2) Start React frontend

Open a new terminal in `cbr-react-app`:

```powershell
cd .\cbr-react-app
npm.cmd install
npm.cmd run dev
```

Frontend runs at `http://localhost:5173`.

## Notes

- The model was trained on **cohesive soils** (`UCSCS` containing clay/silt).  
  If `UCSCS` is not cohesive, the app still predicts but shows a warning.
- `PI`, `AASHTO`, `UCSCS`, and `SPECIFIC GRAVITY` are accepted in the form, but the current saved model does not use them as direct numerical prediction features.
- `UCSCS` is used for cohesive-soil warning logic.
