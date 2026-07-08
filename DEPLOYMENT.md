# Deployment

## Railway API

Deploy the repository root as the Railway service.

Railway will use:

- `requirements.txt` for Python dependencies
- `runtime.txt` for Python 3.11
- `railway.json` / `Procfile` to start `uvicorn main:app`
- `PORT` from Railway automatically

Set this environment variable in Railway after your Vercel domain is ready:

```env
CORS_ORIGINS=https://your-vercel-app.vercel.app
```

For local development:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Vercel Web

Deploy the `apple-quality-web` folder as the Vercel project.

Set this environment variable in Vercel:

```env
VITE_API_URL=https://your-railway-api.up.railway.app
```

For local development, copy `apple-quality-web/.env.example` to `apple-quality-web/.env` and update the value if needed.
