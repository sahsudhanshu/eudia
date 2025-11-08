## Frontend (Next.js)

The frontend now communicates with the Flask backend via REST. Configure the API origin before running the dev server:

```bash
cp .env .env.local   # or create .env.local manually
```

Set the backend URL in `.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Then install dependencies and start the app:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to access the UI. Ensure the Flask service is running so authentication and data calls succeed.
