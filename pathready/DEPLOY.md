# Panduan deploy PathReady ke Render (satu link untuk frontend + backend)

Pakej ini menghoskan laman web (`index.html`) dan API (`main.py`) sebagai **satu servis**.
Bila dibuka, `index.html` terus guna API di alamat yang sama, jadi tak perlu tukar URL dan tak ada masalah CORS.

## Fail dalam folder
| Fail | Fungsi |
|---|---|
| `index.html` | Laman web (alamat API dikesan automatik) |
| `main.py` | Backend FastAPI (`/predict`, `/screen-resume`, dan sajikan `index.html`) |
| `requirements.txt` | Senarai pakej Python |
| `render.yaml` | Tetapan Render (Blueprint) |
| `.gitignore`, `.env.example` | Supaya `.env` (kunci API) tidak ter-upload ke GitHub |

## 1. (Pilihan) Uji di komputer dulu
```bash
pip install -r requirements.txt
# pilihan: salin .env.example kepada .env dan isi OPENAI_API_KEY atau ANTHROPIC_API_KEY
uvicorn main:app --reload
```
Buka http://127.0.0.1:8000 . Tanpa kunci API, ciri resume guna kaedah kata kunci (offline).

## 2. Upload ke GitHub
1. Buat akaun di github.com, kemudian **New repository** (contoh nama: `pathready`).
2. Klik **Add file > Upload files**, seret semua fail dalam folder ini, kemudian **Commit changes**.
3. Pastikan tiada fail `.env` dalam repo.

## 3. Deploy di Render
1. Daftar di render.com guna akaun GitHub.
2. Klik **New + > Blueprint**, pilih repo `pathready`. Render akan baca `render.yaml`.
3. Bila diminta, isi `OPENAI_API_KEY` (atau `ANTHROPIC_API_KEY`). Boleh dikosongkan; ciri resume akan guna kaedah kata kunci.
4. Klik **Apply** dan tunggu build beberapa minit.
5. Buka URL yang diberi, contohnya `https://pathready-xxxx.onrender.com`.

Cara manual (tanpa Blueprint): **New + > Web Service**, pilih repo, Runtime `Python`,
Build Command `pip install -r requirements.txt`,
Start Command `uvicorn main:app --host 0.0.0.0 --port $PORT`,
dan tambah environment variables seperti dalam `render.yaml`.

## 4. Semak selepas deploy
- Buka `https://URL-anda/health`. Patut keluar `{"status":"ok",...}`.
- Di bahagian "ML integration", lencana atas dashboard patut tunjuk **Logistic Regression API** selepas anda tekan Analyse.
- Cuba "Load sample resume" > "Screen resume".

## Perkara penting
- **Servis percuma akan "tidur"** bila lama tiada pengguna. Request pertama selepas itu ambil sekitar 30 saat. Buka laman sekali sebelum demo.
- **Had resume:** `RATE_LIMIT_PER_HOUR` (default 10 dalam `render.yaml`) menghadkan jumlah screening setiap IP sejam supaya kredit API tidak habis. Tukar di Render > Environment.
- **Tetapkan had belanja** di akaun OpenAI atau Anthropic anda.
- **Jangan letak kunci API** dalam `index.html` atau GitHub. Letak di Render > Environment sahaja.
- **Alumni:** penghantaran rekod alumni dimatikan (`ENABLE_ALUMNI` tidak diset), kerana prototaip guna data sampel sahaja.
- **Cakera percuma Render tidak kekal.** Fail `alumni.db` akan hilang bila servis dimulakan semula. Tidak jadi masalah sebab data alumni tidak digunakan.
- **Privasi:** resume dihantar ke penyedia AI jika kunci diset. Pastikan nota privasi di laman kekal, dan dapatkan kebenaran UTP sebelum guna nama atau logo UTP secara awam.
- Jika build gagal kerana versi Python, tambah environment variable `PYTHON_VERSION` (contoh `3.12.8`) di Render dan deploy semula. Pelan percuma dan syarat Render boleh berubah, jadi semak di render.com/pricing.
