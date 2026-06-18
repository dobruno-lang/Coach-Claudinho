from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import httpx
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic
import asyncpg
from contextlib import asynccontextmanager

load_dotenv()

# ─── DB ────────────────────────────────────────────────────────────────────────
async def get_db():
    return await asyncpg.connect(os.getenv("DATABASE_URL"))

async def init_db():
    conn = await get_db()
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            provider TEXT PRIMARY KEY,
            access_token TEXT,
            refresh_token TEXT,
            expires_at BIGINT
        );
        CREATE TABLE IF NOT EXISTS activities (
            id TEXT PRIMARY KEY,
            provider TEXT,
            date DATE,
            type TEXT,
            data JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS daily_metrics (
            date DATE PRIMARY KEY,
            whoop_recovery FLOAT,
            whoop_hrv FLOAT,
            whoop_rhr FLOAT,
            whoop_sleep_score FLOAT,
            whoop_strain FLOAT,
            garmin_body_battery FLOAT,
            garmin_stress FLOAT,
            garmin_steps INT,
            garmin_hrv FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    await conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Personal Coach Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── WHOOP OAuth ───────────────────────────────────────────────────────────────
WHOOP_CLIENT_ID     = os.getenv("WHOOP_CLIENT_ID")
WHOOP_CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")
WHOOP_REDIRECT_URI  = os.getenv("BASE_URL") + "/auth/whoop/callback"
WHOOP_AUTH_URL      = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL     = "https://api.prod.whoop.com/oauth/oauth2/token"
WHOOP_API_BASE      = "https://api.prod.whoop.com/developer/v1"

# ─── Garmin Health API ─────────────────────────────────────────────────────────
GARMIN_CLIENT_ID     = os.getenv("GARMIN_CLIENT_ID")
GARMIN_CLIENT_SECRET = os.getenv("GARMIN_CLIENT_SECRET")
GARMIN_REDIRECT_URI  = os.getenv("BASE_URL") + "/auth/garmin/callback"
GARMIN_AUTH_URL      = "https://connect.garmin.com/oauthConfirm"
GARMIN_TOKEN_URL     = "https://connect.garmin.com/oauth-service/oauth/access_token"
GARMIN_API_BASE      = "https://apis.garmin.com/wellness-api/rest"

# ─── Strava OAuth ──────────────────────────────────────────────────────────────
STRAVA_CLIENT_ID     = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
STRAVA_REDIRECT_URI  = os.getenv("BASE_URL") + "/auth/strava/callback"
STRAVA_AUTH_URL      = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL     = "https://www.strava.com/oauth/token"
STRAVA_API_BASE      = "https://www.strava.com/api/v3"

# ─── COROS API ─────────────────────────────────────────────────────────────────
COROS_CLIENT_ID     = os.getenv("COROS_CLIENT_ID")
COROS_CLIENT_SECRET = os.getenv("COROS_CLIENT_SECRET")
COROS_REDIRECT_URI  = os.getenv("BASE_URL") + "/auth/coros/callback"
COROS_AUTH_URL      = "https://open.coros.com/oauth2/authorize"
COROS_TOKEN_URL     = "https://open.coros.com/oauth2/accesstoken"
COROS_API_BASE      = "https://open.coros.com"

# ─── Token helpers ─────────────────────────────────────────────────────────────
async def save_token(provider: str, access: str, refresh: str, expires_in: int):
    conn = await get_db()
    expires_at = int(datetime.now().timestamp()) + expires_in
    await conn.execute("""
        INSERT INTO tokens (provider, access_token, refresh_token, expires_at)
        VALUES ($1,$2,$3,$4)
        ON CONFLICT (provider) DO UPDATE
        SET access_token=$2, refresh_token=$3, expires_at=$4
    """, provider, access, refresh, expires_at)
    await conn.close()

async def get_token(provider: str) -> dict | None:
    conn = await get_db()
    row = await conn.fetchrow("SELECT * FROM tokens WHERE provider=$1", provider)
    await conn.close()
    if not row:
        return None
    return dict(row)

async def refresh_whoop_token(refresh_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(WHOOP_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": WHOOP_CLIENT_ID,
            "client_secret": WHOOP_CLIENT_SECRET,
        })
        return r.json()

async def refresh_strava_token(refresh_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(STRAVA_TOKEN_URL, data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        })
        return r.json()

async def valid_token(provider: str) -> str | None:
    t = await get_token(provider)
    if not t:
        return None
    now = int(datetime.now().timestamp())
    if t["expires_at"] - now < 300:  # refresh se expira em <5min
        if provider == "whoop":
            new = await refresh_whoop_token(t["refresh_token"])
            await save_token(provider, new["access_token"], new["refresh_token"], new["expires_in"])
            return new["access_token"]
        elif provider == "strava":
            new = await refresh_strava_token(t["refresh_token"])
            await save_token(provider, new["access_token"], new["refresh_token"], new["expires_in"])
            return new["access_token"]
    return t["access_token"]

# ─── Auth routes ───────────────────────────────────────────────────────────────
@app.get("/auth/whoop")
async def auth_whoop():
    url = (f"{WHOOP_AUTH_URL}?client_id={WHOOP_CLIENT_ID}"
           f"&redirect_uri={WHOOP_REDIRECT_URI}&response_type=code"
           f"&scope=read:recovery read:sleep read:workout read:body_measurement offline")
    return RedirectResponse(url)

@app.get("/auth/whoop/callback")
async def auth_whoop_callback(code: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(WHOOP_TOKEN_URL, data={
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": WHOOP_REDIRECT_URI,
            "client_id": WHOOP_CLIENT_ID,
            "client_secret": WHOOP_CLIENT_SECRET,
        })
    data = r.json()
    await save_token("whoop", data["access_token"], data["refresh_token"], data["expires_in"])
    return {"status": "WHOOP conectado!", "provider": "whoop"}

@app.get("/auth/garmin")
async def auth_garmin():
    url = (f"{GARMIN_AUTH_URL}?client_id={GARMIN_CLIENT_ID}"
           f"&redirect_uri={GARMIN_REDIRECT_URI}&response_type=code&scope=")
    return RedirectResponse(url)

@app.get("/auth/garmin/callback")
async def auth_garmin_callback(code: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(GARMIN_TOKEN_URL, data={
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": GARMIN_REDIRECT_URI,
            "client_id": GARMIN_CLIENT_ID,
            "client_secret": GARMIN_CLIENT_SECRET,
        })
    data = r.json()
    await save_token("garmin", data["access_token"], data.get("refresh_token",""), data.get("expires_in", 3600))
    return {"status": "Garmin conectado!", "provider": "garmin"}

@app.get("/auth/strava")
async def auth_strava():
    url = (f"{STRAVA_AUTH_URL}?client_id={STRAVA_CLIENT_ID}"
           f"&redirect_uri={STRAVA_REDIRECT_URI}&response_type=code"
           f"&approval_prompt=auto&scope=read,activity:read_all")
    return RedirectResponse(url)

@app.get("/auth/strava/callback")
async def auth_strava_callback(code: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(STRAVA_TOKEN_URL, data={
            "code": code,
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "grant_type": "authorization_code",
        })
    data = r.json()
    await save_token("strava", data["access_token"], data["refresh_token"], data["expires_in"])
    return {"status": "Strava conectado!", "provider": "strava"}

@app.get("/auth/coros")
async def auth_coros():
    url = (f"{COROS_AUTH_URL}?client_id={COROS_CLIENT_ID}"
           f"&redirect_uri={COROS_REDIRECT_URI}&response_type=code&state=coros")
    return RedirectResponse(url)

@app.get("/auth/coros/callback")
async def auth_coros_callback(code: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(COROS_TOKEN_URL, data={
            "code": code,
            "client_id": COROS_CLIENT_ID,
            "client_secret": COROS_CLIENT_SECRET,
            "redirect_uri": COROS_REDIRECT_URI,
            "grant_type": "authorization_code",
        })
    data = r.json()
    token_data = data.get("data", data)
    await save_token("coros", token_data["access_token"], token_data.get("refresh_token",""), token_data.get("expires_in", 7200))
    return {"status": "COROS conectado!", "provider": "coros"}

# ─── Status das conexões ───────────────────────────────────────────────────────
@app.get("/status")
async def status():
    result = {}
    for provider in ["whoop", "garmin", "strava", "coros"]:
        t = await get_token(provider)
        result[provider] = {
            "connected": t is not None,
            "expires_at": datetime.fromtimestamp(t["expires_at"]).isoformat() if t else None
        }
    return result

# ─── Coleta de dados WHOOP ─────────────────────────────────────────────────────
async def fetch_whoop_data(days: int = 7) -> dict:
    token = await valid_token("whoop")
    if not token:
        return {}
    headers = {"Authorization": f"Bearer {token}"}
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00.000Z")
    
    async with httpx.AsyncClient() as client:
        # Recovery
        r_rec = await client.get(f"{WHOOP_API_BASE}/recovery", headers=headers,
                                  params={"start": start, "limit": days})
        # Sleep
        r_sleep = await client.get(f"{WHOOP_API_BASE}/activity/sleep", headers=headers,
                                    params={"start": start, "limit": days})
        # Workouts
        r_work = await client.get(f"{WHOOP_API_BASE}/activity/workout", headers=headers,
                                   params={"start": start, "limit": days * 2})

    return {
        "recovery": r_rec.json().get("records", []),
        "sleep": r_sleep.json().get("records", []),
        "workouts": r_work.json().get("records", []),
    }

# ─── Coleta de dados Garmin ────────────────────────────────────────────────────
async def fetch_garmin_data(days: int = 7) -> dict:
    token = await valid_token("garmin")
    if not token:
        return {}
    headers = {"Authorization": f"Bearer {token}"}
    upload_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    upload_end = datetime.now().strftime("%Y-%m-%d")

    async with httpx.AsyncClient() as client:
        r_daily = await client.get(
            f"{GARMIN_API_BASE}/dailies",
            headers=headers,
            params={"uploadStartTimeInSeconds": upload_start, "uploadEndTimeInSeconds": upload_end}
        )
        r_sleep = await client.get(
            f"{GARMIN_API_BASE}/sleeps",
            headers=headers,
            params={"uploadStartTimeInSeconds": upload_start, "uploadEndTimeInSeconds": upload_end}
        )

    return {
        "dailies": r_daily.json() if r_daily.status_code == 200 else [],
        "sleep": r_sleep.json() if r_sleep.status_code == 200 else [],
    }

# ─── Coleta de dados Strava ────────────────────────────────────────────────────
async def fetch_strava_data(days: int = 14) -> list:
    token = await valid_token("strava")
    if not token:
        return []
    headers = {"Authorization": f"Bearer {token}"}
    after = int((datetime.now() - timedelta(days=days)).timestamp())

    async with httpx.AsyncClient() as client:
        r = await client.get(f"{STRAVA_API_BASE}/athlete/activities",
                             headers=headers,
                             params={"after": after, "per_page": 30})
    return r.json() if r.status_code == 200 else []

# ─── Coleta de dados COROS ────────────────────────────────────────────────────
async def fetch_coros_data(days: int = 14) -> list:
    token = await valid_token("coros")
    if not token:
        return []
    headers = {"Authorization": f"Bearer {token}"}
    start = int((datetime.now() - timedelta(days=days)).timestamp())
    end   = int(datetime.now().timestamp())

    async with httpx.AsyncClient() as client:
        r = await client.get(f"{COROS_API_BASE}/v2/coros/sport/list",
                             headers=headers,
                             params={"startTime": start, "endTime": end, "size": 50})
    data = r.json()
    return data.get("data", {}).get("dataList", []) if r.status_code == 200 else []

# ─── Sync endpoint ─────────────────────────────────────────────────────────────
@app.post("/sync")
async def sync_all(days: int = 7):
    whoop   = await fetch_whoop_data(days)
    garmin  = await fetch_garmin_data(days)
    strava  = await fetch_strava_data(days)
    coros   = await fetch_coros_data(days)
    return {
        "whoop": whoop,
        "garmin": garmin,
        "strava": strava,
        "coros": coros,
        "synced_at": datetime.now().isoformat()
    }

# ─── Análise com Claude ────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(days: int = 7):
    data = await sync_all(days)

    ai = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""Você é um treinador pessoal especialista em corrida e performance atlética.
Analise os dados abaixo dos últimos {days} dias e retorne um JSON com esta estrutura exata:

{{
  "resumo": "análise geral em 2-3 frases",
  "estado_recovery": "verde|amarelo|vermelho",
  "carga_semana": "baixa|moderada|alta|muito_alta",
  "insights": ["insight 1", "insight 2", "insight 3"],
  "alerta": "alerta importante se houver, ou null",
  "plano_semana": [
    {{
      "dia": "Segunda",
      "data": "YYYY-MM-DD",
      "tipo": "Corrida fácil|Intervalado|Tempo run|Long run|Força|Descanso ativo|Descanso",
      "descricao": "detalhes do treino",
      "duracao_min": 45,
      "distancia_km": 8.0,
      "zona_fc": "Z1-Z2",
      "intensidade": "leve|moderada|alta",
      "justificativa": "por que esse treino hoje"
    }}
  ]
}}

Dados disponíveis:
WHOOP: {json.dumps(data.get('whoop', {}), ensure_ascii=False, default=str)[:3000]}
GARMIN: {json.dumps(data.get('garmin', {}), ensure_ascii=False, default=str)[:2000]}
STRAVA (atividades recentes): {json.dumps(data.get('strava', [])[:5], ensure_ascii=False, default=str)[:2000]}
COROS (corridas recentes): {json.dumps(data.get('coros', [])[:5], ensure_ascii=False, default=str)[:2000]}

Retorne APENAS o JSON, sem markdown, sem explicações."""

    msg = ai.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = msg.content[0].text.strip()
    try:
        analysis = json.loads(raw)
    except:
        analysis = {"raw": raw}

    return {"analysis": analysis, "raw_data": data, "generated_at": datetime.now().isoformat()}

# ─── Gerar planilha Excel ──────────────────────────────────────────────────────
@app.get("/export/excel")
async def export_excel(days: int = 7):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    import io
    from fastapi.responses import StreamingResponse

    result = await analyze(days)
    analysis = result["analysis"]
    plan = analysis.get("plano_semana", [])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Plano de Treino"

    # Cores
    header_fill = PatternFill("solid", fgColor="1A1A2E")
    verde_fill  = PatternFill("solid", fgColor="2D6A4F")
    amarelo_fill= PatternFill("solid", fgColor="E9C46A")
    vermelho_fill=PatternFill("solid", fgColor="C1121F")
    alt_fill    = PatternFill("solid", fgColor="F8F9FA")

    tipo_cores = {
        "Descanso": "ADB5BD",
        "Descanso ativo": "CED4DA",
        "Corrida fácil": "A8DADC",
        "Long run": "457B9D",
        "Intervalado": "E63946",
        "Tempo run": "F4A261",
        "Força": "8338EC",
    }

    bold_white = Font(bold=True, color="FFFFFF", size=11)
    bold_dark  = Font(bold=True, color="1A1A2E", size=11)
    thin = Side(style="thin", color="DEE2E6")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def hcell(cell, value, fill=header_fill, font=bold_white, align="center"):
        cell.value = value
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
        cell.border = border

    def dcell(cell, value, fill=None, align="left", bold=False):
        cell.value = value
        if fill:
            cell.fill = fill
        cell.font = Font(bold=bold, color="1A1A2E", size=10)
        cell.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
        cell.border = border

    # ── Título ──
    ws.merge_cells("A1:I1")
    hcell(ws["A1"], "🏃 PLANO DE TREINO PERSONALIZADO", font=Font(bold=True, color="FFFFFF", size=14))
    ws.row_dimensions[1].height = 35

    ws.merge_cells("A2:I2")
    ws["A2"].value = f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} • Análise dos últimos {days} dias"
    ws["A2"].font = Font(color="6C757D", italic=True, size=10)
    ws["A2"].alignment = Alignment(horizontal="center")

    # ── Resumo ──
    ws.merge_cells("A4:I4")
    hcell(ws["A4"], "ANÁLISE GERAL", font=Font(bold=True, color="FFFFFF", size=11))

    ws.merge_cells("A5:I5")
    ws["A5"].value = analysis.get("resumo", "")
    ws["A5"].font = Font(size=10, color="1A1A2E")
    ws["A5"].alignment = Alignment(wrap_text=True, vertical="center")
    ws["A5"].fill = PatternFill("solid", fgColor="E8F4F8")
    ws.row_dimensions[5].height = 45

    # Status badges
    estado = analysis.get("estado_recovery", "verde")
    carga  = analysis.get("carga_semana", "moderada")
    estado_fill = {"verde": verde_fill, "amarelo": amarelo_fill, "vermelho": vermelho_fill}.get(estado, verde_fill)

    ws.merge_cells("A7:D7")
    hcell(ws["A7"], f"RECOVERY: {estado.upper()}", fill=estado_fill)

    ws.merge_cells("E7:I7")
    hcell(ws["E7"], f"CARGA DA SEMANA: {carga.upper()}", fill=PatternFill("solid", fgColor="264653"))
    ws.row_dimensions[7].height = 25

    # Insights
    ws.merge_cells("A9:I9")
    hcell(ws["A9"], "INSIGHTS")
    for i, insight in enumerate(analysis.get("insights", []), start=10):
        ws.merge_cells(f"A{i}:I{i}")
        ws[f"A{i}"].value = f"• {insight}"
        ws[f"A{i}"].font = Font(size=10)
        ws[f"A{i}"].alignment = Alignment(wrap_text=True, vertical="center")
        ws[f"A{i}"].fill = PatternFill("solid", fgColor="F0F7FF") if i % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        ws.row_dimensions[i].height = 30

    alerta = analysis.get("alerta")
    row_after = 10 + len(analysis.get("insights", []))
    if alerta:
        ws.merge_cells(f"A{row_after}:I{row_after}")
        hcell(ws[f"A{row_after}"], f"⚠️ {alerta}", fill=vermelho_fill)
        ws.row_dimensions[row_after].height = 30
        row_after += 1

    # ── Plano semanal ──
    row = row_after + 1
    ws.merge_cells(f"A{row}:I{row}")
    hcell(ws[f"A{row}"], "PLANO DA SEMANA")
    ws.row_dimensions[row].height = 28
    row += 1

    headers = ["Dia", "Data", "Tipo de Treino", "Descrição", "Duração", "Distância", "Zona FC", "Intensidade", "Justificativa"]
    cols    = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    for col, h in zip(cols, headers):
        hcell(ws[f"{col}{row}"], h)
    ws.row_dimensions[row].height = 22
    row += 1

    for i, treino in enumerate(plan):
        tipo = treino.get("tipo", "")
        cor_hex = tipo_cores.get(tipo, "FFFFFF")
        tipo_fill = PatternFill("solid", fgColor=cor_hex)
        bg_fill = PatternFill("solid", fgColor="F8F9FA") if i % 2 == 0 else None

        dcell(ws[f"A{row}"], treino.get("dia", ""), fill=bg_fill, bold=True)
        dcell(ws[f"B{row}"], treino.get("data", ""), fill=bg_fill, align="center")
        hcell(ws[f"C{row}"], tipo, fill=tipo_fill, font=Font(bold=True, color="1A1A2E" if cor_hex in ["A8DADC","E9C46A","ADB5BD","CED4DA"] else "FFFFFF"))
        dcell(ws[f"D{row}"], treino.get("descricao", ""), fill=bg_fill)
        dcell(ws[f"E{row}"], f"{treino.get('duracao_min', '')} min", fill=bg_fill, align="center")
        dcell(ws[f"F{row}"], f"{treino.get('distancia_km', '')} km" if treino.get("distancia_km") else "–", fill=bg_fill, align="center")
        dcell(ws[f"G{row}"], treino.get("zona_fc", ""), fill=bg_fill, align="center")
        dcell(ws[f"H{row}"], treino.get("intensidade", ""), fill=bg_fill, align="center")
        dcell(ws[f"I{row}"], treino.get("justificativa", ""), fill=bg_fill)
        ws.row_dimensions[row].height = 50
        row += 1

    # Larguras das colunas
    widths = [12, 12, 18, 40, 10, 12, 10, 12, 40]
    for col, w in zip(cols, widths):
        ws.column_dimensions[col].width = w

    ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    filename = f"treino_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})

# ─── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "Personal Coach Agent"}
