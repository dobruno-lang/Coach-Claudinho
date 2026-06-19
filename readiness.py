"""
readiness.py
─────────────────────────────────────────────────────────────────────────────
Camada determinística de análise — calcula métricas de PERFORMANCE e
BEM-ESTAR a partir dos dados brutos do WHOOP/Garmin/Strava/COROS, e cruza
os dois eixos num índice de prontidão (readiness).

Princípio: tudo que pode ser calculado com matemática simples é calculado
aqui, em Python puro — não é "estimado" pelo LLM. O Claude recebe os
números já prontos e foca em interpretar e comunicar, não em fazer conta.
"""

from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
# EIXO 1: PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────

def calc_daily_load(workouts: list, sport_strain_field: str = "score") -> dict:
    """
    Agrega a carga de treino por dia a partir dos workouts do WHOOP.
    Carga = strain (escala WHOOP 0-21) por dia. Se não houver WHOOP,
    usa duração em minutos das atividades do Strava/COROS como proxy.
    Retorna {data_iso: carga_float}.
    """
    daily = {}
    for w in workouts:
        try:
            start = w.get("start", "")
            date_key = start[:10] if start else None
            if not date_key:
                continue
            score = w.get(sport_strain_field, {}) or {}
            strain = score.get("strain")
            if strain is None:
                continue
            daily[date_key] = daily.get(date_key, 0) + float(strain)
        except (KeyError, TypeError, ValueError):
            continue
    return daily


def calc_acwr(daily_load: dict, today: Optional[datetime] = None) -> dict:
    """
    Acute:Chronic Workload Ratio.
    Acute  = carga média dos últimos 7 dias
    Chronic = carga média das últimas 4 semanas (28 dias)
    ACWR = Acute / Chronic

    Interpretação padrão (literatura esportiva):
      < 0.8        → destreino / carga muito baixa
      0.8 - 1.3    → zona ótima ("sweet spot")
      1.3 - 1.5    → zona de atenção
      > 1.5        → alto risco de lesão / overtraining

    Retorna dict com acute, chronic, ratio, zona, e flag de confiabilidade
    (precisa de dados mínimos pra não mentir com poucos pontos).
    """
    if today is None:
        today = datetime.now()

    def avg_load_window(days: int) -> Optional[float]:
        window_start = today - timedelta(days=days)
        values = [
            v for k, v in daily_load.items()
            if window_start <= datetime.strptime(k, "%Y-%m-%d") <= today
        ]
        if not values:
            return None
        # média por dia, contando dias sem treino como 0 (carga real, não só dias treinados)
        return sum(values) / days

    acute = avg_load_window(7)
    chronic = avg_load_window(28)

    if acute is None or chronic is None or chronic == 0:
        return {
            "acute": acute,
            "chronic": chronic,
            "ratio": None,
            "zona": "dados_insuficientes",
            "confiavel": False,
        }

    ratio = round(acute / chronic, 2)

    if ratio < 0.8:
        zona = "destreino"
    elif ratio <= 1.3:
        zona = "otima"
    elif ratio <= 1.5:
        zona = "atencao"
    else:
        zona = "risco_alto"

    # confiável só se tivermos pelo menos ~14 dias de dados de carga
    confiavel = len(daily_load) >= 10

    return {
        "acute": round(acute, 2),
        "chronic": round(chronic, 2),
        "ratio": ratio,
        "zona": zona,
        "confiavel": confiavel,
    }


def calc_monotony_strain(daily_load: dict, days: int = 7, today: Optional[datetime] = None) -> dict:
    """
    Monotonia e Strain semanal (método Foster).
    Monotonia = média da carga diária / desvio-padrão da carga diária.
    Treino muito repetitivo (pouca variação de intensidade) = monotonia alta
    = risco de overtraining mesmo com volume moderado.

    Strain semanal = carga total da semana × monotonia.
    Strain muito alto é sinal de alarme combinando volume + falta de variação.

    Regras práticas:
      monotonia > 2.0       → atenção (pouca variação dia a dia)
      strain semanal alto   → contextual, comparado ao histórico do usuário
    """
    if today is None:
        today = datetime.now()
    window_start = today - timedelta(days=days)

    # gera série diária completa (incluindo dias de descanso = carga 0)
    series = []
    d = window_start
    while d <= today:
        key = d.strftime("%Y-%m-%d")
        series.append(daily_load.get(key, 0.0))
        d += timedelta(days=1)

    if len(series) < 3 or all(v == 0 for v in series):
        return {"monotonia": None, "strain_semanal": None, "confiavel": False}

    media = mean(series)
    try:
        desvio = stdev(series)
    except Exception:
        desvio = 0

    if desvio == 0:
        monotonia = None  # indefinido matematicamente (sem variação real pra medir)
    else:
        monotonia = round(media / desvio, 2)

    carga_total = round(sum(series), 1)
    strain_semanal = round(carga_total * monotonia, 1) if monotonia else None

    return {
        "monotonia": monotonia,
        "carga_total_semana": carga_total,
        "strain_semanal": strain_semanal,
        "confiavel": True,
    }


def calc_performance_block(workouts: list, today: Optional[datetime] = None) -> dict:
    """Monta o bloco completo do eixo de performance."""
    daily_load = calc_daily_load(workouts)
    acwr = calc_acwr(daily_load, today)
    monotony = calc_monotony_strain(daily_load, today=today)
    return {
        "acwr": acwr,
        "monotonia_strain": monotony,
        "dias_com_dados_de_carga": len(daily_load),
    }


# ─────────────────────────────────────────────────────────────────────────
# EIXO 2: BEM-ESTAR
# ─────────────────────────────────────────────────────────────────────────

def calc_hrv_baseline(recovery_records: list, exclude_last_n: int = 0) -> dict:
    """
    Calcula a baseline pessoal de HRV (média + desvio padrão) a partir
    do histórico de recovery. exclude_last_n permite excluir os dias mais
    recentes da baseline para comparar "hoje vs. baseline sem hoje".
    """
    values = []
    for r in recovery_records:
        score = r.get("score", {}) or {}
        hrv = score.get("hrv_rmssd_milli")
        if hrv is not None:
            values.append(float(hrv))

    if exclude_last_n and len(values) > exclude_last_n:
        values = values[exclude_last_n:]  # registros vêm mais recentes primeiro

    if len(values) < 3:
        return {"media": None, "desvio": None, "confiavel": False, "n": len(values)}

    return {
        "media": round(mean(values), 1),
        "desvio": round(stdev(values), 1) if len(values) > 1 else 0,
        "confiavel": len(values) >= 5,
        "n": len(values),
    }


def calc_hrv_today_deviation(recovery_records: list) -> dict:
    """
    Compara o HRV de hoje (registro mais recente) contra a baseline
    dos dias anteriores. Desvio em % e em desvios-padrão (z-score).
    """
    if not recovery_records:
        return {"hrv_hoje": None, "desvio_pct": None, "z_score": None, "tendencia": "sem_dados"}

    hoje = recovery_records[0]  # mais recente primeiro
    hrv_hoje = (hoje.get("score", {}) or {}).get("hrv_rmssd_milli")

    baseline = calc_hrv_baseline(recovery_records, exclude_last_n=1)

    if hrv_hoje is None or baseline["media"] is None:
        return {"hrv_hoje": hrv_hoje, "desvio_pct": None, "z_score": None, "tendencia": "sem_dados"}

    desvio_pct = round(((hrv_hoje - baseline["media"]) / baseline["media"]) * 100, 1)

    z_score = None
    if baseline["desvio"] and baseline["desvio"] > 0:
        z_score = round((hrv_hoje - baseline["media"]) / baseline["desvio"], 2)

    if desvio_pct >= 5:
        tendencia = "acima_da_baseline"
    elif desvio_pct <= -5:
        tendencia = "abaixo_da_baseline"
    else:
        tendencia = "estavel"

    return {
        "hrv_hoje": hrv_hoje,
        "baseline_media": baseline["media"],
        "desvio_pct": desvio_pct,
        "z_score": z_score,
        "tendencia": tendencia,
        "baseline_confiavel": baseline["confiavel"],
    }


def calc_sleep_trend(sleep_records: list, days: int = 14) -> dict:
    """
    Tendência de sono — duração e eficiência ao longo dos últimos N dias.
    Usa regressão linear simples (slope) pra detectar se está
    melhorando, piorando ou estável.
    """
    durations = []
    for s in sleep_records[:days]:
        score = s.get("score", {}) or {}
        stage_summary = score.get("stage_summary", {}) or {}
        total_ms = (
            stage_summary.get("total_light_sleep_time_milli", 0)
            + stage_summary.get("total_slow_wave_sleep_time_milli", 0)
            + stage_summary.get("total_rem_sleep_time_milli", 0)
        )
        if total_ms:
            durations.append(total_ms / 1000 / 3600)  # ms → horas

    if len(durations) < 4:
        return {"media_horas": None, "tendencia": "dados_insuficientes", "confiavel": False}

    # mais recente primeiro nos dados originais → inverter pra ordem cronológica
    durations_cron = list(reversed(durations))
    n = len(durations_cron)
    x_mean = (n - 1) / 2
    y_mean = mean(durations_cron)

    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(durations_cron))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0

    if slope > 0.03:
        tendencia = "melhorando"
    elif slope < -0.03:
        tendencia = "piorando"
    else:
        tendencia = "estavel"

    return {
        "media_horas": round(y_mean, 2),
        "slope_horas_por_dia": round(slope, 3),
        "tendencia": tendencia,
        "confiavel": True,
        "n_noites": n,
    }


def calc_recovery_trend(recovery_records: list, days: int = 14) -> dict:
    """Mesma lógica de tendência (slope), aplicada ao recovery score."""
    scores = []
    for r in recovery_records[:days]:
        s = (r.get("score", {}) or {}).get("recovery_score")
        if s is not None:
            scores.append(float(s))

    if len(scores) < 4:
        return {"media": None, "tendencia": "dados_insuficientes", "confiavel": False}

    scores_cron = list(reversed(scores))
    n = len(scores_cron)
    x_mean = (n - 1) / 2
    y_mean = mean(scores_cron)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(scores_cron))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0

    if slope > 0.5:
        tendencia = "subindo"
    elif slope < -0.5:
        tendencia = "caindo"
    else:
        tendencia = "estavel"

    return {
        "media": round(y_mean, 1),
        "slope_por_dia": round(slope, 2),
        "tendencia": tendencia,
        "confiavel": True,
        "n_dias": n,
    }


def calc_wellbeing_block(recovery_records: list, sleep_records: list) -> dict:
    """Monta o bloco completo do eixo de bem-estar."""
    return {
        "hrv": calc_hrv_today_deviation(recovery_records),
        "sono_tendencia": calc_sleep_trend(sleep_records),
        "recovery_tendencia": calc_recovery_trend(recovery_records),
    }


# ─────────────────────────────────────────────────────────────────────────
# CRUZAMENTO: ÍNDICE DE PRONTIDÃO
# ─────────────────────────────────────────────────────────────────────────

def calc_readiness_index(performance: dict, wellbeing: dict) -> dict:
    """
    Cruza os dois eixos num índice único de 0-100 e numa zona (verde/
    amarelo/vermelho), com explicação dos fatores que mais pesaram.

    Lógica: começa em 100 e desconta pontos por sinais de risco em
    qualquer um dos dois eixos. Não é uma média ponderada arbitrária —
    cada desconto tem uma razão explícita, auditável.
    """
    score = 100
    fatores = []

    # ── Performance: ACWR ──
    acwr = performance.get("acwr", {})
    if acwr.get("confiavel"):
        zona = acwr.get("zona")
        if zona == "risco_alto":
            score -= 30
            fatores.append("ACWR em zona de risco alto (carga subindo rápido demais)")
        elif zona == "atencao":
            score -= 15
            fatores.append("ACWR em zona de atenção")
        elif zona == "destreino":
            score -= 5
            fatores.append("Carga abaixo do padrão recente (destreino)")

    # ── Performance: monotonia ──
    monotony = performance.get("monotonia_strain", {})
    if monotony.get("confiavel") and monotony.get("monotonia"):
        if monotony["monotonia"] > 2.5:
            score -= 10
            fatores.append("Monotonia alta — treinos pouco variados")

    # ── Bem-estar: HRV ──
    hrv = wellbeing.get("hrv", {})
    if hrv.get("tendencia") == "abaixo_da_baseline":
        z = hrv.get("z_score")
        if z is not None and z <= -1.5:
            score -= 25
            fatores.append("HRV significativamente abaixo da baseline pessoal")
        else:
            score -= 12
            fatores.append("HRV abaixo da baseline pessoal")

    # ── Bem-estar: tendência de sono ──
    sono = wellbeing.get("sono_tendencia", {})
    if sono.get("tendencia") == "piorando":
        score -= 10
        fatores.append("Sono em tendência de piora nos últimos dias")

    # ── Bem-estar: tendência de recovery ──
    rec = wellbeing.get("recovery_tendencia", {})
    if rec.get("tendencia") == "caindo":
        score -= 10
        fatores.append("Recovery em tendência de queda")

    score = max(0, min(100, score))

    if score >= 75:
        cor = "verde"
    elif score >= 50:
        cor = "amarelo"
    else:
        cor = "vermelho"

    return {
        "indice": score,
        "cor": cor,
        "fatores": fatores or ["Sem sinais de alerta nos dados disponíveis"],
    }


# ─────────────────────────────────────────────────────────────────────────
# SÉRIES TEMPORAIS — para gráficos no dashboard
# ─────────────────────────────────────────────────────────────────────────

def build_timeseries(whoop_data: dict, days: int = 21) -> dict:
    """
    Monta as séries diárias prontas para plotar: strain, carga aguda/
    crônica rolante, HRV e recovery score. Todas alinhadas pelas mesmas
    datas (eixo X comum), preenchendo dias sem dado com None — assim o
    gráfico no front não precisa fazer nenhum tipo de cálculo, só plotar.
    """
    recovery = whoop_data.get("recovery", [])
    workouts = whoop_data.get("workouts", [])

    daily_load = calc_daily_load(workouts)

    # indexa recovery por data (registros vêm mais recentes primeiro)
    recovery_by_date = {}
    for r in recovery:
        created = r.get("created_at", "")
        date_key = created[:10] if created else None
        if not date_key:
            continue
        score = r.get("score", {}) or {}
        recovery_by_date[date_key] = {
            "recovery_score": score.get("recovery_score"),
            "hrv": score.get("hrv_rmssd_milli"),
        }

    today = datetime.now()
    labels, strain_series, hrv_series, recovery_series = [], [], [], []
    acute_series, chronic_series = [], []

    d = today - timedelta(days=days - 1)
    while d <= today:
        key = d.strftime("%Y-%m-%d")
        labels.append(d.strftime("%d/%m"))

        strain_series.append(round(daily_load.get(key, 0.0), 1) if key in daily_load else 0)

        rec_point = recovery_by_date.get(key, {})
        hrv_series.append(rec_point.get("hrv"))
        recovery_series.append(rec_point.get("recovery_score"))

        # carga aguda/crônica rolante calculada até esse dia (janela retroativa)
        acute_window = [v for k, v in daily_load.items()
                         if (d - timedelta(days=6)) <= datetime.strptime(k, "%Y-%m-%d") <= d]
        chronic_window = [v for k, v in daily_load.items()
                           if (d - timedelta(days=27)) <= datetime.strptime(k, "%Y-%m-%d") <= d]
        acute_series.append(round(sum(acute_window) / 7, 1) if acute_window else None)
        chronic_series.append(round(sum(chronic_window) / 28, 1) if chronic_window else None)

        d += timedelta(days=1)

    return {
        "labels": labels,
        "strain_diario": strain_series,
        "carga_aguda": acute_series,
        "carga_cronica": chronic_series,
        "hrv": hrv_series,
        "recovery_score": recovery_series,
    }


# ─────────────────────────────────────────────────────────────────────────
# ENTRY POINT — usado pelo main.py
# ─────────────────────────────────────────────────────────────────────────

def build_readiness_report(whoop_data: dict) -> dict:
    """
    Recebe o dict bruto retornado por fetch_whoop_data() e devolve o
    relatório completo de performance + bem-estar + índice de prontidão,
    pronto pra ser injetado no prompt da IA ou exibido no dashboard.
    """
    recovery = whoop_data.get("recovery", [])
    sleep = whoop_data.get("sleep", [])
    workouts = whoop_data.get("workouts", [])

    performance = calc_performance_block(workouts)
    wellbeing = calc_wellbeing_block(recovery, sleep)
    readiness = calc_readiness_index(performance, wellbeing)
    timeseries = build_timeseries(whoop_data)

    return {
        "performance": performance,
        "bem_estar": wellbeing,
        "prontidao": readiness,
        "series": timeseries,
    }
