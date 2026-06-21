"""
bodycomp.py
─────────────────────────────────────────────────────────────────────────────
Extração de dados de composição corporal a partir de fotos/imagens dos
relatórios InBody, usando Claude com visão (não OCR tradicional).

Por quê visão em vez de OCR: relatórios InBody são tabelas densas com
números pequenos. OCR tradicional (Tesseract etc.) erra silenciosamente
em dígitos parecidos (1/7, 3/8, 0/8), o que é particularmente arriscado
em dados de saúde que alimentam decisões de treino. Claude com visão lê
o layout completo com contexto (sabe que "Peso" é um campo de kg, sabe
distinguir a tabela principal da tabela de análise músculo-gordura) e
retorna direto em JSON estruturado, validável antes de persistir.
"""

import base64
import json
import re
from datetime import datetime
from typing import Optional

import anthropic


EXTRACTION_PROMPT = """Você está vendo uma foto de um relatório InBody (analisador de composição corporal).
Extraia os seguintes campos exatamente como aparecem no relatório. Use ponto decimal (não vírgula) nos números.
Se um campo não estiver visível ou legível, use null — não invente valores.

Retorne APENAS um JSON com esta estrutura exata, sem markdown, sem explicação:

{
  "exam_date": "YYYY-MM-DD",
  "exam_time": "HH:MM",
  "weight_kg": 0.0,
  "body_water_l": 0.0,
  "protein_kg": 0.0,
  "minerals_kg": 0.0,
  "fat_mass_kg": 0.0,
  "skeletal_muscle_mass_kg": 0.0,
  "bmi": 0.0,
  "body_fat_pct": 0.0,
  "inbody_score": 0,
  "fat_free_mass_kg": 0.0,
  "basal_metabolic_rate_kcal": 0,
  "visceral_fat_level": 0,
  "waist_hip_ratio": 0.0,
  "smi": 0.0
}

Campos esperados no relatório (em português):
- exam_date/exam_time: campo "Data / Hora"
- weight_kg: "Peso" na tabela de Análise da Composição Corporal (kg)
- body_water_l: "Água Corporal Total" (L)
- protein_kg: "Proteína" (kg)
- minerals_kg: "Minerais" (kg)
- fat_mass_kg: "Massa de Gordura" (kg)
- skeletal_muscle_mass_kg: "Massa Muscular Esquelética" na seção Análise Músculo-Gordura (kg)
- bmi: "IMC" na seção Análise de Obesidade
- body_fat_pct: "PGC" (%) na seção Análise de Obesidade
- inbody_score: "Pontuação InBody" (X/100)
- fat_free_mass_kg: "Massa Livre de Gordura" em Dados adicionais
- basal_metabolic_rate_kcal: "Taxa Metabólica Basal" em Dados adicionais
- visceral_fat_level: "Nível" em Nível de Gordura Visceral
- waist_hip_ratio: "Relação Cintura-Quadril"
- smi: "SMI" em Dados adicionais

Retorne APENAS o JSON."""


def extract_inbody_data(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """
    Envia a imagem do relatório InBody para o Claude e extrai os campos
    estruturados. Retorna o dict já parseado, ou {"error": "..."} se a
    extração falhar ou o JSON vier malformado.
    """
    client = anthropic.Anthropic()
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        return {"error": "Claude não retornou JSON válido", "raw": raw}
    except Exception as e:
        return {"error": str(e)}


def validate_inbody_data(data: dict) -> tuple[bool, list[str]]:
    """
    Validação de sanidade nos valores extraídos antes de persistir.
    Faixas baseadas em limites fisiológicos plausíveis para adultos —
    não são limites clínicos, só uma rede de segurança contra erros
    grosseiros de leitura (ex: ler "13,9" como "139").
    """
    warnings = []

    checks = {
        "weight_kg": (30, 250),
        "bmi": (10, 60),
        "body_fat_pct": (2, 60),
        "inbody_score": (0, 100),
        "basal_metabolic_rate_kcal": (800, 4000),
        "visceral_fat_level": (1, 30),
        "skeletal_muscle_mass_kg": (10, 80),
    }

    for field, (lo, hi) in checks.items():
        value = data.get(field)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            warnings.append(f"{field}: valor não numérico ({data.get(field)})")
            continue
        if not (lo <= value <= hi):
            warnings.append(f"{field}: {value} fora da faixa plausível ({lo}-{hi})")

    if not data.get("exam_date"):
        warnings.append("exam_date ausente")
    else:
        try:
            datetime.strptime(data["exam_date"], "%Y-%m-%d")
        except ValueError:
            warnings.append(f"exam_date em formato inesperado: {data.get('exam_date')}")

    is_valid = len(warnings) == 0
    return is_valid, warnings


def build_trend_summary(history: list[dict]) -> dict:
    """
    Recebe o histórico de exames (mais recente primeiro) e monta um
    resumo de tendência pronto para o prompt do /analyze — comparando
    o exame mais recente com o anterior, e com o primeiro do histórico
    disponível (linha de base).
    """
    if not history:
        return {"disponivel": False}

    latest = history[0]
    summary = {
        "disponivel": True,
        "ultimo_exame": {
            "data": latest.get("exam_date"),
            "peso_kg": latest.get("weight_kg"),
            "percentual_gordura": latest.get("body_fat_pct"),
            "massa_muscular_kg": latest.get("skeletal_muscle_mass_kg"),
            "gordura_visceral": latest.get("visceral_fat_level"),
            "score_inbody": latest.get("inbody_score"),
        },
        "total_exames": len(history),
    }

    if len(history) >= 2:
        previous = history[1]

        def delta(field):
            a, b = latest.get(field), previous.get(field)
            if a is None or b is None:
                return None
            return round(a - b, 2)

        summary["variacao_desde_ultimo"] = {
            "peso_kg": delta("weight_kg"),
            "percentual_gordura": delta("body_fat_pct"),
            "massa_muscular_kg": delta("skeletal_muscle_mass_kg"),
            "data_anterior": previous.get("exam_date"),
        }

    if len(history) >= 3:
        baseline = history[-1]

        def delta_baseline(field):
            a, b = latest.get(field), baseline.get(field)
            if a is None or b is None:
                return None
            return round(a - b, 2)

        summary["variacao_desde_inicio"] = {
            "peso_kg": delta_baseline("weight_kg"),
            "percentual_gordura": delta_baseline("body_fat_pct"),
            "massa_muscular_kg": delta_baseline("skeletal_muscle_mass_kg"),
            "data_inicio": baseline.get("exam_date"),
        }

    return summary
