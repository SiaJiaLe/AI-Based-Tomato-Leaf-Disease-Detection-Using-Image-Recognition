"""Seed treatment_options table with Malaysian-context disease treatments.

Usage (from backend/ directory):
    python scripts/seed_treatments.py
"""
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from infrastructure.persistence.models import TreatmentOptionRecord
from shared.config import settings

TREATMENTS = [
    # ── Bacterial Spot ──────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Bacterial_spot",
        "treatment_type": "chemical",
        "product_name": "Kocide 2000 (Copper Hydroxide)",
        "active_ingredient": "Copper hydroxide 53.8%",
        "application_method": "Spray at 3–4 g/L water every 7–10 days. Apply in early morning or late evening.",
        "estimated_cost_myr": 45.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Bacterial_spot",
        "treatment_type": "chemical",
        "product_name": "Agrimycin (Streptomycin Sulfate)",
        "active_ingredient": "Streptomycin sulfate 15%",
        "application_method": "Mix 1 g/L water and spray every 5–7 days. Do not apply within 10 days of harvest.",
        "estimated_cost_myr": 60.00,
        "severity_min": "moderate",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Bacterial_spot",
        "treatment_type": "organic",
        "product_name": "Neem Oil (Cold Pressed)",
        "active_ingredient": "Azadirachtin 0.3%",
        "application_method": "Mix 5 mL/L with a few drops of dish soap. Spray weekly as preventive measure.",
        "estimated_cost_myr": 20.00,
        "severity_min": "mild",
        "severity_max": "mild",
    },
    # ── Early Blight ────────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Early_blight",
        "treatment_type": "chemical",
        "product_name": "Dithane M-45 (Mancozeb)",
        "active_ingredient": "Mancozeb 80%",
        "application_method": "Mix 2 g/L water. Apply every 7–14 days. Minimum 3 days pre-harvest interval.",
        "estimated_cost_myr": 25.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Early_blight",
        "treatment_type": "chemical",
        "product_name": "Daconil (Chlorothalonil)",
        "active_ingredient": "Chlorothalonil 72%",
        "application_method": "Dilute 1.5 mL/L water. Spray foliage thoroughly every 7 days.",
        "estimated_cost_myr": 35.00,
        "severity_min": "moderate",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Early_blight",
        "treatment_type": "organic",
        "product_name": "Neem Oil (Cold Pressed)",
        "active_ingredient": "Azadirachtin 0.3%",
        "application_method": "Mix 5 mL/L with emulsifier. Spray weekly as preventive or at first sign of disease.",
        "estimated_cost_myr": 20.00,
        "severity_min": "mild",
        "severity_max": "moderate",
    },
    # ── Late Blight ─────────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Late_blight",
        "treatment_type": "chemical",
        "product_name": "Ridomil Gold MZ (Metalaxyl-M + Mancozeb)",
        "active_ingredient": "Metalaxyl-M 4% + Mancozeb 64%",
        "application_method": "Mix 2.5 g/L water. Apply immediately at first symptom every 7–14 days. Critical: apply before rain.",
        "estimated_cost_myr": 80.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Late_blight",
        "treatment_type": "chemical",
        "product_name": "Curzate M8 (Cymoxanil + Mancozeb)",
        "active_ingredient": "Cymoxanil 8% + Mancozeb 64%",
        "application_method": "Mix 2 g/L water. Apply every 5–7 days during high-risk periods. Alternate with Ridomil to prevent resistance.",
        "estimated_cost_myr": 55.00,
        "severity_min": "moderate",
        "severity_max": "severe",
    },
    # ── Leaf Mold ───────────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Leaf_Mold",
        "treatment_type": "chemical",
        "product_name": "Amistar (Azoxystrobin)",
        "active_ingredient": "Azoxystrobin 25%",
        "application_method": "Mix 0.8 mL/L water. Apply every 7–10 days. Highly effective against fungal leaf diseases.",
        "estimated_cost_myr": 120.00,
        "severity_min": "moderate",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Leaf_Mold",
        "treatment_type": "chemical",
        "product_name": "Daconil (Chlorothalonil)",
        "active_ingredient": "Chlorothalonil 72%",
        "application_method": "Dilute 1.5 mL/L. Spray underside of leaves where mold develops. Repeat weekly.",
        "estimated_cost_myr": 35.00,
        "severity_min": "mild",
        "severity_max": "moderate",
    },
    {
        "disease_label": "Tomato___Leaf_Mold",
        "treatment_type": "organic",
        "product_name": "Potassium Bicarbonate Spray",
        "active_ingredient": "Potassium bicarbonate 85%",
        "application_method": "Dissolve 5 g/L with a wetting agent. Spray foliage and leaf undersides twice weekly.",
        "estimated_cost_myr": 15.00,
        "severity_min": "mild",
        "severity_max": "mild",
    },
    # ── Septoria Leaf Spot ──────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Septoria_leaf_spot",
        "treatment_type": "chemical",
        "product_name": "Dithane M-45 (Mancozeb)",
        "active_ingredient": "Mancozeb 80%",
        "application_method": "Mix 2 g/L water. Apply every 7–10 days starting at first lesion appearance.",
        "estimated_cost_myr": 25.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Septoria_leaf_spot",
        "treatment_type": "chemical",
        "product_name": "Copper Oxychloride 50% WP",
        "active_ingredient": "Copper oxychloride 50%",
        "application_method": "Mix 3 g/L water. Spray every 7 days. Provides both preventive and curative action.",
        "estimated_cost_myr": 30.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    # ── Spider Mites ────────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Spider_mites Two-spotted_spider_mite",
        "treatment_type": "chemical",
        "product_name": "Biomite (Abamectin)",
        "active_ingredient": "Abamectin 1.8% EC",
        "application_method": "Dilute 0.5 mL/L water. Spray leaf undersides thoroughly. Do not apply more than twice per season to avoid resistance.",
        "estimated_cost_myr": 80.00,
        "severity_min": "moderate",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Spider_mites Two-spotted_spider_mite",
        "treatment_type": "chemical",
        "product_name": "Oberon (Spiromesifen)",
        "active_ingredient": "Spiromesifen 22.9% SC",
        "application_method": "Mix 0.5 mL/L water. Highly effective against all mite life stages. Apply when mites are first detected.",
        "estimated_cost_myr": 100.00,
        "severity_min": "moderate",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Spider_mites Two-spotted_spider_mite",
        "treatment_type": "organic",
        "product_name": "Neem Oil (Cold Pressed)",
        "active_ingredient": "Azadirachtin 0.3%",
        "application_method": "Mix 5 mL/L with soap emulsifier. Spray all leaf surfaces every 3–5 days. Effective for mild infestations.",
        "estimated_cost_myr": 20.00,
        "severity_min": "mild",
        "severity_max": "moderate",
    },
    # ── Target Spot ─────────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Target_Spot",
        "treatment_type": "chemical",
        "product_name": "Amistar (Azoxystrobin)",
        "active_ingredient": "Azoxystrobin 25%",
        "application_method": "Mix 0.8 mL/L water. Apply every 7–10 days. Rotate with different fungicide classes.",
        "estimated_cost_myr": 120.00,
        "severity_min": "moderate",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Target_Spot",
        "treatment_type": "chemical",
        "product_name": "Folicur (Tebuconazole)",
        "active_ingredient": "Tebuconazole 25% EC",
        "application_method": "Dilute 0.5 mL/L water. Apply at first sign of infection and repeat every 14 days.",
        "estimated_cost_myr": 60.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    # ── TYLCV ───────────────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "treatment_type": "chemical",
        "product_name": "Confidor (Imidacloprid) — for whitefly control",
        "active_ingredient": "Imidacloprid 20% SL",
        "application_method": "Soil drench or foliar spray at 0.5 mL/L. Controls whitefly vector. Apply weekly.",
        "estimated_cost_myr": 40.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    {
        "disease_label": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "treatment_type": "organic",
        "product_name": "Reflective Silver Mulch",
        "active_ingredient": None,
        "application_method": "Lay reflective mulch on soil around plants to repel whiteflies. Most effective as prevention before symptoms appear.",
        "estimated_cost_myr": 8.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    # ── Tomato Mosaic Virus ─────────────────────────────────────────────────
    {
        "disease_label": "Tomato___Tomato_mosaic_virus",
        "treatment_type": "organic",
        "product_name": "Mineral Oil Spray (Preventive)",
        "active_ingredient": "Petroleum-based mineral oil 98%",
        "application_method": "Mix 10 mL/L water. Spray as a preventive coating to reduce aphid feeding and virus transmission. Not curative.",
        "estimated_cost_myr": 25.00,
        "severity_min": "mild",
        "severity_max": "mild",
    },
    {
        "disease_label": "Tomato___Tomato_mosaic_virus",
        "treatment_type": "organic",
        "product_name": "Skim Milk Spray",
        "active_ingredient": "Non-fat dry milk",
        "application_method": "Mix 10% skim milk solution. Spray on tools and plant surfaces to inactivate virus particles. Apply before handling plants.",
        "estimated_cost_myr": 5.00,
        "severity_min": "mild",
        "severity_max": "severe",
    },
    # ── Healthy ─────────────────────────────────────────────────────────────
    {
        "disease_label": "Tomato___healthy",
        "treatment_type": "organic",
        "product_name": "Preventive Neem Oil Spray",
        "active_ingredient": "Azadirachtin 0.3%",
        "application_method": "Spray monthly as a preventive measure to deter pests and fungal spores. Mix 5 mL/L with emulsifier.",
        "estimated_cost_myr": 20.00,
        "severity_min": "mild",
        "severity_max": "mild",
    },
]


async def seed():
    engine = create_async_engine(settings.database_url, echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with SessionLocal() as session:
        from sqlalchemy import select
        existing = await session.execute(select(TreatmentOptionRecord).limit(1))
        if existing.scalar_one_or_none() is not None:
            print("treatment_options already seeded — skipping.")
            await engine.dispose()
            return

        for t in TREATMENTS:
            record = TreatmentOptionRecord(id=uuid.uuid4(), **t)
            session.add(record)

        await session.commit()
        print(f"Seeded {len(TREATMENTS)} treatment options.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
