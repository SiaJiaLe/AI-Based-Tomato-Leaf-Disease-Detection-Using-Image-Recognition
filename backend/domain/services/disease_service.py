"""Domain service — maps disease labels to treatment advice."""
from domain.entities.disease import Disease


_DISEASE_ADVICE: dict[str, Disease] = {
    "Tomato___Bacterial_spot": Disease(
        name="Bacterial Spot",
        severity_hint="Causes water-soaked lesions that turn dark and scabby. Spreads rapidly in wet conditions.",
        treatment_tip=(
            "Apply copper-based bactericide (e.g., copper hydroxide) every 7–10 days. "
            "Remove and destroy infected leaves. Avoid overhead irrigation. "
            "Ensure good air circulation between plants."
        ),
    ),
    "Tomato___Early_blight": Disease(
        name="Early Blight",
        severity_hint="Dark concentric ring lesions on lower leaves first. Accelerates in warm, humid weather.",
        treatment_tip=(
            "Apply mancozeb or chlorothalonil fungicide on a 7–14 day schedule. "
            "Remove affected lower leaves. Rotate crops each season. "
            "Mulch soil to prevent spore splash-up from soil."
        ),
    ),
    "Tomato___Late_blight": Disease(
        name="Late Blight",
        severity_hint="Fast-spreading water-soaked lesions with white mold on leaf undersides. Can destroy a crop in days.",
        treatment_tip=(
            "Apply metalaxyl-M (e.g., Ridomil Gold) or cymoxanil immediately. "
            "Destroy all infected plant material — do not compost. "
            "Improve field drainage. Scout neighboring plants urgently as spread is rapid."
        ),
    ),
    "Tomato___Leaf_Mold": Disease(
        name="Leaf Mold",
        severity_hint="Yellow patches on upper leaf surface with olive-gray mold on underside. Thrives in high humidity.",
        treatment_tip=(
            "Apply azoxystrobin or chlorothalonil fungicide. "
            "Reduce greenhouse humidity by improving ventilation. "
            "Avoid wetting foliage when watering. Space plants wider for airflow."
        ),
    ),
    "Tomato___Septoria_leaf_spot": Disease(
        name="Septoria Leaf Spot",
        severity_hint="Small circular spots with dark borders and light centers on lower leaves. Severe defoliation if unchecked.",
        treatment_tip=(
            "Apply mancozeb or copper oxychloride fungicide on a 7–10 day schedule. "
            "Remove infected lower foliage promptly. "
            "Mulch soil surface to reduce spore splash. Rotate tomatoes with non-solanaceous crops."
        ),
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": Disease(
        name="Spider Mites (Two-Spotted)",
        severity_hint="Tiny stippling on leaf surface; fine webbing visible underneath. Severe infestation causes rapid leaf bronzing.",
        treatment_tip=(
            "Apply abamectin miticide (e.g., Biomite) or spiromesifen. "
            "Neem oil is an organic alternative for mild infestations. "
            "Increase humidity — mites thrive in dry conditions. "
            "Introduce predatory mites (Phytoseiidae) for biological control."
        ),
    ),
    "Tomato___Target_Spot": Disease(
        name="Target Spot",
        severity_hint="Circular brown lesions with concentric rings resembling a target. Affects leaves, stems, and fruit.",
        treatment_tip=(
            "Apply azoxystrobin or tebuconazole fungicide. "
            "Remove severely infected plant debris from the field. "
            "Improve air circulation and avoid overhead irrigation."
        ),
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": Disease(
        name="Tomato Yellow Leaf Curl Virus (TYLCV)",
        severity_hint="Upward curling and yellowing of young leaves; stunted growth. Transmitted by whitefly — no chemical cure.",
        treatment_tip=(
            "There is no cure — remove and destroy infected plants immediately to limit spread. "
            "Control whitefly vector with imidacloprid or reflective silver mulch. "
            "Use virus-resistant tomato varieties for future planting. "
            "Install insect-proof netting in greenhouse settings."
        ),
    ),
    "Tomato___Tomato_mosaic_virus": Disease(
        name="Tomato Mosaic Virus (ToMV)",
        severity_hint="Mosaic pattern of light and dark green patches on leaves; leaf distortion and stunting. Spreads via contact.",
        treatment_tip=(
            "There is no cure — remove and destroy infected plants. "
            "Sanitize all tools with bleach solution (10%) between plants. "
            "Control aphid vectors. Wash hands before handling plants. "
            "Use certified virus-free seed or resistant varieties for future crops."
        ),
    ),
    "Tomato___healthy": Disease(
        name="Healthy",
        severity_hint="No disease symptoms detected.",
        treatment_tip=(
            "No treatment needed. Continue regular monitoring every 7 days. "
            "Maintain good cultural practices: adequate spacing, consistent watering, "
            "balanced fertilization, and prompt removal of dead or damaged leaves."
        ),
    ),
}


class DiseaseService:
    def get_disease(self, label: str) -> Disease:
        return _DISEASE_ADVICE.get(
            label,
            Disease(
                name=label,
                severity_hint="Unknown disease detected.",
                treatment_tip="Consult an agricultural extension officer for advice.",
            ),
        )

    def get_advice_for_label(self, label: str) -> str:
        return self.get_disease(label).treatment_tip
