"""PredictDiseaseUseCase — orchestrates the full prediction pipeline."""
import uuid
from PIL import Image
import io

from application.dtos.prediction_response import (
    AlternativeLabelDTO,
    PredictionResponse,
    TreatmentOptionDTO,
)
from domain.entities.prediction import Prediction
from domain.repositories.prediction_repository import PredictionRepository
from domain.repositories.treatment_repository import TreatmentRepository
from domain.services.confidence_policy import ConfidencePolicy
from domain.services.disease_service import DiseaseService
from infrastructure.ml.lesion_segmenter import LesionSegmenter
from infrastructure.ml.resnet34_inferencer import ResNet34Inferencer
from infrastructure.storage.image_store import ImageStore


class PredictDiseaseUseCase:
    def __init__(
        self,
        inferencer: ResNet34Inferencer,
        segmenter: LesionSegmenter,
        disease_service: DiseaseService,
        confidence_policy: ConfidencePolicy,
        prediction_repo: PredictionRepository,
        treatment_repo: TreatmentRepository,
        image_store: ImageStore,
    ):
        self._inferencer = inferencer
        self._segmenter = segmenter
        self._disease_service = disease_service
        self._confidence_policy = confidence_policy
        self._prediction_repo = prediction_repo
        self._treatment_repo = treatment_repo
        self._image_store = image_store

    async def execute(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        session_id: str | None = None,
    ) -> PredictionResponse:
        # 1. Run inference (TTA)
        label, confidence, alternatives = self._inferencer.predict(image_bytes)

        # 2. Severity estimation via lesion segmentation
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        affected_ratio = self._segmenter.estimate_affected_ratio(pil_image)
        severity_level = self._confidence_policy.classify_severity(affected_ratio)

        # 3. Confidence-aware fields
        is_low_confidence = self._confidence_policy.should_show_alternatives(confidence)
        certainty_label = self._confidence_policy.get_certainty_label(confidence)

        # 4. Treatment advice
        disease = self._disease_service.get_disease(label)

        # 5. Treatment options from DB
        treatment_entities = await self._treatment_repo.get_for_disease(label)
        treatment_dtos = [
            TreatmentOptionDTO(
                id=t.id,
                treatment_type=t.treatment_type,
                product_name=t.product_name,
                active_ingredient=t.active_ingredient,
                application_method=t.application_method,
                estimated_cost_myr=t.estimated_cost_myr,
            )
            for t in treatment_entities
        ]

        # 6. Save image
        image_path = self._image_store.save(image_bytes, filename)

        # 7. Persist prediction
        prediction = Prediction(
            prediction_id=str(uuid.uuid4()),
            label=label,
            confidence=confidence,
            advice=disease.treatment_tip,
            image_path=image_path,
            severity_level=severity_level,
            affected_area_ratio=affected_ratio,
            is_low_confidence=is_low_confidence,
            certainty_label=certainty_label,
            session_id=session_id,
        )
        saved = await self._prediction_repo.save(prediction)

        # 8. Build response
        alternative_dtos = (
            [AlternativeLabelDTO(**a) for a in alternatives]
            if is_low_confidence
            else []
        )

        return PredictionResponse(
            prediction_id=saved.prediction_id,
            label=saved.label,
            confidence=saved.confidence,
            advice=saved.advice,
            severity_level=saved.severity_level,
            affected_area_ratio=saved.affected_area_ratio,
            is_low_confidence=saved.is_low_confidence,
            certainty_label=saved.certainty_label,
            alternative_labels=alternative_dtos,
            treatment_options=treatment_dtos,
            timestamp=saved.timestamp.isoformat(),
        )
