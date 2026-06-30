"""Confidence and severity policy — pure domain logic, no framework imports."""


class ConfidencePolicy:
    LOW_CONFIDENCE_THRESHOLD = 60.0
    HIGH_CONFIDENCE_THRESHOLD = 85.0

    @staticmethod
    def should_show_alternatives(confidence_pct: float) -> bool:
        return confidence_pct < ConfidencePolicy.LOW_CONFIDENCE_THRESHOLD

    @staticmethod
    def get_certainty_label(confidence_pct: float) -> str:
        if confidence_pct >= ConfidencePolicy.HIGH_CONFIDENCE_THRESHOLD:
            return "high_certainty"
        elif confidence_pct >= ConfidencePolicy.LOW_CONFIDENCE_THRESHOLD:
            return "moderate_certainty"
        return "low_certainty_seek_confirmation"

    @staticmethod
    def classify_severity(affected_area_ratio: float) -> str:
        if affected_area_ratio < 0.15:
            return "mild"
        elif affected_area_ratio < 0.40:
            return "moderate"
        return "severe"
