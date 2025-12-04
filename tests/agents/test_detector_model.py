"""Tests for the detector_model module."""
import pytest
from pydantic import ValidationError

from agents.models.detector_model import DetectorModel


class TestDetectorModel:
    """Test cases for the DetectorModel."""

    def test_detector_model_creation(self) -> None:
        """Test that DetectorModel can be created with valid data."""
        model = DetectorModel(
            label="fake",
            explanation="This claim contains misinformation.",
        )

        assert model.label == "fake"
        assert model.explanation == "This claim contains misinformation."

    def test_detector_model_validation(self) -> None:
        """Test that DetectorModel validates required fields."""
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            DetectorModel()

    def test_detector_model_dict_conversion(self) -> None:
        """Test that DetectorModel can be converted to dict."""
        model = DetectorModel(
            label="real",
            explanation="This claim is supported by evidence.",
        )

        model_dict = model.model_dump()

        assert model_dict["label"] == "real"
        assert model_dict["explanation"] == "This claim is supported by evidence."

    def test_detector_model_json_conversion(self) -> None:
        """Test that DetectorModel can be converted to JSON."""
        model = DetectorModel(
            label="fake",
            explanation="Test explanation",
        )

        json_str = model.model_dump_json()

        assert "fake" in json_str
        assert "Test explanation" in json_str

    def test_detector_model_from_dict(self) -> None:
        """Test that DetectorModel can be created from dict."""
        data = {
            "label": "real",
            "explanation": "Valid information",
        }

        model = DetectorModel(**data)

        assert model.label == "real"
        assert model.explanation == "Valid information"
