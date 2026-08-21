import pytest

from interview_coach.nlp.label_mapping import normalize_label


def test_known_label_is_explicitly_mapped():
    assert normalize_label("fear") == "uncertain_hesitant"


def test_unknown_label_is_not_guessed():
    with pytest.raises(KeyError):
        normalize_label("probably_confident-ish")
