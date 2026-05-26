# Kept as a light import smoke test so Stage-1 module discovery is covered.
from partcat_hkg.models.stage1 import PartCATHKGStage1


def test_stage1_class_importable():
    assert PartCATHKGStage1.__name__ == "PartCATHKGStage1"
