from confidence_tom.generator.generator import SubjectGenerator
from confidence_tom.generator.models import SubjectOutputV2
from confidence_tom.observer.models import RecursiveLevelResult
from confidence_tom.observer.protocols import build_protocol_context


def test_compatibility_imports_work() -> None:
    assert SubjectGenerator.__name__ == "SubjectGenerator"
    assert SubjectOutputV2.__name__ == "SubjectOutputV2"
    assert RecursiveLevelResult.__name__ == "RecursiveLevelResult"
    assert callable(build_protocol_context)
