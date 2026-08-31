from pathlib import Path


def test_four_model_projects_are_included() -> None:
    sources = list((Path(__file__).parents[1] / "src").glob("*.py"))
    assert len(sources) == 4


def test_every_project_reports_an_out_of_sample_metric() -> None:
    sources = (Path(__file__).parents[1] / "src").glob("*.py")
    for source in sources:
        text = source.read_text()
        assert "test_" in text, source.name
        assert "randomSplit" in text or "--test" in text, source.name

