from text2table.text2table import Text2Table


def test_build_prompt_includes_labels_and_entities():
    extractor = Text2Table(labels=["Drug", "Drug dosage"])
    entities = [
        {"label": "Drug", "text": "R13", "score": 0.98},
        {"label": "Drug dosage", "text": "36 mg/kg", "score": 0.91},
    ]
    prompt = extractor.build_prompt("Example text body.", entities)

    assert "Drug, Drug dosage" in prompt
    assert "R13" in prompt
    assert "36 mg/kg" in prompt
    assert "Example text body." in prompt


def test_build_prompt_without_gliner_mentions_inference():
    extractor = Text2Table(labels=["Name", "Age"], use_gliner=False)
    prompt = extractor.build_prompt("Some source text", entities=[])

    assert "Entity extraction disabled" in prompt
    assert "Some source text" in prompt
