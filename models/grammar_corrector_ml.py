from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "vennify/t5-base-grammar-correction"

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _tokenizer, _model


def correct_grammar_ml(text: str) -> str:
    if not text.strip():
        return ""

    tokenizer, model = load_model()

    input_text = f"grammar: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    outputs = model.generate(
        inputs,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
