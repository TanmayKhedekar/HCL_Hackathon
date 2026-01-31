def generate_explanation(prob):
    if prob > 0.85:
        return "Unnatural pitch consistency and robotic speech patterns detected"
    elif prob > 0.65:
        return "Over-smoothed voice texture and lack of micro-pauses detected"
    else:
        return "Natural speech variability and human-like prosody detected"
