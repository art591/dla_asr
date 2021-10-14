import Levenshtein
import jiwer

# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    return min(1, Levenshtein.distance(target_text, predicted_text) / (len(predicted_text) + 1e-7))


def calc_wer(target_text, predicted_text) -> float:
    return jiwer.wer(target_text, predicted_text)
