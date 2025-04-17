
import hashlib

def seq_encoder(sequence, method="md5"):
    hasher = eval(f"hashlib.{method}")
    return hasher(sequence.encode(encoding="utf-8")).hexdigest()
