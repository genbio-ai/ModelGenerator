

import hashlib


def hash_seq(seq, method='md5'):
    """
    hash the string sequence
    :param seq:
    :param method:
    :return:
    """
    if method == "md5":
        hasher = hashlib.md5
    else:
        raise NotImplementedError
    code = hasher(seq.encode(encoding='utf-8')).hexdigest()

    return code
