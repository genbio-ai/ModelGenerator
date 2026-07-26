# Copyright 2025 GenBio AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
