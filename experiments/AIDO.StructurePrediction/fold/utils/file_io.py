# Copyright 2025 GenBio AI
# Copyright 2024 ByteDance and/or its affiliates.
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


import json
from pathlib import Path
from typing import Any, Union

from fold.utils.torch_utils import map_values_to_list


def save_json(data: dict, output_fpath: Union[str, Path], indent: int = 4):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        output_fpath (Union[str, Path]): The output file path.
        indent (int, optional): The indentation level for the JSON file. Defaults to 4.
    """
    data_json = data.copy()
    data_json = map_values_to_list(data_json)
    with open(output_fpath, "w") as f:
        if indent is not None:
            json.dump(data_json, f, indent=indent)
        else:
            json.dump(data_json, f)
