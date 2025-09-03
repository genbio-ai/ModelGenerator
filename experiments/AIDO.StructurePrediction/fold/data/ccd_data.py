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

from pathlib import Path

COMPONENTS_FILE = None
RKDIT_MOL_PKL = None

def set_components_file(components_file):
    global COMPONENTS_FILE
    COMPONENTS_FILE = components_file

def set_rkdit_mol_pkl(rkdit_mol_pkl):
    global RKDIT_MOL_PKL
    RKDIT_MOL_PKL = Path(rkdit_mol_pkl)


def get_components_file():
    global COMPONENTS_FILE
    return COMPONENTS_FILE

def get_rkdit_mol_pkl():
    global RKDIT_MOL_PKL
    return RKDIT_MOL_PKL
