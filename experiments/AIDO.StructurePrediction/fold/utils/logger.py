
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


# -*-coding:utf-8-*-

import logging.config


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


@singleton
class Logger:
    def __init__(self):
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d] %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("root")
