#!/bin/bash
#
# Copyright 2021 AlQuraishi Laboratory
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
#
# Downloads and unzips the BFD database for AlphaFold.
#
# Usage: bash download_bfd.sh /path/to/download/directory
# not that, the download uniref30 cannot be used by mmseqs2. if one can download uniref30 from
# http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz, after unzip this download file, use the follow
# script to convert data format.
# mmseqs tsv2exprofiledb uniref30_2103 uniref30_2103_db
# mmseqs createindex uniref30_2103_db tmp
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/uniref30"
SOURCE_URL="http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz"
BASENAME=$(basename "${SOURCE_URL}")
MAX_CONNECTIONS="${2:-4}"

mkdir -p "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}" -x "${MAX_CONNECTIONS}" --check-certificate=false
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
  --directory="${ROOT_DIR}"
rm "${ROOT_DIR}/${BASENAME}"
