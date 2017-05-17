set -e

if [ -z "$1" ]; then
  echo "usage download_and_preproces_mscoco.sh [data dir]"
  exit
fi

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
}

cd ${SCRATCH_DIR}

# Download the images.
BASE_IMAGE_URL="http://msvocds.blob.core.windows.net/coco2014"

TRAIN_IMAGE_FILE="train2014.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TRAIN_IMAGE_FILE}
TRAIN_IMAGE_DIR="${SCRATCH_DIR}/train2014"

VAL_IMAGE_FILE="val2014.zip"
download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE}
VAL_IMAGE_DIR="${SCRATCH_DIR}/val2014"

# Download the captions.
BASE_CAPTIONS_URL="http://msvocds.blob.core.windows.net/annotations-1-0-3"
CAPTIONS_FILE="captions_train-val2014.zip"
download_and_unzip ${BASE_CAPTIONS_URL} ${CAPTIONS_FILE}
TRAIN_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/captions_train2014.json"
VAL_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/captions_val2014.json"
