#!/bin/bash

# Check if a folder name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <Folder Name> <Bucket Name>"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Usage: $0 <Folder Name> <Bucket Name>"
  exit 1
fi

# Variables
FOLDER_NAME="$1"
ZIP_FILE="${FOLDER_NAME}.zip"
S3_BUCKET="$2"
S3_KEY="lambda/${FOLDER_NAME}/${ZIP_FILE}"
ROLE_ARN="arn:aws:iam::438465160412:role/LambdaExecution"
REGION="us-east-1"
LOG_FILE="${FOLDER_NAME}_lambda_log.txt"

# Move to the directory of the script
cd $FOLDER_NAME

if [[ "$FOLDER_NAME" == *"video"* ]]; then
  echo "Folder name contains 'video'. Downloading FFmpeg..." | tee -a $LOG_FILE

  # Create a bin directory for FFmpeg
  mkdir -p bin

  # Download and extract FFmpeg
  curl -L $FFMPEG_URL -o ffmpeg.tar.xz
  tar -xf ffmpeg.tar.xz --strip-components=1 -C bin
  rm ffmpeg.tar.xz

  # Ensure FFmpeg is executable
  chmod +x bin/ffmpeg bin/ffprobe

  echo "FFmpeg downloaded and added to bin directory." | tee -a $LOG_FILE
fi

pip install -r requirements.txt --target .

# Step 1: Create the ZIP file
echo "Creating deployment package from folder contents: $FOLDER_NAME..." | tee -a $LOG_FILE
zip -r ../$ZIP_FILE . > /dev/null

# Move back to the parent directory
cd ..

# Step 2: Upload the ZIP file to S3
echo "Uploading deployment package to S3..." | tee -a $LOG_FILE
aws s3 cp $ZIP_FILE s3://$S3_BUCKET/$S3_KEY --region $REGION | tee -a $LOG_FILE

# Step 3: Check if the Lambda function exists
echo "Checking if Lambda function $FOLDER_NAME exists..." | tee -a $LOG_FILE
aws lambda get-function --function-name $FOLDER_NAME --region $REGION > /dev/null 2>&1

if [ $? -eq 0 ]; then
  # Lambda function exists, update it
  echo "Lambda function $FOLDER_NAME exists. Updating it..." | tee -a $LOG_FILE
  aws lambda update-function-code \
    --function-name $FOLDER_NAME \
    --s3-bucket $S3_BUCKET \
    --s3-key $S3_KEY \
    --region $REGION | tee -a $LOG_FILE
else
  # Lambda function does not exist, create it
  echo "Lambda function $FOLDER_NAME does not exist. Creating it..." | tee -a $LOG_FILE
  aws lambda create-function \
    --function-name $FOLDER_NAME \
    --runtime python3.11 \
    --role $ROLE_ARN \
    --handler $FOLDER_NAME.lambda_handler \
    --code S3Bucket=$S3_BUCKET,S3Key=$S3_KEY \
    --timeout 120 \
    --memory-size 512 \
    #--layers "arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p311-numpy:14" \
    --region $REGION | tee -a $LOG_FILE
fi

# Cleanup
echo "Cleaning up..."
rm $ZIP_FILE

cd $FOLDER_NAME

find . -type f ! -name "$FOLDER_NAME.py" ! -name "requirements.txt" -delete
find . -type d -empty -delete

cd ..