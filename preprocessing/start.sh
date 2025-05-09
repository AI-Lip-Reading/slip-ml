# get arguments
notebook_name=$1 # this will also be the name of the docker image as well as the ecr repo name
acct_num=438465160412
aws_region="us-east-1" # Change this to your desired region

# convert notebook to script
jupyter nbconvert --to script $notebook_name.ipynb --output the_script

# get the required packages
pipenv requirements > requirements.txt

# build the docker image
docker build -t $notebook_name:latest .

# get the AWS credentials
aws ecr get-login-password --region $aws_region | docker login --username AWS --password-stdin $acct_num.dkr.ecr.$aws_region.amazonaws.com

# create repo if it doesn't exist
# Check if the repository already exists
if aws ecr describe-repositories --repository-names "$notebook_name" --region "$aws_region" > /dev/null 2>&1; then
  echo "ECR repository '$notebook_name' already exists in $aws_region."
else
  # If it doesn't exist, create it
  echo "Creating ECR repository '$notebook_name' in $aws_region..."
  aws ecr create-repository --repository-name "$notebook_name" --region "$aws_region"
  if [ $? -eq 0 ]; then
    echo "ECR repository '$notebook_name' created successfully."
  else
    echo "Error creating ECR repository '$notebook_name'."
  fi
fi

# push the docker image to ecr
docker push $acct_num.dkr.ecr.$aws_region.amazonaws.com/$notebook_name:latest


aws ecs register-task-definition \
  --family $notebook_name \
  --requires-compatibilities EC2 \
  --cpu "16384" \
  --memory "32768" \
  --task-role-arn arn:aws:iam::$acct_num:role/BatchEcsTaskExecutionRole \
  --execution-role-arn arn:aws:iam::$acct_num:role/BatchEcsTaskExecutionRole \
  --container-definitions "[{
    \"name\": \"$notebook_name\",
    \"image\": \"$acct_num.dkr.ecr.$aws_region.amazonaws.com/$notebook_name\",
    \"resourceRequirements\": [{
      \"type\": \"GPU\",
      \"value\": \"1\"
    }],
    \"essential\": true
  }]" \
  --network-mode bridge

