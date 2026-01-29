# Dockerfile for AWS Lambda - Vector Indexing Function

FROM public.ecr.aws/lambda/python:3.12

# Copy dependency files
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Lambda function code
COPY handler.py ${LAMBDA_TASK_ROOT}

# Set the Lambda handler
CMD ["handler.lambda_handler"]
