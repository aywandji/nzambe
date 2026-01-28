# Dockerfile for AWS Lambda - Vector Indexing Function
# Optimized for Lambda runtime with LlamaIndex and OpenAI dependencies

FROM public.ecr.aws/lambda/python:3.11

# Install uv for faster package installation
#COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy dependency files
COPY lambda/s3-indexer/requirements.txt .

# Install Python dependencies
RUn pip install --no-cache-dir -r requirements.txt

# Copy Lambda function code
COPY lambda/s3-indexer/handler.py .

# Set the Lambda handler
CMD ["handler.lambda_handler"]
