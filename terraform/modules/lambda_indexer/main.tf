#####################################
# Lambda Indexer Module
# Creates Lambda function for processing S3 uploads and building vector indices
#####################################

#####################################
# CloudWatch Log Group
#####################################
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.name_prefix}-indexer"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.name_prefix}-lambda-logs"
    Environment = var.environment
  }
}

#####################################
# IAM Role - Lambda Execution
#####################################
resource "aws_iam_role" "lambda_execution" {
  name = "${var.name_prefix}-lambda-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.name_prefix}-lambda-execution-role"
    Environment = var.environment
  }
}

# Attach basic Lambda execution policy
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# S3 and S3 Vectors access policy for Lambda
resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "${var.name_prefix}-lambda-s3-access"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { # source bucket policy
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          var.source_bucket_arn,
          "${var.source_bucket_arn}/*"
        ]
      },
      { # s3vectors bucket and index policy
        Effect = "Allow"
        Action = [
          "s3vectors:QueryVectors",
          "s3vectors:DeleteVectors",
          "s3vectors:GetVectors",
          "s3vectors:PutVectors",
          "s3vectors:ListVectors",
          "s3vectors:GetIndex"
        ]
        Resource = [
          var.vector_store_bucket_arn,
          "${var.vector_store_bucket_arn}/*",
          var.s3vectors_index_name,
          var.s3vectors_index_arn
        ]
      }
    ]
  })
}

# Secrets Manager access for OpenAI API key
resource "aws_iam_role_policy" "lambda_secrets_access" {
  name = "${var.name_prefix}-lambda-secrets-access"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [var.openai_secret_arn]
      }
    ]
  })
}

#####################################
# Lambda Function
#####################################
resource "aws_lambda_function" "indexer" {
  function_name = "${var.name_prefix}-indexer"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = var.lambda_image_uri
  timeout       = var.timeout
  memory_size   = var.memory_size

  environment {
    variables = {
      S3_VECTORS_BUCKET_NAME       = var.vector_store_bucket_name
      S3_VECTORS_INDEX_NAME        = var.s3vectors_index_name
      OPENAI_SECRET_ARN            = var.openai_secret_arn
      CHUNK_SIZE                   = var.chunk_size
      CHUNK_OVERLAP                = var.chunk_overlap
      EMBEDDING_MODEL              = var.embedding_model
      VECTOR_INDEX_DISTANCE_METRIC = var.vector_index_distance_metric
      VECTOR_INDEX_DATA_TYPE       = var.vector_index_data_type
    }
  }

  tags = {
    Name        = "${var.name_prefix}-indexer"
    Environment = var.environment
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy_attachment.lambda_basic_execution,
    aws_iam_role_policy.lambda_s3_access,
    aws_iam_role_policy.lambda_secrets_access
  ]
}

resource "null_resource" "sam_metadata_lambda_indexer" {
  triggers = {
    resource_name     = "aws_lambda_function.indexer"
    resource_type     = "IMAGE_LAMBDA_FUNCTION"
    docker_context    = "/home/arnaud/projects/nzambe/lambda/s3-indexer"
    docker_file       = "Dockerfile"
    docker_tag        = "local"
    docker_build_args = jsonencode({})
  }
}

#####################################
# Lambda Permission for S3
#####################################
resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.indexer.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.source_bucket_arn
}

#####################################
# S3 Bucket Notification
#####################################
resource "aws_s3_bucket_notification" "source_bucket" {
  bucket = var.source_bucket_name

  lambda_function {
    lambda_function_arn = aws_lambda_function.indexer.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".txt"
  }

  depends_on = [aws_lambda_permission.allow_s3]
}
