provider "aws" {
  region                   = var.aws_region
  shared_config_files      = ["~/.aws/config"]
  shared_credentials_files = ["~/.aws/credentials"]
}

# Data source to get ECR repository URL from global resources
data "aws_ecr_repository" "app" {
  name = "${var.project_name}-ecr-repo"
}

# Data sources to get global S3 buckets
data "aws_s3_bucket" "rag_documents" {
  bucket = "${var.project_name}-rag-documents"
}
#####################################
# Secrets Manager - Application Secrets
#####################################
resource "aws_secretsmanager_secret" "langfuse_secret_key" {
  name        = "${var.project_name}-${var.environment}-langfuse-secret-key"
  description = "Langfuse Secret Key for ${var.environment}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-langfuse-secret-key"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret" "langfuse_public_key" {
  name        = "${var.project_name}-${var.environment}-langfuse-public-key"
  description = "Langfuse Public Key for ${var.environment}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-langfuse-public-key"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret" "langfuse_base_url" {
  name        = "${var.project_name}-${var.environment}-langfuse-base-url"
  description = "Langfuse Base URL for ${var.environment}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-langfuse-base-url"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret" "openai_api_key" {
  name        = "${var.project_name}-${var.environment}-openai-api-key"
  description = "Openai api key for ${var.environment}"

  tags = {
    Name        = "${var.project_name}-${var.environment}-openai-api-key"
    Environment = var.environment
  }
}

# Note: Secret values must be set manually via AWS Console or CLI
# Example: aws secretsmanager put-secret-value --secret-id <secret-name> --secret-string "your-value"

#####################################
# VPC Module
#####################################
module "vpc" {
  source = "../../modules/vpc"

  name_prefix       = "${var.project_name}-${var.environment}"
  environment       = var.environment
  vpc_cidr          = var.vpc_cidr
  az_count          = var.az_count
  nat_gateway_count = var.nat_gateway_count
}

#####################################
# ALB Module
#####################################
module "alb" {
  source = "../../modules/alb"

  name_prefix                = "${var.project_name}-${var.environment}"
  environment                = var.environment
  vpc_id                     = module.vpc.vpc_id
  public_subnet_ids          = module.vpc.public_subnet_ids
  target_port                = var.container_port
  health_check_path          = var.health_check_path
  enable_deletion_protection = var.alb_deletion_protection
}

#####################################
# ECS Module
#####################################
module "ecs" {
  source = "../../modules/ecs"

  name_prefix           = "${var.project_name}-${var.environment}"
  environment           = var.environment
  vpc_id                = module.vpc.vpc_id
  private_subnet_ids    = module.vpc.private_subnet_ids
  alb_security_group_id = module.alb.alb_security_group_id
  target_group_arn      = module.alb.target_group_arn
  aws_region            = var.aws_region

  container_name  = var.container_name
  container_image = "${data.aws_ecr_repository.app.repository_url}:${var.rag_server_image_tag}"
  container_port  = var.container_port

  task_cpu           = var.task_cpu
  task_memory        = var.task_memory
  desired_count      = var.desired_count
  log_retention_days = var.log_retention_days

  # Secrets configuration
  secrets_arns = [
    aws_secretsmanager_secret.langfuse_secret_key.arn,
    aws_secretsmanager_secret.langfuse_public_key.arn,
    aws_secretsmanager_secret.langfuse_base_url.arn,
    aws_secretsmanager_secret.openai_api_key.arn,
  ]

  secrets_env_vars = [
    {
      name      = "LANGFUSE_SECRET_KEY"
      valueFrom = aws_secretsmanager_secret.langfuse_secret_key.arn
    },
    {
      name      = "LANGFUSE_PUBLIC_KEY"
      valueFrom = aws_secretsmanager_secret.langfuse_public_key.arn
    },
    {
      name      = "LANGFUSE_BASE_URL"
      valueFrom = aws_secretsmanager_secret.langfuse_base_url.arn
    },
    {
      name      = "OPENAI_API_KEY"
      valueFrom = aws_secretsmanager_secret.openai_api_key.arn
    }
  ]

  environment_variables = concat([
    {
      name  = "NZAMBE_INDEX__S3_VECTORS_BUCKET_NAME"
      value = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_name
    },
    {
      name  = "NZAMBE_INDEX__S3_VECTORS_INDEX_ARN"
      value = aws_s3vectors_index.vector_index.index_arn
    },
    {
      name  = "NZAMBE_INDEX__S3_VECTORS_INDEX_DATA_TYPE"
      value = aws_s3vectors_index.vector_index.data_type
    },
    {
      name  = "NZAMBE_INDEX__S3_VECTORS_INDEX_DISTANCE_METRIC"
      value = aws_s3vectors_index.vector_index.distance_metric
    },
    {
      name  = "NZAMBE_ENV"
      value = "staging"
    }

  ], var.additional_env_vars)

  # Auto-scaling configuration
  min_capacity = 1
  max_capacity = 3

  # Capacity provider - 100% Fargate Spot for staging (cost savings)
  fargate_spot_weight = 100

  # S3 bucket for vector store access
  s3_vector_store_bucket_arn = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_arn
}


#####################################
# S3 Vectors Bucket for Vector Store
#####################################
resource "aws_s3vectors_vector_bucket" "s3vectors_bucket" {
  vector_bucket_name = "${var.project_name}-${var.environment}-s3-vectors-storage"

  tags = {
    Name        = "${var.project_name}-${var.environment}-s3-vectors-storage"
    Purpose     = "Bucket for Vector index storage"
    environment = var.environment
  }
}

resource "aws_s3vectors_index" "vector_index" {
  index_name         = "${var.project_name}-${var.environment}-s3-vectors-index-${var.vector_index_embedding_model}"
  vector_bucket_name = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_name

  data_type       = var.vector_index_data_type
  dimension       = var.vector_index_dimension
  distance_metric = var.vector_index_distance_metric

  tags = {
    environment = var.environment
  }
}
#####################################
# Lambda Indexer Module
#####################################
module "lambda_indexer" {
  source = "../../modules/lambda_indexer"

  name_prefix                  = "${var.project_name}-${var.environment}"
  environment                  = var.environment
  lambda_image_uri             = "${data.aws_ecr_repository.app.repository_url}:${var.lambda_image_tag}"
  source_bucket_name           = data.aws_s3_bucket.rag_documents.id
  source_bucket_arn            = data.aws_s3_bucket.rag_documents.arn
  vector_store_bucket_name     = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_name
  vector_store_bucket_arn      = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_arn
  s3vectors_index_arn          = aws_s3vectors_index.vector_index.index_arn
  openai_secret_arn            = aws_secretsmanager_secret.openai_api_key.arn
  timeout                      = 300
  memory_size                  = 512
  log_retention_days           = var.log_retention_days
  chunk_size                   = var.vector_index_chunk_size
  chunk_overlap                = var.vector_index_chunk_overlap
  embedding_model              = var.vector_index_embedding_model
  vector_index_data_type       = var.vector_index_data_type
  vector_index_distance_metric = var.vector_index_distance_metric
}
