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
  target_group_arn      = module.alb.rag_server_target_group_arn
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
    module.langfuse.langfuse_secret_key_secret_arn,
    module.langfuse.langfuse_public_key_secret_arn,
    aws_secretsmanager_secret.openai_api_key.arn,
  ]

  secrets_env_vars = [
    {
      name      = "LANGFUSE_SECRET_KEY"
      valueFrom = module.langfuse.langfuse_secret_key_secret_arn
    },
    {
      name      = "LANGFUSE_PUBLIC_KEY"
      valueFrom = module.langfuse.langfuse_public_key_secret_arn
    },
    {
      name      = "OPENAI_API_KEY"
      valueFrom = aws_secretsmanager_secret.openai_api_key.arn
    }
  ]

  environment_variables = concat([
    {
      name  = "LANGFUSE_BASE_URL"
      value = module.langfuse.langfuse_url
    },
    {
      name  = "NZAMBE_INDEX__S3_VECTORS_BUCKET_NAME"
      value = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_name
    },
    {
      name  = "NZAMBE_INDEX__S3_VECTORS_INDEX_NAME"
      value = aws_s3vectors_index.vector_index.index_name
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
  s3vectors_index_arn        = aws_s3vectors_index.vector_index.index_arn
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

  metadata_configuration {
    non_filterable_metadata_keys = ["_node_content"]
  }

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
  s3vectors_index_name         = aws_s3vectors_index.vector_index.index_name
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

#####################################
# Langfuse Target Group (for ALB)
#####################################
resource "aws_lb_target_group" "langfuse" {
  name        = "${var.project_name}-${var.environment}-langfuse-tg"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip" # Required for Fargate

  health_check {
    enabled             = true
    path                = "/api/public/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  deregistration_delay = 30

  tags = {
    Name        = "${var.project_name}-${var.environment}-langfuse-tg"
    Environment = var.environment
  }
}

# Listener rule to route RAG server traffic (higher priority)
resource "aws_lb_listener_rule" "rag_server" {
  listener_arn = module.alb.listener_arn
  priority     = 50

  action {
    type             = "forward"
    target_group_arn = module.alb.rag_server_target_group_arn
  }

  condition {
    path_pattern {
      values = ["/query", "/retrieve_docs", "/health"]
    }
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-rag-rule"
    Environment = var.environment
  }
}

# Listener rule to route all other traffic to Langfuse (lower priority as catch-all)
resource "aws_lb_listener_rule" "langfuse" {
  listener_arn = module.alb.listener_arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.langfuse.arn
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-langfuse-rule"
    Environment = var.environment
  }
}

#####################################
# Langfuse Module
#####################################
module "langfuse" {
  source = "../../modules/langfuse"

  name_prefix               = "${var.project_name}-${var.environment}"
  environment               = var.environment
  aws_region                = var.aws_region
  vpc_id                    = module.vpc.vpc_id
  private_subnet_ids        = module.vpc.private_subnet_ids
  ecs_cluster_id            = module.ecs.cluster_id
  alb_security_group_id     = module.alb.alb_security_group_id
  langfuse_target_group_arn = aws_lb_target_group.langfuse.arn
  langfuse_url              = "http://${module.alb.alb_dns_name}"

  # Database configuration
  database_name       = var.langfuse_database_name
  database_username   = var.langfuse_database_username
  aurora_min_capacity = var.langfuse_aurora_min_capacity
  aurora_max_capacity = var.langfuse_aurora_max_capacity
  skip_final_snapshot = var.langfuse_skip_final_snapshot

  # Redis configuration
  redis_node_type = var.langfuse_redis_node_type

  # ECS configuration
  langfuse_web_desired_count    = var.langfuse_web_desired_count
  langfuse_worker_desired_count = var.langfuse_worker_desired_count
  log_retention_days            = var.log_retention_days

  # Langfuse initialization
  langfuse_init_org_id       = var.langfuse_init_org_id
  langfuse_init_org_name     = var.langfuse_init_org_name
  langfuse_init_project_id   = var.langfuse_init_project_id
  langfuse_init_project_name = var.langfuse_init_project_name
  langfuse_init_user_email   = var.langfuse_init_user_email
}
