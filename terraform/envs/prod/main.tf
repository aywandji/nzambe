provider "aws" {
  region                   = var.aws_region
  shared_config_files      = ["~/.aws/config"]
  shared_credentials_files = ["~/.aws/credentials"]
}

# Data source to get ECR repository URL from global resources
data "aws_ecr_repository" "app" {
  name = "${var.project_name}-ecr-repo"
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

# Note: Secret values must be set manually via AWS Console or CLI
# Example: aws secretsmanager put-secret-value --secret-id <secret-name> --secret-string "your-value"

#####################################
# VPC Module
#####################################
module "vpc" {
  source = "../../modules/vpc"

  name_prefix        = "${var.project_name}-${var.environment}"
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  az_count           = var.az_count
  nat_gateway_count  = var.nat_gateway_count
}

#####################################
# ALB Module
#####################################
module "alb" {
  source = "../../modules/alb"

  name_prefix           = "${var.project_name}-${var.environment}"
  environment           = var.environment
  vpc_id                = module.vpc.vpc_id
  public_subnet_ids     = module.vpc.public_subnet_ids
  target_port           = var.container_port
  health_check_path     = var.health_check_path
  enable_deletion_protection = var.alb_deletion_protection
}

#####################################
# ECS Module
#####################################
module "ecs" {
  source = "../../modules/ecs"

  name_prefix            = "${var.project_name}-${var.environment}"
  environment            = var.environment
  vpc_id                 = module.vpc.vpc_id
  private_subnet_ids     = module.vpc.private_subnet_ids
  alb_security_group_id  = module.alb.alb_security_group_id
  target_group_arn       = module.alb.target_group_arn
  aws_region             = var.aws_region

  container_name         = var.container_name
  container_image        = "${data.aws_ecr_repository.app.repository_url}:${var.image_tag}"
  container_port         = var.container_port

  task_cpu               = var.task_cpu
  task_memory            = var.task_memory
  desired_count          = var.desired_count
  log_retention_days     = var.log_retention_days

  # Secrets configuration
  secrets_arns = [
    aws_secretsmanager_secret.langfuse_secret_key.arn,
    aws_secretsmanager_secret.langfuse_public_key.arn,
    aws_secretsmanager_secret.langfuse_base_url.arn,
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
    }
  ]

  environment_variables = var.additional_env_vars

  # Auto-scaling enabled for production
  enable_autoscaling = true
  min_capacity       = var.min_capacity
  max_capacity       = var.max_capacity

  # Capacity provider - 30% Fargate Spot for prod (balance cost and reliability)
  enable_fargate_spot  = true
  fargate_spot_weight  = 30
}
