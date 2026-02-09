output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.alb_dns_name
}

output "alb_url" {
  description = "URL of the Application Load Balancer"
  value       = "http://${module.alb.alb_dns_name}"
}

output "ecs_cluster_name" {
  description = "ECS Cluster name"
  value       = module.ecs.cluster_name
}

output "ecs_service_name" {
  description = "ECS Service name"
  value       = module.ecs.service_name
}

output "ecr_repository_url" {
  description = "ECR Repository URL"
  value       = data.aws_ecr_repository.app.repository_url
}

output "cloudwatch_log_group" {
  description = "CloudWatch Log Group name"
  value       = module.ecs.log_group_name
}

output "secrets_arns" {
  sensitive   = true
  description = "ARNs of the secrets"
  value = {
    langfuse_secret_key = module.langfuse.langfuse_secret_key_secret_arn
    langfuse_public_key = module.langfuse.langfuse_public_key_secret_arn
    openai_api_key      = aws_secretsmanager_secret.openai_api_key.arn
  }
}

output "current_rag_server_image_tag" {
  description = "Currently deployed image tag (from last terraform apply)"
  value       = var.rag_server_image_tag
}

output "current_lambda_image_tag" {
  description = "Currently deployed image tag (from last terraform apply)"
  value       = var.lambda_image_tag
}

output "vector_store_bucket_name" {
  description = "Name of the S3 vectors bucket for vector index storage"
  value       = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_name
}

output "vector_store_bucket_arn" {
  description = "ARN of the S3 vectors bucket for vector index storage"
  value       = aws_s3vectors_vector_bucket.s3vectors_bucket.vector_bucket_arn
}

#####################################
# Langfuse Outputs
#####################################
output "langfuse_url" {
  description = "URL to access Langfuse UI"
  value       = module.langfuse.langfuse_url
}

output "langfuse_web_service_name" {
  description = "Name of the Langfuse Web ECS service"
  value       = module.langfuse.langfuse_web_service_name
}

output "langfuse_worker_service_name" {
  description = "Name of the Langfuse Worker ECS service"
  value       = module.langfuse.langfuse_worker_service_name
}

output "langfuse_clickhouse_service_name" {
  description = "Name of the ClickHouse ECS service"
  value       = module.langfuse.clickhouse_service_name
}

output "langfuse_database_endpoint" {
  description = "Langfuse Aurora PostgreSQL endpoint"
  value       = module.langfuse.database_endpoint
}

output "langfuse_redis_endpoint" {
  description = "Langfuse ElastiCache Redis endpoint"
  value       = module.langfuse.redis_endpoint
  sensitive   = true
}

output "langfuse_s3_bucket" {
  description = "Langfuse S3 events bucket name"
  value       = module.langfuse.s3_events_bucket_name
}

output "langfuse_init_user_password" {
  description = "Langfuse initial user password (auto-generated)"
  value       = module.langfuse.langfuse_init_user_password
  sensitive   = true
}
