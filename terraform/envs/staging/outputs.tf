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
    langfuse_secret_key = aws_secretsmanager_secret.langfuse_secret_key.arn
    langfuse_public_key = aws_secretsmanager_secret.langfuse_public_key.arn
    langfuse_base_url   = aws_secretsmanager_secret.langfuse_base_url.arn
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
