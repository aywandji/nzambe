#####################################
# Langfuse Module Outputs
#####################################

#####################################
# Langfuse Access
#####################################
output "langfuse_url" {
  description = "URL to access Langfuse UI"
  value       = var.langfuse_url
}

output "langfuse_web_service_name" {
  description = "Name of the Langfuse Web ECS service"
  value       = aws_ecs_service.langfuse_web.name
}

output "langfuse_worker_service_name" {
  description = "Name of the Langfuse Worker ECS service"
  value       = aws_ecs_service.langfuse_worker.name
}

#####################################
# API Keys (for RAG Server Integration)
#####################################
output "langfuse_public_key_secret_arn" {
  description = "ARN of the secret containing Langfuse public key"
  value       = aws_secretsmanager_secret.langfuse_init_project_public_key.arn
}

output "langfuse_secret_key_secret_arn" {
  description = "ARN of the secret containing Langfuse secret key"
  value       = aws_secretsmanager_secret.langfuse_init_project_secret_key.arn
}

#####################################
# Database
#####################################
output "database_endpoint" {
  description = "Aurora PostgreSQL cluster endpoint"
  value       = aws_rds_cluster.langfuse.endpoint
}

output "database_name" {
  description = "Database name"
  value       = var.database_name
}

#####################################
# Redis
#####################################
output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_cluster.langfuse.cache_nodes[0].address
}

#####################################
# ClickHouse
#####################################
output "clickhouse_service_name" {
  description = "ClickHouse ECS service name"
  value       = aws_ecs_service.clickhouse.name
}

output "clickhouse_dns_name" {
  description = "ClickHouse DNS name (via Cloud Map)"
  value       = "${aws_service_discovery_service.clickhouse.name}.${aws_service_discovery_private_dns_namespace.langfuse.name}"
}

#####################################
# S3
#####################################
output "s3_events_bucket_name" {
  description = "S3 bucket name for Langfuse events"
  value       = aws_s3_bucket.langfuse_events.id
}

output "s3_events_bucket_arn" {
  description = "S3 bucket ARN for Langfuse events"
  value       = aws_s3_bucket.langfuse_events.arn
}

#####################################
# Security Groups
#####################################
output "langfuse_tasks_security_group_id" {
  description = "Security group ID for Langfuse tasks"
  value       = aws_security_group.langfuse_tasks.id
}

output "clickhouse_security_group_id" {
  description = "Security group ID for ClickHouse"
  value       = aws_security_group.clickhouse.id
}

#####################################
# Secrets (for reference)
#####################################
output "nextauth_secret_arn" {
  description = "ARN of NEXTAUTH_SECRET"
  value       = aws_secretsmanager_secret.nextauth_secret.arn
}

output "salt_secret_arn" {
  description = "ARN of SALT secret"
  value       = aws_secretsmanager_secret.salt.arn
}

output "encryption_key_secret_arn" {
  description = "ARN of ENCRYPTION_KEY secret"
  value       = aws_secretsmanager_secret.encryption_key.arn
}

output "langfuse_init_user_password_secret_arn" {
  description = "ARN of the secret containing Langfuse initial user password"
  value       = aws_secretsmanager_secret.langfuse_init_user_password.arn
}

output "langfuse_init_user_password" {
  description = "Langfuse initial user password for UI access (auto-generated)"
  value       = random_password.langfuse_init_user_password.result
  sensitive   = true
}
