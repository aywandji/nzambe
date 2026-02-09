#####################################
# Langfuse Module Variables
#####################################

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "environment" {
  description = "Environment name (staging, prod, etc.)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where Langfuse will be deployed"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs"
  type        = list(string)
}

variable "ecs_cluster_id" {
  description = "ECS cluster ID"
  type        = string
}

variable "alb_security_group_id" {
  description = "ALB security group ID"
  type        = string
}

variable "langfuse_target_group_arn" {
  description = "Target group ARN for Langfuse Web service"
  type        = string
}

variable "langfuse_url" {
  description = "Full URL where Langfuse will be accessible (e.g., https://your-alb.com)"
  type        = string
}

#####################################
# Aurora PostgreSQL Configuration
#####################################
variable "database_name" {
  description = "Database name"
  type        = string
  default     = "langfuse"
}

variable "database_username" {
  description = "Database master username"
  type        = string
  default     = "langfuse"
}

variable "aurora_engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
  default     = "16.4"
}

variable "aurora_min_capacity" {
  description = "Minimum ACU for Aurora Serverless v2"
  type        = number
  default     = 0.5
}

variable "aurora_max_capacity" {
  description = "Maximum ACU for Aurora Serverless v2"
  type        = number
  default     = 2
}

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 1
}

variable "skip_final_snapshot" {
  description = "Skip final snapshot when destroying"
  type        = bool
  default     = false
}

#####################################
# ElastiCache Redis Configuration
#####################################
variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.1"
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t4g.small"
}

#####################################
# ClickHouse Configuration
#####################################
variable "clickhouse_version" {
  description = "ClickHouse version"
  type        = string
  default     = "24.11"
}

variable "clickhouse_user" {
  description = "ClickHouse username"
  type        = string
  default     = "clickhouse"
}

variable "clickhouse_cpu" {
  description = "CPU units for ClickHouse task"
  type        = string
  default     = "2048"
}

variable "clickhouse_memory" {
  description = "Memory for ClickHouse task in MB"
  type        = string
  default     = "8192"
}

#####################################
# Langfuse Web Configuration
#####################################
variable "langfuse_web_cpu" {
  description = "CPU units for Langfuse Web task"
  type        = string
  default     = "2048"
}

variable "langfuse_web_memory" {
  description = "Memory for Langfuse Web task in MB"
  type        = string
  default     = "4096"
}

variable "langfuse_web_desired_count" {
  description = "Desired number of Langfuse Web tasks"
  type        = number
  default     = 1
}

#####################################
# Langfuse Worker Configuration
#####################################
variable "langfuse_worker_cpu" {
  description = "CPU units for Langfuse Worker task"
  type        = string
  default     = "2048"
}

variable "langfuse_worker_memory" {
  description = "Memory for Langfuse Worker task in MB"
  type        = string
  default     = "4096"
}

variable "langfuse_worker_desired_count" {
  description = "Desired number of Langfuse Worker tasks"
  type        = number
  default     = 1
}

#####################################
# CloudWatch Logs
#####################################
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 2
}

#####################################
# Langfuse Initialization
#####################################
variable "langfuse_init_org_id" {
  description = "Initial organization ID"
  type        = string
  default     = "default-org"
}

variable "langfuse_init_org_name" {
  description = "Initial organization name"
  type        = string
  default     = "Default Organization"
}

variable "langfuse_init_project_id" {
  description = "Initial project ID"
  type        = string
  default     = "default-project"
}

variable "langfuse_init_project_name" {
  description = "Initial project name"
  type        = string
  default     = "Default Project"
}

variable "langfuse_init_user_email" {
  description = "Initial user email"
  type        = string
}

variable "langfuse_init_user_name" {
  description = "Initial user name"
  type        = string
  default     = "Admin"
}

#####################################
# Additional Configuration
#####################################
variable "additional_env_vars" {
  description = "Additional environment variables for Langfuse containers"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}
