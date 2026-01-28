variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "environment" {
  description = "Environment name (staging or prod)"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where ECS will be deployed"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for ECS tasks"
  type        = list(string)
}

variable "alb_security_group_id" {
  description = "Security group ID of the ALB"
  type        = string
}

variable "target_group_arn" {
  description = "ARN of the ALB target group"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "container_name" {
  description = "Name of the container"
  type        = string
  default     = "app"
}

variable "container_image" {
  description = "Docker image to run (ECR repository URI)"
  type        = string
}

variable "container_port" {
  description = "Port exposed by the container"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "CPU units for the task (256 = 0.25 vCPU, 512 = 0.5 vCPU, 1024 = 1 vCPU)"
  type        = string
  default     = "512"
}

variable "task_memory" {
  description = "Memory for the task in MB"
  type        = string
  default     = "1024"
}

variable "desired_count" {
  description = "Number of task instances to run"
  type        = number
  default     = 1
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "secrets_arns" {
  description = "List of Secrets Manager ARNs that the task can access"
  type        = list(string)
  default     = []
}

variable "secrets_env_vars" {
  description = "List of secrets to inject as environment variables"
  type = list(object({
    name      = string
    valueFrom = string
  }))
  default = []
}

variable "environment_variables" {
  description = "List of environment variables (non-secret)"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

variable "s3_vector_store_bucket_arn" {
  description = "ARN of S3 bucket for RAG document vector store"
  type        = string
  default     = ""
}

# Auto-scaling variables
variable "min_capacity" {
  description = "Minimum number of tasks"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of tasks"
  type        = number
  default     = 4
}

# Capacity Provider variables
variable "fargate_spot_weight" {
  description = "Weight for FARGATE_SPOT when enabled (0-100). Higher values use more Spot capacity"
  type        = number
  default     = 70
  validation {
    condition     = var.fargate_spot_weight >= 0 && var.fargate_spot_weight <= 100
    error_message = "fargate_spot_weight must be between 0 and 100"
  }
}
