variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones (2 for staging)"
  type        = number
  default     = 2
}

variable "nat_gateway_count" {
  description = "Number of NAT gateways (1 for staging cost-saving)"
  type        = number
  default     = 1
}

# ALB Configuration
variable "alb_deletion_protection" {
  description = "Enable ALB deletion protection"
  type        = bool
  default     = false
}

# ECS Configuration
variable "container_name" {
  description = "Name of the container"
  type        = string
  default     = "nzambe-app"
}

variable "container_port" {
  description = "Port exposed by the container"
  type        = number
  default     = 8000
}

variable "health_check_path" {
  description = "Path for health check endpoint"
  type        = string
  default     = "/health"
}

variable "rag_server_image_tag" {
  description = "Rag fastapi server - Docker image tag to deploy"
  type        = string
}

variable "lambda_image_tag" {
  description = "Lambda Docker image tag to deploy"
  type        = string
}

# Staging: 0.5 vCPU / 1GB RAM
variable "task_cpu" {
  description = "CPU units for the task (512 = 0.5 vCPU)"
  type        = string
  default     = "512"
}

variable "task_memory" {
  description = "Memory for the task in MB"
  type        = string
  default     = "1024"
}

variable "desired_count" {
  description = "Number of task instances (1 fixed for staging)"
  type        = number
  default     = 1
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days (7 for staging)"
  type        = number
  default     = 7
}

variable "additional_env_vars" {
  description = "Additional environment variables for the container (non sensitive)"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

variable "vector_index_chunk_size" {
  description = "Chunk size for each document indexing"
  type        = number
  default     = 512
}

variable "vector_index_chunk_overlap" {
  description = "Overlap size for each document indexing"
  type        = number
  default     = 120
}

variable "vector_index_dimension" {
  description = "Dimension of the vectors inside the index"
  type        = number
  default     = 1536 # embedding dimension of text-embedding-3-small. Both must match
}

variable "vector_index_embedding_model" {
  description = "Embedding model for each document indexing"
  type        = string
  default     = "text-embedding-3-small"
}

variable "vector_index_distance_metric" {
  description = "Distance metric for vector index"
  type        = string
  default     = "cosine"
}

variable "vector_index_data_type" {
  description = "Data type for the vector index"
  type        = string
  default     = "float32"
}

#####################################
# Langfuse Configuration
#####################################
variable "langfuse_database_name" {
  description = "Langfuse database name"
  type        = string
  default     = "langfuse"
}

variable "langfuse_database_username" {
  description = "Langfuse database username"
  type        = string
  default     = "langfuse"
}

variable "langfuse_aurora_min_capacity" {
  description = "Minimum ACU for Langfuse Aurora Serverless v2"
  type        = number
  default     = 0.5
}

variable "langfuse_aurora_max_capacity" {
  description = "Maximum ACU for Langfuse Aurora Serverless v2"
  type        = number
  default     = 1
}

variable "langfuse_skip_final_snapshot" {
  description = "Skip final snapshot when destroying Langfuse database (staging only)"
  type        = bool
  default     = true
}

variable "langfuse_redis_node_type" {
  description = "Redis node type for Langfuse"
  type        = string
  default     = "cache.t4g.small"
}

variable "langfuse_web_desired_count" {
  description = "Desired number of Langfuse Web tasks"
  type        = number
  default     = 1
}

variable "langfuse_worker_desired_count" {
  description = "Desired number of Langfuse Worker tasks"
  type        = number
  default     = 1
}

variable "langfuse_init_org_id" {
  description = "Initial Langfuse organization ID"
  type        = string
}

variable "langfuse_init_org_name" {
  description = "Initial Langfuse organization name"
  type        = string
}

variable "langfuse_init_project_id" {
  description = "Initial Langfuse project ID"
  type        = string
}

variable "langfuse_init_project_name" {
  description = "Initial Langfuse project name"
  type        = string
}

variable "langfuse_init_user_email" {
  description = "Initial Langfuse user email"
  type        = string
}
