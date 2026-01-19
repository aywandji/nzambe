variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "nzambe"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.1.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones (3 for production)"
  type        = number
  default     = 3
}

variable "nat_gateway_count" {
  description = "Number of NAT gateways (3 for production HA, 1 per AZ)"
  type        = number
  default     = 3
}

# ALB Configuration
variable "alb_deletion_protection" {
  description = "Enable ALB deletion protection"
  type        = bool
  default     = true
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

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

# Production: 1 vCPU / 2GB RAM
variable "task_cpu" {
  description = "CPU units for the task (1024 = 1 vCPU)"
  type        = string
  default     = "1024"
}

variable "task_memory" {
  description = "Memory for the task in MB"
  type        = string
  default     = "2048"
}

variable "desired_count" {
  description = "Initial number of task instances"
  type        = number
  default     = 2
}

variable "min_capacity" {
  description = "Minimum number of tasks for auto-scaling"
  type        = number
  default     = 2
}

variable "max_capacity" {
  description = "Maximum number of tasks for auto-scaling"
  type        = number
  default     = 4
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days (30 for production)"
  type        = number
  default     = 30
}

variable "additional_env_vars" {
  description = "Additional environment variables for the container"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}
