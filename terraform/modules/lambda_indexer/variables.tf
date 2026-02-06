variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "environment" {
  description = "Environment name (e.g., staging, prod)"
  type        = string
}

variable "lambda_image_uri" {
  description = "ECR image URI for Lambda function"
  type        = string
}

variable "source_bucket_name" {
  description = "Name of the S3 bucket containing source documents"
  type        = string
}

variable "source_bucket_arn" {
  description = "ARN of the S3 bucket containing source documents"
  type        = string
}

variable "vector_store_bucket_name" {
  description = "Name of the S3 bucket for vector index storage"
  type        = string
}

variable "vector_store_bucket_arn" {
  description = "ARN of the S3 bucket for vector index storage"
  type        = string
}

variable "s3vectors_index_arn" {
  description = "ARN of the S3 Vectors index"
  type        = string
}

variable "s3vectors_index_name" {
  description = "Name of the S3 Vectors index"
  type        = string
}

variable "openai_secret_arn" {
  description = "ARN of the Secrets Manager secret containing OpenAI API key"
  type        = string
}

variable "timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 300
}

variable "memory_size" {
  description = "Lambda function memory size in MB"
  type        = number
  default     = 2048
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "chunk_size" {
  description = "Text chunk size for document splitting"
  type        = number
  default     = 512
}

variable "chunk_overlap" {
  description = "Text chunk overlap for document splitting"
  type        = number
  default     = 120
}

variable "embedding_model" {
  description = "OpenAI embedding model name"
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
