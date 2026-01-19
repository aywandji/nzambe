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

variable "create_rag_bucket" {
  description = "Create S3 bucket for RAG documents"
  type        = bool
  default     = true
}
