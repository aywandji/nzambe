output "s3_bucket_name" {
  description = "Name of the S3 bucket for Terraform state"
  value       = aws_s3_bucket.terraform_state.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for Terraform state"
  value       = aws_s3_bucket.terraform_state.arn
}

output "dynamodb_table_name" {
  description = "Name of the DynamoDB table for state locking"
  value       = aws_dynamodb_table.terraform_locks.name
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.app.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.app.arn
}

output "ecr_repository_name" {
  description = "Name of the ECR repository"
  value       = aws_ecr_repository.app.name
}

output "rag_bucket_name" {
  description = "Name of the S3 bucket for RAG documents"
  value       = var.create_rag_bucket ? aws_s3_bucket.rag_documents[0].id : ""
}

output "rag_bucket_arn" {
  description = "ARN of the S3 bucket for RAG documents"
  value       = var.create_rag_bucket ? aws_s3_bucket.rag_documents[0].arn : ""
}
