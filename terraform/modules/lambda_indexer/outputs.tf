output "lambda_function_arn" {
  description = "ARN of the Lambda indexer function"
  value       = aws_lambda_function.indexer.arn
}

output "lambda_function_name" {
  description = "Name of the Lambda indexer function"
  value       = aws_lambda_function.indexer.function_name
}

output "lambda_log_group_name" {
  description = "Name of the CloudWatch log group for Lambda"
  value       = aws_cloudwatch_log_group.lambda.name
}
