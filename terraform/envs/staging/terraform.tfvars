# Staging Environment Configuration

#####################################
# Project Configuration
#####################################
project_name = "nzambe"
environment  = "staging"
aws_region   = "us-west-2"

#####################################
# RAG Server Configuration
#####################################
rag_server_image_tag = "latest"
lambda_image_tag     = "lambda-latest"

#####################################
# Langfuse Configuration
#####################################
# Organization and Project Configuration
langfuse_init_org_id       = "wandji"
langfuse_init_org_name     = "Wandji Corporation"
langfuse_init_project_id   = "nzambe"
langfuse_init_project_name = "Nzambe RAG"

# User Configuration
langfuse_init_user_email = "yankwarmand@gmail.com" # Your email for initial Langfuse admin user
# Note: Password is auto-generated and stored in AWS Secrets Manager

# Optional: Customize Langfuse resources (defaults are staging-appropriate)
# langfuse_database_name         = "langfuse"
# langfuse_database_username     = "langfuse"
# langfuse_aurora_min_capacity   = 0.5  # Minimum ACU
# langfuse_aurora_max_capacity   = 1    # Maximum ACU
# langfuse_skip_final_snapshot   = true
# langfuse_redis_node_type       = "cache.t4g.small"
# langfuse_web_desired_count     = 1
# langfuse_worker_desired_count  = 1
