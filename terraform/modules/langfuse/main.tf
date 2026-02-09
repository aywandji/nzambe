#####################################
# Langfuse Module - AWS Infrastructure
# Components: Aurora PostgreSQL, ElastiCache Redis, S3, Security Groups
#####################################

#####################################
# S3 Bucket for Event Storage
#####################################
resource "aws_s3_bucket" "langfuse_events" {
  bucket = "${var.name_prefix}-langfuse-events"

  tags = {
    Name        = "${var.name_prefix}-langfuse-events"
    Environment = var.environment
    Purpose     = "Langfuse event storage"
  }
}

resource "aws_s3_bucket_versioning" "langfuse_events" {
  bucket = aws_s3_bucket.langfuse_events.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "langfuse_events" {
  bucket = aws_s3_bucket.langfuse_events.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "langfuse_events" {
  bucket = aws_s3_bucket.langfuse_events.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#####################################
# Security Groups
#####################################

# Security Group for Langfuse ECS Tasks (Web + Worker)
resource "aws_security_group" "langfuse_tasks" {
  name        = "${var.name_prefix}-langfuse-tasks-sg"
  description = "Security group for Langfuse ECS tasks"
  vpc_id      = var.vpc_id

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.name_prefix}-langfuse-tasks-sg"
    Environment = var.environment
  }
}

# Security Group for ClickHouse ECS Task
resource "aws_security_group" "clickhouse" {
  name        = "${var.name_prefix}-clickhouse-sg"
  description = "Security group for ClickHouse"
  vpc_id      = var.vpc_id

  ingress {
    description     = "HTTP port from Langfuse tasks"
    from_port       = 8123
    to_port         = 8123
    protocol        = "tcp"
    security_groups = [aws_security_group.langfuse_tasks.id]
  }

  ingress {
    description     = "Native TCP port from Langfuse tasks"
    from_port       = 9000
    to_port         = 9000
    protocol        = "tcp"
    security_groups = [aws_security_group.langfuse_tasks.id]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.name_prefix}-clickhouse-sg"
    Environment = var.environment
  }
}

# Security Group for Aurora PostgreSQL
resource "aws_security_group" "aurora" {
  name        = "${var.name_prefix}-langfuse-aurora-sg"
  description = "Security group for Aurora PostgreSQL"
  vpc_id      = var.vpc_id

  ingress {
    description     = "PostgreSQL from Langfuse tasks"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.langfuse_tasks.id]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.name_prefix}-langfuse-aurora-sg"
    Environment = var.environment
  }
}

# Security Group for ElastiCache Redis
resource "aws_security_group" "redis" {
  name        = "${var.name_prefix}-langfuse-redis-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Redis from Langfuse tasks"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.langfuse_tasks.id]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.name_prefix}-langfuse-redis-sg"
    Environment = var.environment
  }
}

# Allow ALB to reach Langfuse Web
resource "aws_vpc_security_group_ingress_rule" "langfuse_from_alb" {
  from_port                    = 3000
  to_port                      = 3000
  ip_protocol                  = "tcp"
  security_group_id            = aws_security_group.langfuse_tasks.id
  referenced_security_group_id = var.alb_security_group_id
  description                  = "Allow ALB to reach Langfuse Web"
}

#####################################
# Aurora PostgreSQL Serverless v2
#####################################
resource "aws_db_subnet_group" "langfuse" {
  name       = "${var.name_prefix}-langfuse-db-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name        = "${var.name_prefix}-langfuse-db-subnet-group"
    Environment = var.environment
  }
}

resource "aws_rds_cluster" "langfuse" {
  cluster_identifier        = "${var.name_prefix}-langfuse-cluster"
  engine                    = "aurora-postgresql"
  engine_mode               = "provisioned"
  engine_version            = var.aurora_engine_version
  database_name             = var.database_name
  master_username           = var.database_username
  master_password           = random_password.db_password.result
  db_subnet_group_name      = aws_db_subnet_group.langfuse.name
  vpc_security_group_ids    = [aws_security_group.aurora.id]
  skip_final_snapshot       = var.skip_final_snapshot
  final_snapshot_identifier = var.skip_final_snapshot ? null : "${var.name_prefix}-langfuse-final-snapshot"
  backup_retention_period   = var.backup_retention_period
  preferred_backup_window   = "03:00-04:00"
  storage_encrypted         = true

  serverlessv2_scaling_configuration {
    min_capacity = var.aurora_min_capacity
    max_capacity = var.aurora_max_capacity
  }

  tags = {
    Name        = "${var.name_prefix}-langfuse-cluster"
    Environment = var.environment
  }
}

resource "aws_rds_cluster_instance" "langfuse" {
  identifier         = "${var.name_prefix}-langfuse-instance"
  cluster_identifier = aws_rds_cluster.langfuse.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.langfuse.engine
  engine_version     = aws_rds_cluster.langfuse.engine_version

  tags = {
    Name        = "${var.name_prefix}-langfuse-instance"
    Environment = var.environment
  }
}

#####################################
# Database Connection URL Secrets
#####################################
resource "aws_secretsmanager_secret" "database_url" {
  name        = "${var.name_prefix}-langfuse-database-url"
  description = "Langfuse PostgreSQL DATABASE_URL with embedded credentials"

  tags = {
    Name        = "${var.name_prefix}-langfuse-database-url"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id     = aws_secretsmanager_secret.database_url.id
  secret_string = "postgresql://${var.database_username}:${random_password.db_password.result}@${aws_rds_cluster.langfuse.endpoint}:5432/${var.database_name}"
}

resource "aws_secretsmanager_secret" "direct_url" {
  name        = "${var.name_prefix}-langfuse-direct-url"
  description = "Langfuse PostgreSQL DIRECT_URL with embedded credentials"

  tags = {
    Name        = "${var.name_prefix}-langfuse-direct-url"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "direct_url" {
  secret_id     = aws_secretsmanager_secret.direct_url.id
  secret_string = "postgresql://${var.database_username}:${random_password.db_password.result}@${aws_rds_cluster.langfuse.endpoint}:5432/${var.database_name}"
}

#####################################
# ElastiCache Redis
#####################################
resource "aws_elasticache_subnet_group" "langfuse" {
  name       = "${var.name_prefix}-langfuse-redis-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name        = "${var.name_prefix}-langfuse-redis-subnet-group"
    Environment = var.environment
  }
}

resource "aws_elasticache_cluster" "langfuse" {
  cluster_id           = "${var.name_prefix}-langfuse-redis"
  engine               = "redis"
  engine_version       = var.redis_engine_version
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.langfuse.name
  security_group_ids   = [aws_security_group.redis.id]

  tags = {
    Name        = "${var.name_prefix}-langfuse-redis"
    Environment = var.environment
  }
}

#####################################
# EFS for ClickHouse Persistence
#####################################
resource "aws_efs_file_system" "clickhouse" {
  creation_token = "${var.name_prefix}-clickhouse-data"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name        = "${var.name_prefix}-clickhouse-data"
    Environment = var.environment
  }
}

resource "aws_efs_mount_target" "clickhouse" {
  count           = length(var.private_subnet_ids)
  file_system_id  = aws_efs_file_system.clickhouse.id
  subnet_id       = var.private_subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "aws_security_group" "efs" {
  name        = "${var.name_prefix}-clickhouse-efs-sg"
  description = "Security group for ClickHouse EFS"
  vpc_id      = var.vpc_id

  ingress {
    description     = "NFS from ClickHouse"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.clickhouse.id]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.name_prefix}-clickhouse-efs-sg"
    Environment = var.environment
  }
}

#####################################
# CloudWatch Log Groups
#####################################
resource "aws_cloudwatch_log_group" "langfuse_web" {
  name              = "/ecs/${var.name_prefix}-langfuse-web"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.name_prefix}-langfuse-web-logs"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "langfuse_worker" {
  name              = "/ecs/${var.name_prefix}-langfuse-worker"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.name_prefix}-langfuse-worker-logs"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "clickhouse" {
  name              = "/ecs/${var.name_prefix}-clickhouse"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.name_prefix}-clickhouse-logs"
    Environment = var.environment
  }
}
