#####################################
# Langfuse ECS Services
# - ClickHouse (analytics database with EFS persistence)
# - Langfuse Web (UI/API)
# - Langfuse Worker (background jobs)
#####################################

#####################################
# IAM Roles
#####################################

# Task Execution Role (for ECS to pull images, write logs, read secrets)
resource "aws_iam_role" "langfuse_task_execution" {
  name = "${var.name_prefix}-langfuse-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.name_prefix}-langfuse-task-execution-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "langfuse_task_execution_policy" {
  role       = aws_iam_role.langfuse_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Additional policy for Secrets Manager access
resource "aws_iam_role_policy" "langfuse_task_execution_secrets" {
  name = "${var.name_prefix}-langfuse-task-execution-secrets"
  role = aws_iam_role.langfuse_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.nextauth_secret.arn,
          aws_secretsmanager_secret.salt.arn,
          aws_secretsmanager_secret.encryption_key.arn,
          aws_secretsmanager_secret.langfuse_init_project_secret_key.arn,
          aws_secretsmanager_secret.langfuse_init_project_public_key.arn,
          aws_secretsmanager_secret.clickhouse_password.arn,
          aws_secretsmanager_secret.db_password.arn,
          aws_secretsmanager_secret.langfuse_init_user_password.arn,
          aws_secretsmanager_secret.database_url.arn,
          aws_secretsmanager_secret.direct_url.arn,
          aws_secretsmanager_secret.clickhouse_url.arn,
          aws_secretsmanager_secret.clickhouse_migration_url.arn
        ]
      }
    ]
  })
}

# Task Role (for application to access AWS services like S3)
resource "aws_iam_role" "langfuse_task" {
  name = "${var.name_prefix}-langfuse-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.name_prefix}-langfuse-task-role"
    Environment = var.environment
  }
}

# S3 access policy for Langfuse tasks
resource "aws_iam_role_policy" "langfuse_s3_access" {
  name = "${var.name_prefix}-langfuse-s3-access"
  role = aws_iam_role.langfuse_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.langfuse_events.arn,
          "${aws_s3_bucket.langfuse_events.arn}/*"
        ]
      }
    ]
  })
}

#####################################
# ECS Task Definitions
#####################################

# ClickHouse Task Definition
resource "aws_ecs_task_definition" "clickhouse" {
  family                   = "${var.name_prefix}-clickhouse"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.clickhouse_cpu
  memory                   = var.clickhouse_memory
  execution_role_arn       = aws_iam_role.langfuse_task_execution.arn
  task_role_arn            = aws_iam_role.langfuse_task.arn

  volume {
    name = "clickhouse-data"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.clickhouse.id
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([
    {
      name      = "clickhouse"
      image     = "docker.io/clickhouse/clickhouse-server:${var.clickhouse_version}"
      essential = true

      portMappings = [
        {
          containerPort = 8123
          protocol      = "tcp"
        },
        {
          containerPort = 9000
          protocol      = "tcp"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "clickhouse-data"
          containerPath = "/var/lib/clickhouse"
          readOnly      = false
        }
      ]

      secrets = [
        {
          name      = "CLICKHOUSE_PASSWORD"
          valueFrom = aws_secretsmanager_secret.clickhouse_password.arn
        }
      ]

      environment = [
        {
          name  = "CLICKHOUSE_DB"
          value = "default"
        },
        {
          name  = "CLICKHOUSE_USER"
          value = var.clickhouse_user
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.clickhouse.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "wget --spider -q http://localhost:8123/ping || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Name        = "${var.name_prefix}-clickhouse"
    Environment = var.environment
  }
}

# Langfuse Web Task Definition
resource "aws_ecs_task_definition" "langfuse_web" {
  family                   = "${var.name_prefix}-langfuse-web"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.langfuse_web_cpu
  memory                   = var.langfuse_web_memory
  execution_role_arn       = aws_iam_role.langfuse_task_execution.arn
  task_role_arn            = aws_iam_role.langfuse_task.arn

  container_definitions = jsonencode([
    {
      name      = "langfuse-web"
      image     = "docker.io/langfuse/langfuse:3"
      essential = true

      portMappings = [
        {
          containerPort = 3000
          protocol      = "tcp"
        }
      ]

      secrets = [
        {
          name      = "NEXTAUTH_SECRET"
          valueFrom = aws_secretsmanager_secret.nextauth_secret.arn
        },
        {
          name      = "SALT"
          valueFrom = aws_secretsmanager_secret.salt.arn
        },
        {
          name      = "ENCRYPTION_KEY"
          valueFrom = aws_secretsmanager_secret.encryption_key.arn
        },
        {
          name      = "LANGFUSE_INIT_PROJECT_SECRET_KEY"
          valueFrom = aws_secretsmanager_secret.langfuse_init_project_secret_key.arn
        },
        {
          name      = "LANGFUSE_INIT_PROJECT_PUBLIC_KEY"
          valueFrom = aws_secretsmanager_secret.langfuse_init_project_public_key.arn
        },
        {
          name      = "DATABASE_URL"
          valueFrom = aws_secretsmanager_secret.database_url.arn
        },
        {
          name      = "DIRECT_URL"
          valueFrom = aws_secretsmanager_secret.direct_url.arn
        },
        {
          name      = "CLICKHOUSE_URL"
          valueFrom = aws_secretsmanager_secret.clickhouse_url.arn
        },
        {
          name      = "CLICKHOUSE_MIGRATION_URL"
          valueFrom = aws_secretsmanager_secret.clickhouse_migration_url.arn
        },
        {
          name      = "LANGFUSE_INIT_USER_PASSWORD"
          valueFrom = aws_secretsmanager_secret.langfuse_init_user_password.arn
        }
      ]

      environment = concat([
        {
          name  = "REDIS_CONNECTION_STRING"
          value = "redis://${aws_elasticache_cluster.langfuse.cache_nodes[0].address}:6379"
        },
        {
          name  = "NEXTAUTH_URL"
          value = var.langfuse_url
        },
        {
          name  = "LANGFUSE_S3_EVENT_UPLOAD_BUCKET"
          value = aws_s3_bucket.langfuse_events.id
        },
        {
          name  = "LANGFUSE_S3_EVENT_UPLOAD_REGION"
          value = var.aws_region
        },
        {
          name  = "LANGFUSE_S3_EVENT_UPLOAD_PREFIX"
          value = "events/"
        },
        {
          name  = "LANGFUSE_S3_MEDIA_UPLOAD_BUCKET"
          value = aws_s3_bucket.langfuse_events.id
        },
        {
          name  = "LANGFUSE_S3_MEDIA_UPLOAD_REGION"
          value = var.aws_region
        },
        {
          name  = "LANGFUSE_S3_MEDIA_UPLOAD_PREFIX"
          value = "media/"
        },
        {
          name  = "LANGFUSE_INIT_ORG_ID"
          value = var.langfuse_init_org_id
        },
        {
          name  = "LANGFUSE_INIT_ORG_NAME"
          value = var.langfuse_init_org_name
        },
        {
          name  = "LANGFUSE_INIT_PROJECT_ID"
          value = var.langfuse_init_project_id
        },
        {
          name  = "LANGFUSE_INIT_PROJECT_NAME"
          value = var.langfuse_init_project_name
        },
        {
          name  = "LANGFUSE_INIT_USER_EMAIL"
          value = var.langfuse_init_user_email
        },
        {
          name  = "LANGFUSE_INIT_USER_NAME"
          value = var.langfuse_init_user_name
        }
      ], var.additional_env_vars)

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.langfuse_web.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3000/api/public/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 90
      }
    }
  ])

  tags = {
    Name        = "${var.name_prefix}-langfuse-web"
    Environment = var.environment
  }
}

# Langfuse Worker Task Definition
resource "aws_ecs_task_definition" "langfuse_worker" {
  family                   = "${var.name_prefix}-langfuse-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.langfuse_worker_cpu
  memory                   = var.langfuse_worker_memory
  execution_role_arn       = aws_iam_role.langfuse_task_execution.arn
  task_role_arn            = aws_iam_role.langfuse_task.arn

  container_definitions = jsonencode([
    {
      name      = "langfuse-worker"
      image     = "docker.io/langfuse/langfuse-worker:3"
      essential = true

      portMappings = [
        {
          containerPort = 3030
          protocol      = "tcp"
        }
      ]

      secrets = [
        {
          name      = "NEXTAUTH_SECRET"
          valueFrom = aws_secretsmanager_secret.nextauth_secret.arn
        },
        {
          name      = "SALT"
          valueFrom = aws_secretsmanager_secret.salt.arn
        },
        {
          name      = "ENCRYPTION_KEY"
          valueFrom = aws_secretsmanager_secret.encryption_key.arn
        },
        {
          name      = "DATABASE_URL"
          valueFrom = aws_secretsmanager_secret.database_url.arn
        },
        {
          name      = "DIRECT_URL"
          valueFrom = aws_secretsmanager_secret.direct_url.arn
        },
        {
          name      = "CLICKHOUSE_URL"
          valueFrom = aws_secretsmanager_secret.clickhouse_url.arn
        },
        {
          name      = "CLICKHOUSE_MIGRATION_URL"
          valueFrom = aws_secretsmanager_secret.clickhouse_migration_url.arn
        }
      ]

      environment = concat([
        {
          name  = "REDIS_CONNECTION_STRING"
          value = "redis://${aws_elasticache_cluster.langfuse.cache_nodes[0].address}:6379"
        },
        {
          name  = "LANGFUSE_S3_EVENT_UPLOAD_BUCKET"
          value = aws_s3_bucket.langfuse_events.id
        },
        {
          name  = "LANGFUSE_S3_EVENT_UPLOAD_REGION"
          value = var.aws_region
        },
        {
          name  = "LANGFUSE_S3_EVENT_UPLOAD_PREFIX"
          value = "events/"
        },
        {
          name  = "LANGFUSE_S3_MEDIA_UPLOAD_BUCKET"
          value = aws_s3_bucket.langfuse_events.id
        },
        {
          name  = "LANGFUSE_S3_MEDIA_UPLOAD_REGION"
          value = var.aws_region
        },
        {
          name  = "LANGFUSE_S3_MEDIA_UPLOAD_PREFIX"
          value = "media/"
        }
      ], var.additional_env_vars)

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.langfuse_worker.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3030/api/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 90
      }
    }
  ])

  tags = {
    Name        = "${var.name_prefix}-langfuse-worker"
    Environment = var.environment
  }
}

#####################################
# Service Discovery (Cloud Map)
#####################################
resource "aws_service_discovery_private_dns_namespace" "langfuse" {
  name        = "${var.name_prefix}-langfuse.local"
  description = "Private DNS namespace for Langfuse services"
  vpc         = var.vpc_id

  tags = {
    Name        = "${var.name_prefix}-langfuse-namespace"
    Environment = var.environment
  }
}

resource "aws_service_discovery_service" "clickhouse" {
  name = "clickhouse"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.langfuse.id

    dns_records {
      ttl  = 10
      type = "A"
    }
  }

  health_check_config {
    failure_threshold = 1
  }

  tags = {
    Name        = "${var.name_prefix}-clickhouse-discovery"
    Environment = var.environment
  }
}

#####################################
# ClickHouse Connection URL Secrets
#####################################
resource "aws_secretsmanager_secret" "clickhouse_url" {
  name        = "${var.name_prefix}-langfuse-clickhouse-url"
  description = "Langfuse ClickHouse HTTP URL with embedded credentials"

  tags = {
    Name        = "${var.name_prefix}-langfuse-clickhouse-url"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "clickhouse_url" {
  secret_id     = aws_secretsmanager_secret.clickhouse_url.id
  secret_string = "http://${var.clickhouse_user}:${random_password.clickhouse_password.result}@${aws_service_discovery_service.clickhouse.name}.${aws_service_discovery_private_dns_namespace.langfuse.name}:8123"
}

resource "aws_secretsmanager_secret" "clickhouse_migration_url" {
  name        = "${var.name_prefix}-langfuse-clickhouse-migration-url"
  description = "Langfuse ClickHouse migration URL with embedded credentials"

  tags = {
    Name        = "${var.name_prefix}-langfuse-clickhouse-migration-url"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "clickhouse_migration_url" {
  secret_id     = aws_secretsmanager_secret.clickhouse_migration_url.id
  secret_string = "clickhouse://${var.clickhouse_user}:${random_password.clickhouse_password.result}@${aws_service_discovery_service.clickhouse.name}.${aws_service_discovery_private_dns_namespace.langfuse.name}:9000"
}

#####################################
# ECS Services
#####################################

# ClickHouse Service
resource "aws_ecs_service" "clickhouse" {
  name            = "${var.name_prefix}-clickhouse"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.clickhouse.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.clickhouse.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.clickhouse.arn
  }

  tags = {
    Name        = "${var.name_prefix}-clickhouse-service"
    Environment = var.environment
  }

  depends_on = [aws_efs_mount_target.clickhouse]
}

# Langfuse Web Service
resource "aws_ecs_service" "langfuse_web" {
  name            = "${var.name_prefix}-langfuse-web"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.langfuse_web.arn
  desired_count   = var.langfuse_web_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.langfuse_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.langfuse_target_group_arn
    container_name   = "langfuse-web"
    container_port   = 3000
  }

  tags = {
    Name        = "${var.name_prefix}-langfuse-web-service"
    Environment = var.environment
  }

  depends_on = [
    aws_ecs_service.clickhouse,
    aws_rds_cluster_instance.langfuse
  ]
}

# Langfuse Worker Service
resource "aws_ecs_service" "langfuse_worker" {
  name            = "${var.name_prefix}-langfuse-worker"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.langfuse_worker.arn
  desired_count   = var.langfuse_worker_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.langfuse_tasks.id]
    assign_public_ip = false
  }

  tags = {
    Name        = "${var.name_prefix}-langfuse-worker-service"
    Environment = var.environment
  }

  depends_on = [
    aws_ecs_service.clickhouse,
    aws_rds_cluster_instance.langfuse
  ]
}
