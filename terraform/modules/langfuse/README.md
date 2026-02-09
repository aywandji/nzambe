# Langfuse Terraform Module

This Terraform module deploys Langfuse v3 on AWS ECS with all required infrastructure components.

## Architecture

### ECS Services (3 services)
1. **Langfuse Web** - Main UI and API (Port 3000)
   - Docker image: `langfuse/langfuse:3`
   - Resources: 2 vCPU, 4GB RAM
   - Health check: `/api/public/health`

2. **Langfuse Worker** - Background job processor (Port 3030)
   - Docker image: `langfuse/langfuse-worker:3`
   - Resources: 2 vCPU, 4GB RAM
   - Processes queued events and background tasks

3. **ClickHouse** - Analytics database (Ports 8123, 9000)
   - Docker image: `clickhouse/clickhouse-server:24.11`
   - Resources: 2 vCPU, 8GB RAM
   - Persistent storage: EFS

### AWS Managed Services
- **Aurora PostgreSQL Serverless v2** - Transactional database
- **ElastiCache Redis** - Cache and queue
- **S3** - Event and media storage
- **EFS** - ClickHouse data persistence
- **Secrets Manager** - Secure credential storage
- **CloudWatch Logs** - Application logs

### Networking
- **Service Discovery (AWS Cloud Map)** - Internal DNS for ClickHouse
- **Security Groups** - Network isolation and access control
- **Application Load Balancer** - HTTP(S) traffic routing

## Prerequisites

1. **Existing Infrastructure** (passed as variables):
   - VPC with private subnets
   - ECS Cluster
   - Application Load Balancer with security group
   - ALB Target Group (created in parent module)

2. **Required Variables**:
   - `langfuse_init_user_email` - Email for initial admin user
   - `langfuse_init_user_password` - Password for initial admin user

## Usage

```hcl
module "langfuse" {
  source = "../../modules/langfuse"

  name_prefix               = "myproject-staging"
  environment               = "staging"
  aws_region                = "us-west-2"
  vpc_id                    = module.vpc.vpc_id
  private_subnet_ids        = module.vpc.private_subnet_ids
  ecs_cluster_id            = module.ecs.cluster_id
  alb_security_group_id     = module.alb.alb_security_group_id
  langfuse_target_group_arn = aws_lb_target_group.langfuse.arn
  langfuse_url              = "http://${module.alb.alb_dns_name}"

  # Required initialization
  langfuse_init_user_email    = "admin@example.com"
  langfuse_init_user_password = "SecurePassword123!"

  # Optional: Customize resources
  langfuse_web_desired_count    = 1
  langfuse_worker_desired_count = 1
  aurora_min_capacity           = 0.5
  aurora_max_capacity           = 2
}
```

## Deployment Steps

### 1. Configure Variables
Create a `terraform.tfvars` file:
```hcl
langfuse_init_user_email    = "your-email@example.com"
langfuse_init_user_password = "YourSecurePassword123!"
```

### 2. Initialize and Plan
```bash
cd terraform/envs/staging
terraform init
terraform plan
```

### 3. Deploy
```bash
terraform apply
```

**Expected deployment time:** 10-15 minutes
- Aurora: ~5 minutes
- ElastiCache: ~3 minutes
- ECS services: ~3-5 minutes

### 4. Verify Deployment

Check ECS services are running:
```bash
aws ecs describe-services \
  --cluster nzambe-staging-cluster \
  --services nzambe-staging-langfuse-web nzambe-staging-langfuse-worker nzambe-staging-clickhouse
```

Check health endpoints:
```bash
# Get ALB DNS name
ALB_DNS=$(terraform output -raw alb_dns_name)

# Check Langfuse health
curl http://$ALB_DNS/api/public/health

# Expected response: {"status":"OK"}
```

### 5. Access Langfuse UI

Navigate to the ALB URL:
```bash
echo "Langfuse URL: http://$(terraform output -raw alb_dns_name)"
```

Login with:
- **Email**: Value from `langfuse_init_user_email`
- **Password**: Value from `langfuse_init_user_password`

### 6. Get API Keys

After first login, Langfuse will display the API keys. **Important**: Copy these keys immediately!

Update AWS Secrets Manager with the actual keys:
```bash
# Update with actual keys from Langfuse UI
aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-public-key \
  --secret-string "pk-lf-..."

aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-secret-key \
  --secret-string "sk-lf-..."
```

### 7. Restart RAG Server

Restart the RAG server to pick up the new Langfuse keys:
```bash
aws ecs update-service \
  --cluster nzambe-staging-cluster \
  --service nzambe-staging-service \
  --force-new-deployment
```

### 8. Test Integration

Send a test query to your RAG server:
```bash
curl -X POST "http://$ALB_DNS/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of life?"}'
```

Check Langfuse dashboard for the trace (refresh the page).

## Configuration

### Resource Sizing

**Staging defaults (cost-optimized):**
- Aurora: 0.5-1 ACU (~$30-60/month)
- Redis: cache.t4g.small (~$15/month)
- Langfuse Web: 1 task @ 2 vCPU/4GB (~$25/month)
- Langfuse Worker: 1 task @ 2 vCPU/4GB (~$25/month)
- ClickHouse: 1 task @ 2 vCPU/8GB (~$50/month)

**Total: ~$145-175/month**

**Production recommendations:**
```hcl
aurora_min_capacity           = 1
aurora_max_capacity           = 4
redis_node_type               = "cache.m6g.large"
langfuse_web_desired_count    = 2
langfuse_worker_desired_count = 2
```

### Environment Variables

All required environment variables are automatically set by the module:

**Langfuse Web & Worker:**
- `DATABASE_URL` - PostgreSQL connection string
- `CLICKHOUSE_URL` - ClickHouse HTTP endpoint
- `CLICKHOUSE_MIGRATION_URL` - ClickHouse TCP endpoint
- `REDIS_CONNECTION_STRING` - Redis endpoint
- `NEXTAUTH_URL` - Public URL
- `NEXTAUTH_SECRET` - Session secret (auto-generated)
- `SALT` - API key salt (auto-generated)
- `ENCRYPTION_KEY` - Data encryption key (auto-generated)
- `LANGFUSE_S3_*` - S3 bucket configuration

**ClickHouse:**
- `CLICKHOUSE_USER` - Username
- `CLICKHOUSE_PASSWORD` - Password (auto-generated)

## Outputs

```hcl
output "langfuse_url"                      # URL to access Langfuse
output "langfuse_web_service_name"         # ECS service name for Web
output "langfuse_worker_service_name"      # ECS service name for Worker
output "clickhouse_service_name"           # ECS service name for ClickHouse
output "langfuse_public_key_secret_arn"    # Secrets Manager ARN
output "langfuse_secret_key_secret_arn"    # Secrets Manager ARN
output "database_endpoint"                 # Aurora endpoint
output "redis_endpoint"                    # ElastiCache endpoint
output "s3_events_bucket_name"             # S3 bucket for events
```

## Monitoring

### CloudWatch Logs

View logs for each service:
```bash
# Langfuse Web logs
aws logs tail /ecs/nzambe-staging-langfuse-web --follow

# Langfuse Worker logs
aws logs tail /ecs/nzambe-staging-langfuse-worker --follow

# ClickHouse logs
aws logs tail /ecs/nzambe-staging-clickhouse --follow
```

### ECS Metrics

View service status in AWS Console:
- ECS → Clusters → nzambe-staging-cluster
- Check CPU/Memory utilization
- View running tasks

## Troubleshooting

### Issue: Langfuse Web won't start

**Check logs:**
```bash
aws logs tail /ecs/nzambe-staging-langfuse-web --follow
```

**Common causes:**
- Database not ready (wait 5-10 minutes after Aurora creation)
- ClickHouse not healthy (check ClickHouse service)
- Environment variable misconfiguration

### Issue: ClickHouse fails to start

**Check logs:**
```bash
aws logs tail /ecs/nzambe-staging-clickhouse --follow
```

**Common causes:**
- EFS mount issues (verify EFS mount targets exist)
- Insufficient memory (increase to 8GB minimum)

### Issue: No traces appearing in Langfuse

**Verify RAG server has correct keys:**
```bash
# Check secrets are set
aws secretsmanager get-secret-value --secret-id nzambe-staging-langfuse-public-key
aws secretsmanager get-secret-value --secret-id nzambe-staging-langfuse-secret-key
aws secretsmanager get-secret-value --secret-id nzambe-staging-langfuse-base-url

# Check RAG server logs for Langfuse auth errors
aws logs tail /ecs/nzambe-staging --follow | grep -i langfuse
```

### Issue: Database connection errors

**Verify security groups:**
```bash
# Aurora should allow inbound 5432 from Langfuse tasks SG
# Check security group rules in AWS Console
```

## Security

### Secrets Management
- All secrets auto-generated with Terraform `random_password`
- Stored in AWS Secrets Manager with encryption at rest
- Injected into ECS tasks via `secrets` field (not environment variables)

### Network Security
- All services run in private subnets (no public IPs)
- Security groups restrict access:
  - Langfuse Web: Only accessible from ALB
  - Aurora: Only accessible from Langfuse tasks
  - Redis: Only accessible from Langfuse tasks
  - ClickHouse: Only accessible from Langfuse tasks

### IAM Roles
- Task Execution Role: Pull images, write logs, read secrets
- Task Role: Access S3 bucket for event storage

## Backup and Recovery

### Aurora PostgreSQL
- Automated backups enabled (7-day retention)
- Point-in-time recovery (PITR)
- Final snapshot created on destroy (production)

### ClickHouse Data
- Stored on EFS with automatic backups
- Consider AWS Backup for EFS snapshots

### S3 Events
- Versioning enabled on S3 bucket
- Objects retained indefinitely

## Cost Optimization

### Staging Environment
1. **Use Fargate Spot** - 70% cost savings
2. **Low Aurora ACU** - 0.5 minimum
3. **Small Redis node** - cache.t4g.small
4. **Single task** - No redundancy needed

### Production Environment
1. **Aurora Reserved Capacity** - 30-50% savings
2. **ElastiCache Reserved Nodes** - 30-50% savings
3. **Right-size ClickHouse** - Monitor and adjust
4. **S3 Lifecycle Policies** - Archive old events to Glacier

## Maintenance

### Updating Langfuse Version

Update Docker image tags in `ecs.tf`:
```hcl
image = "docker.io/langfuse/langfuse:3.1"  # New version
```

Deploy:
```bash
terraform apply
```

ECS will perform rolling update automatically.

### Scaling

Increase task count:
```hcl
langfuse_web_desired_count    = 2
langfuse_worker_desired_count = 2
```

Increase database capacity:
```hcl
aurora_min_capacity = 1
aurora_max_capacity = 4
```

## Destroying Resources

**Warning:** This will delete all Langfuse data!

```bash
terraform destroy
```

To preserve data, take snapshots first:
1. Create Aurora manual snapshot
2. Create EFS backup
3. Export S3 bucket data

## Support

- **Langfuse Documentation**: https://langfuse.com/docs
- **Langfuse GitHub**: https://github.com/langfuse/langfuse
- **AWS ECS Documentation**: https://docs.aws.amazon.com/ecs/

## License

This module is part of the Nzambe project.
