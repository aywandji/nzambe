# Langfuse v3 on ECS - Deployment Guide

## Overview

This guide walks you through deploying Langfuse v3 on your existing ECS cluster to enable observability for your RAG server.

**Architecture:** Langfuse Web + Langfuse Worker + ClickHouse on ECS, with Aurora PostgreSQL, ElastiCache Redis, and S3.

**Estimated Cost:** ~$145-175/month for staging environment

## Quick Start

### Prerequisites
- AWS CLI configured with appropriate credentials
- Terraform >= 1.2 installed
- Existing nzambe infrastructure deployed (VPC, ECS, ALB)

### 1. Configure Langfuse Variables

Create `terraform/envs/staging/terraform.tfvars`:

```hcl
# Copy from terraform.tfvars.example
langfuse_init_user_email    = "your-email@example.com"
langfuse_init_user_password = "YourSecurePassword123!"
```

Or use environment variables:
```bash
export TF_VAR_langfuse_init_user_email="your-email@example.com"
export TF_VAR_langfuse_init_user_password="YourSecurePassword123!"
```

### 2. Initialize Terraform

```bash
cd terraform/envs/staging
terraform init
```

### 3. Review the Plan

```bash
terraform plan
```

**Expected resources to be created:**
- 30+ new resources including:
  - 3 ECS services (Langfuse Web, Worker, ClickHouse)
  - 1 Aurora PostgreSQL Serverless v2 cluster
  - 1 ElastiCache Redis cluster
  - 1 S3 bucket
  - 1 EFS file system (for ClickHouse)
  - Multiple security groups
  - CloudWatch log groups
  - AWS Secrets Manager secrets

### 4. Deploy

```bash
terraform apply
```

Type `yes` when prompted.

**Deployment time:** ~10-15 minutes

**What's happening:**
1. Creating Aurora PostgreSQL cluster (~5 min)
2. Creating ElastiCache Redis (~3 min)
3. Creating EFS and S3 resources (~1 min)
4. Starting ClickHouse ECS service (~2 min)
5. Starting Langfuse Web and Worker (~3 min)
6. Running database migrations (~2 min)

### 5. Get Langfuse URL

```bash
terraform output langfuse_url
```

Example output: `http://nzambe-staging-alb-1234567890.us-west-2.elb.amazonaws.com`

### 6. Access Langfuse UI

Open the URL in your browser and login with:
- **Email**: Value from `langfuse_init_user_email`
- **Password**: Value from `langfuse_init_user_password`

**Important:** On first login, Langfuse will display the API keys. **Copy them immediately!**

### 7. Update Secrets with Actual API Keys

After getting the keys from Langfuse UI, update AWS Secrets Manager:

```bash
# Replace with actual keys from Langfuse UI
aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-public-key \
  --secret-string "pk-lf-1234567890abcdef"

aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-secret-key \
  --secret-string "sk-lf-1234567890abcdef"
```

### 8. Restart RAG Server

Force a new deployment to pick up the Langfuse keys:

```bash
aws ecs update-service \
  --cluster nzambe-staging-cluster \
  --service nzambe-staging-service \
  --force-new-deployment
```

Wait for the service to stabilize (~2-3 minutes):

```bash
aws ecs wait services-stable \
  --cluster nzambe-staging-cluster \
  --services nzambe-staging-service
```

### 9. Test Integration

Send a test query to your RAG server:

```bash
ALB_DNS=$(terraform output -raw alb_dns_name)

curl -X POST "http://$ALB_DNS/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of life?"}'
```

### 10. Verify Traces in Langfuse

1. Open Langfuse UI in browser
2. Navigate to "Traces" section
3. You should see a new trace for your query
4. Click on the trace to see detailed spans (embedding, retrieval, generation)

**Success!** Your RAG server is now tracked by Langfuse.

## Architecture Details

### Components Deployed

#### ECS Services (3 containers on existing cluster)

1. **Langfuse Web** - Main UI/API
   - Image: `langfuse/langfuse:3`
   - Port: 3000
   - Resources: 2 vCPU, 4GB RAM
   - Connected to: Aurora, ClickHouse, Redis, S3

2. **Langfuse Worker** - Background jobs
   - Image: `langfuse/langfuse-worker:3`
   - Port: 3030
   - Resources: 2 vCPU, 4GB RAM
   - Processes: Event ingestion, batch operations

3. **ClickHouse** - Analytics database
   - Image: `clickhouse/clickhouse-server:24.11`
   - Ports: 8123 (HTTP), 9000 (Native)
   - Resources: 2 vCPU, 8GB RAM
   - Storage: EFS (persistent)

#### AWS Managed Services

- **Aurora PostgreSQL Serverless v2**
  - Engine: PostgreSQL 16.4
  - Capacity: 0.5-1 ACU (staging)
  - Purpose: Transactional data (projects, users, settings)

- **ElastiCache Redis**
  - Node: cache.t4g.small
  - Purpose: Cache and queue

- **S3 Bucket**
  - Purpose: Event storage, media uploads
  - Features: Versioning, encryption

- **EFS**
  - Purpose: ClickHouse data persistence
  - Features: Encrypted, automatic backups

### Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ VPC (10.0.0.0/16)                                          │
│                                                             │
│  ┌────────────────┐         ┌─────────────────────────┐   │
│  │  Public Subnet │         │   Private Subnets        │   │
│  │                │         │                          │   │
│  │  ┌──────────┐  │         │  ┌────────────────┐     │   │
│  │  │   ALB    │──┼────────>│  │ Langfuse Web   │     │   │
│  │  └──────────┘  │         │  │ (ECS Fargate)  │     │   │
│  │                │         │  └────────┬───────┘     │   │
│  └────────────────┘         │           │              │   │
│                              │           v              │   │
│                              │  ┌────────────────┐     │   │
│                              │  │ Langfuse Worker│     │   │
│                              │  │ (ECS Fargate)  │     │   │
│                              │  └────────┬───────┘     │   │
│                              │           │              │   │
│                              │           v              │   │
│                              │  ┌────────────────┐     │   │
│                              │  │  ClickHouse    │     │   │
│                              │  │ (ECS + EFS)    │     │   │
│                              │  └────────────────┘     │   │
│                              │           │              │   │
│                              │           v              │   │
│                              │  ┌────────────────┐     │   │
│                              │  │ Aurora PostgreSQL│   │   │
│                              │  │  ElastiCache    │   │   │
│                              │  └────────────────┘     │   │
│                              └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Security

**Network Isolation:**
- All services in private subnets (no public IPs)
- Only ALB exposed to internet
- Security groups enforce least privilege

**Secrets Management:**
- All credentials in AWS Secrets Manager
- Auto-generated with strong random passwords
- Encrypted at rest

**IAM Roles:**
- Task Execution Role: Pull images, logs, secrets
- Task Role: S3 access only

## Monitoring & Troubleshooting

### View Logs

```bash
# Langfuse Web
aws logs tail /ecs/nzambe-staging-langfuse-web --follow

# Langfuse Worker
aws logs tail /ecs/nzambe-staging-langfuse-worker --follow

# ClickHouse
aws logs tail /ecs/nzambe-staging-clickhouse --follow

# RAG Server (for Langfuse integration logs)
aws logs tail /ecs/nzambe-staging --follow | grep -i langfuse
```

### Check Service Status

```bash
# Get service status
aws ecs describe-services \
  --cluster nzambe-staging-cluster \
  --services nzambe-staging-langfuse-web \
  --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount,Health:healthCheckGracePeriodSeconds}'

# Get task health
aws ecs list-tasks \
  --cluster nzambe-staging-cluster \
  --service-name nzambe-staging-langfuse-web

# Describe task for detailed status
aws ecs describe-tasks \
  --cluster nzambe-staging-cluster \
  --tasks <task-arn>
```

### Common Issues

#### Issue: Langfuse Web won't start

**Symptom:** Task keeps restarting

**Check:**
```bash
aws logs tail /ecs/nzambe-staging-langfuse-web --follow
```

**Common Causes:**
1. Aurora not ready (wait 5-10 minutes)
2. ClickHouse not healthy
3. Database migration failed

**Solution:**
```bash
# Wait for Aurora to be available
aws rds wait db-cluster-available \
  --db-cluster-identifier nzambe-staging-langfuse-cluster

# Restart service
aws ecs update-service \
  --cluster nzambe-staging-cluster \
  --service nzambe-staging-langfuse-web \
  --force-new-deployment
```

#### Issue: ClickHouse fails to start

**Symptom:** Container exits immediately

**Check:**
```bash
aws logs tail /ecs/nzambe-staging-clickhouse --follow
```

**Common Causes:**
1. EFS mount failed
2. Insufficient memory
3. Port conflicts

**Solution:**
```bash
# Verify EFS mount targets
aws efs describe-mount-targets \
  --file-system-id $(terraform output -json | jq -r '.langfuse_efs_id.value')

# Increase memory if needed (edit variables.tf)
clickhouse_memory = "8192"  # Minimum 8GB
```

#### Issue: No traces in Langfuse

**Symptom:** RAG queries work but don't appear in Langfuse

**Check:**
```bash
# Verify secrets are set correctly
aws secretsmanager get-secret-value \
  --secret-id nzambe-staging-langfuse-base-url

aws secretsmanager get-secret-value \
  --secret-id nzambe-staging-langfuse-public-key

# Check RAG server logs for Langfuse errors
aws logs tail /ecs/nzambe-staging --follow | grep -i langfuse
```

**Common Causes:**
1. API keys not updated in Secrets Manager
2. RAG server not restarted after key update
3. Langfuse authentication failure

**Solution:**
```bash
# Update secrets with actual keys from Langfuse UI
aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-public-key \
  --secret-string "pk-lf-..."

aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-secret-key \
  --secret-string "sk-lf-..."

# Force restart RAG server
aws ecs update-service \
  --cluster nzambe-staging-cluster \
  --service nzambe-staging-service \
  --force-new-deployment
```

## Cost Breakdown

### Staging Environment (~$145-175/month)

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| Aurora PostgreSQL | 0.5-1 ACU | $30-60 |
| ElastiCache Redis | cache.t4g.small | $15 |
| Langfuse Web | 1 task @ 2 vCPU/4GB | $25 |
| Langfuse Worker | 1 task @ 2 vCPU/4GB | $25 |
| ClickHouse | 1 task @ 2 vCPU/8GB | $50 |
| EFS | ~50GB | $5 |
| S3 | ~10GB | $0.23 |
| CloudWatch Logs | ~10GB | $5 |
| **Total** | | **$155-180** |

### Cost Optimization Tips

1. **Use Fargate Spot** (already configured for staging)
   - 70% cost savings on ECS tasks
   - Acceptable for non-critical staging environment

2. **Scale down when not in use**
   ```bash
   # Scale down to zero (evenings/weekends)
   aws ecs update-service --cluster nzambe-staging-cluster \
     --service nzambe-staging-langfuse-web --desired-count 0
   aws ecs update-service --cluster nzambe-staging-cluster \
     --service nzambe-staging-langfuse-worker --desired-count 0
   ```

3. **Aurora Auto-pause** (not currently supported for Serverless v2)
   - Consider using Aurora Serverless v1 if auto-pause is critical

4. **S3 Lifecycle Policies**
   ```hcl
   # Archive old events to Glacier after 90 days
   lifecycle_rule {
     enabled = true
     transition {
       days          = 90
       storage_class = "GLACIER"
     }
   }
   ```

## Maintenance

### Updating Langfuse Version

1. Check for new releases: https://github.com/langfuse/langfuse/releases

2. Update Docker tags in `terraform/modules/langfuse/ecs.tf`:
   ```hcl
   image = "docker.io/langfuse/langfuse:3.1"        # Update version
   image = "docker.io/langfuse/langfuse-worker:3.1"
   ```

3. Apply changes:
   ```bash
   terraform apply
   ```

ECS will perform a rolling update automatically.

### Scaling Up

For production or heavy usage:

```hcl
# terraform/envs/prod/terraform.tfvars
langfuse_web_desired_count    = 2
langfuse_worker_desired_count = 2
aurora_min_capacity           = 1
aurora_max_capacity           = 4
redis_node_type               = "cache.m6g.large"
```

### Backup and Recovery

**Aurora:**
- Automated backups: 7 days retention
- Manual snapshots: Create before major changes
  ```bash
  aws rds create-db-cluster-snapshot \
    --db-cluster-identifier nzambe-staging-langfuse-cluster \
    --db-cluster-snapshot-identifier manual-snapshot-$(date +%Y%m%d)
  ```

**EFS (ClickHouse data):**
- Automated with AWS Backup (configure separately)
- Manual backup:
  ```bash
  aws backup start-backup-job \
    --backup-vault-name default \
    --resource-arn <efs-arn>
  ```

**S3 (Events):**
- Versioning enabled automatically
- Consider replication for critical data

## Destroying Langfuse

**Warning:** This will delete all Langfuse data permanently!

### Option 1: Destroy Langfuse Only

```bash
cd terraform/envs/staging

# Remove Langfuse module from main.tf
# Comment out or delete the module "langfuse" block

terraform apply
```

### Option 2: Destroy Everything

```bash
terraform destroy
```

### Before Destroying (Data Preservation)

1. **Export data from Langfuse UI** (if available)

2. **Create Aurora snapshot:**
   ```bash
   aws rds create-db-cluster-snapshot \
     --db-cluster-identifier nzambe-staging-langfuse-cluster \
     --db-cluster-snapshot-identifier final-snapshot-$(date +%Y%m%d)
   ```

3. **Backup S3 data:**
   ```bash
   aws s3 sync s3://nzambe-staging-langfuse-events ./langfuse-backup/
   ```

4. **Backup EFS (ClickHouse):**
   ```bash
   # Create EFS backup via AWS Backup console
   ```

## Next Steps

### Production Deployment

1. Copy staging configuration to prod:
   ```bash
   cp -r terraform/envs/staging terraform/envs/prod
   ```

2. Update prod-specific values in `terraform/envs/prod/terraform.tfvars`

3. Deploy with higher resources:
   ```hcl
   langfuse_aurora_min_capacity   = 1
   langfuse_aurora_max_capacity   = 4
   langfuse_web_desired_count     = 2
   langfuse_worker_desired_count  = 2
   langfuse_skip_final_snapshot   = false  # Important!
   ```

### Advanced Configuration

- **Custom Domain**: Add ACM certificate and Route53 record
- **HTTPS**: Update ALB listener with SSL certificate
- **Auto-scaling**: Add Application Auto Scaling policies
- **Monitoring**: Set up CloudWatch alarms and dashboards
- **Backup**: Configure AWS Backup for automated snapshots

## Support

- **Langfuse Docs**: https://langfuse.com/docs
- **Langfuse GitHub**: https://github.com/langfuse/langfuse
- **Module README**: `terraform/modules/langfuse/README.md`

## Summary

✅ Langfuse v3 deployed on existing ECS cluster
✅ Complete observability for RAG server
✅ Cost-effective staging configuration (~$150/month)
✅ Production-ready architecture
✅ Fully automated with Terraform
✅ Integration with existing infrastructure

**Total deployment time:** ~30 minutes (including testing)
