# Terraform Infrastructure for Nzambe FastAPI RAG Application

This Terraform configuration deploys a containerized FastAPI application to AWS ECS Fargate with complete environment parity between staging and production.

## Architecture Overview

- **VPC**: Multi-AZ VPC with public subnets (ALB) and private subnets (Fargate tasks)
- **Compute**: ECS Fargate for serverless container execution
- **Load Balancing**: Application Load Balancer distributing traffic to Fargate tasks
- **Security**: IAM roles with the least privilege, Secrets Manager for sensitive data
- **Storage**:
  - ECR for container images with lifecycle policies for image retention
  - Two S3 buckets for Terraform state and RAG documents. The buckets have versioning, encryption and public access blocked

## Directory Structure

```
terraform/
├── modules/
│   ├── vpc/              # VPC, Subnets, NAT Gateway, Route Tables
│   ├── alb/              # Application Load Balancer, Target Groups
│   └── ecs/              # ECS Cluster, Service, Task Definition, Auto-scaling
├── envs/
│   ├── staging/          # Staging environment configuration
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── backend.tf
│   └── prod/             # Production environment configuration
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       └── backend.tf
└── global/               # Shared resources (ECR, S3 state bucket)
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    └── terraform.tf
```

## Environment Differences

| Feature              | Staging                | Production              |
|----------------------|------------------------|-------------------------|
| **Fargate Capacity** | 1 Task (fixed)         | 2–4 Tasks (Auto-scaled) |
| **Task Size**        | 0.5 vCPU / 1GB RAM     | 1 vCPU / 2GB RAM        |
| **Availability**     | 2 AZs                  | 3 AZs                   |
| **Logs Retention**   | 7 Days                 | 30 Days                 |
| **NAT Gateways**     | 1 (cost-saving)        | 3 (1 per AZ, HA)        |
| **Auto-scaling**     | Disabled               | Enabled                 |
| **VPC CIDR**         | 10.0.0.0/16            | 10.1.0.0/16             |

## Prerequisites

1. **AWS CLI** configured with credentials
   ```bash
   aws configure
   ```

2. **Terraform** >= 1.2 installed
   ```bash
   terraform version
   ```

3. **Docker** image built and ready to push
   ```bash
   docker build --build-arg APP_VERSION=$(git describe --tags --always) -t nzambe:latest .
   ```

## Deployment Steps

### Step 1: Deploy Global Resources (First Time Only)

The global resources include the ECR repository, S3 bucket for Terraform state, and DynamoDB table for state locking.

```bash
cd terraform/global

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the configuration
terraform apply

# Note the outputs (ECR repository URL, S3 bucket name, etc.)
terraform output
```

After this step, you'll have:
- ECR repository: `<account-id>.dkr.ecr.us-west-2.amazonaws.com/nzambe-ecr-repo`
- S3 bucket: `nzambe-terraform-state`
- DynamoDB table: `nzambe-terraform-locks`

### Step 2: Push Docker Image to ECR

```bash
# Get ECR login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Tag your image
docker tag nzambe:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/nzambe-ecr-repo:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/nzambe-ecr-repo:latest
```

### Step 3: Configure Secrets in AWS Secrets Manager

Before deploying the application, you need to set the secret values:

```bash
# For staging
aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-secret-key \
  --secret-string "sk-lf-your-secret-key"

aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-public-key \
  --secret-string "pk-lf-your-public-key"

aws secretsmanager put-secret-value \
  --secret-id nzambe-staging-langfuse-base-url \
  --secret-string "https://your-langfuse-url.com"

# Repeat for production (replace 'staging' with 'prod')
```

### Step 4: Deploy Staging Environment

```bash
cd terraform/envs/staging

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the configuration
terraform apply

# Get the ALB URL
terraform output alb_url
```

Access your application at the ALB URL output: `http://<alb-dns-name>`

### Step 5: (Optional) Enable Remote State Backend

After the S3 bucket is created in Step 1, you can migrate to remote state storage:

```bash
# Uncomment the backend configuration in backend.tf
# Then run:
terraform init -migrate-state
```

### Step 6: Deploy Production Environment

```bash
cd terraform/envs/prod

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the configuration
terraform apply

# Get the ALB URL
terraform output alb_url
```

## Updating the Application

To deploy a new version of your application:

1. **Build and push new Docker image:**
   ```bash
   docker build --build-arg APP_VERSION=v1.2.3 -t nzambe:v1.2.3 .
   docker tag nzambe:v1.2.3 <ecr-url>/nzambe-ecr-repo:v1.2.3
   docker push <ecr-url>/nzambe-ecr-repo:v1.2.3
   ```

2. **Update the image tag in Terraform:**
   ```bash
   cd terraform/envs/staging
   terraform apply -var="image_tag=v1.2.3"
   ```

3. **Force new deployment (if using the same tag):**
   ```bash
   aws ecs update-service \
     --cluster nzambe-staging-cluster \
     --service nzambe-staging-service \
     --force-new-deployment \
     --region us-west-2
   ```

## Monitoring and Logs

### CloudWatch Logs
View application logs:
```bash
# Staging
aws logs tail /ecs/nzambe-staging --follow --region us-west-2

# Production
aws logs tail /ecs/nzambe-prod --follow --region us-west-2
```

### ECS Service Status
```bash
# Check service status
aws ecs describe-services \
  --cluster nzambe-staging-cluster \
  --services nzambe-staging-service \
  --region us-west-2
```

### Health Check
```bash
curl http://<alb-dns-name>/health
```

## Security Best Practices

1. **Secrets Management**
   - All sensitive data is stored in AWS Secrets Manager
   - Never commit secrets to version control
   - Rotate secrets regularly

2. **Network Isolation**
   - Fargate tasks run in private subnets
   - Only ALB has public access
   - Security groups restrict traffic to necessary ports only

3. **IAM Least Privilege**
   - Task execution role: ECR pull + CloudWatch logs + Secrets Manager
   - Task role: S3 read-only access to specific prefixes

4. **Encryption**
   - Secrets Manager: encrypted at rest
   - S3 buckets: AES-256 encryption enabled
   - ECR: images scanned on push

## Cost Optimization

**Staging Environment:**
- 1 NAT Gateway instead of 3 (saves ~$64/month)
- Smaller task size (0.5 vCPU / 1GB instead of 1 vCPU / 2GB)
- Fixed 1 task (no auto-scaling)
- 7-day log retention instead of 30 days

**Estimated Monthly Costs:**
- Staging: ~$50-70
- Production: ~$150-200 (depending on traffic and scaling)

## Troubleshooting

### Task fails to start
1. Check CloudWatch logs for container errors
2. Verify ECR image exists and is accessible
3. Ensure secrets are properly configured
4. Check task execution role permissions

### Cannot access ALB
1. Verify security group rules
2. Check target group health checks
3. Ensure tasks are running in private subnets
4. Verify route tables and NAT Gateway

### Out of memory errors
1. Increase `task_memory` in variables.tf
2. Monitor memory usage in CloudWatch
3. Consider vertical scaling (larger tasks) or horizontal scaling (more tasks)

## Cleanup

To destroy all resources:

```bash
# Destroy environments first
cd terraform/envs/staging
terraform destroy

cd ../prod
terraform destroy

# Then destroy global resources
cd ../../global
terraform destroy
```

**Note:** Delete all objects in S3 buckets before destroying, as buckets with objects cannot be deleted.

## Additional Resources

- [AWS ECS Fargate Documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [FastAPI Deployment Best Practices](https://fastapi.tiangolo.com/deployment/)
