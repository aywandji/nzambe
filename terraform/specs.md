# Technical Specification: ECS Fargate for FastAPI RAG

## 1. Objective
Deploy a containerized FastAPI application to AWS ECS Fargate with environment parity between staging and production. The setup minimizes code duplication using Terraform modules and variables.

## 2. Infrastructure Architecture
*   **VPC:** Multi-AZ VPC with Public subnets (for the Application Load Balancer), Private subnets (for Fargate tasks), and NAT Gateways to allow tasks to reach external LLM APIs (OpenAI, Anthropic, etc.).
*   **Compute:** ECS Cluster utilizing the Fargate launch type for serverless container execution.
*   **Networking:**
    *   **ALB:** Application Load Balancer to distribute incoming traffic.
    *   **Security Groups:** Strictly defined to allow traffic only from the ALB to Fargate tasks on port 8000.
*   **Storage/Secrets:**
    *   **IAM Roles:** Dedicated roles for task execution and application logic.
    *   **S3:** For hosting RAG document stores or vector indices.
    *   **Secrets Manager:** For secure storage of API keys and database credentials.

## 3. Directory Structure
A modular approach ensures that core logic is written once and reused across environments.

```plain text
terraform/
├── modules/
│   ├── vpc/             # VPC, Subnets, IGW, NAT Gateway
│   ├── ecs/             # Cluster, Service, Task Definition, IAM Roles
│   └── alb/             # Load Balancer, Target Group, Listeners
├── envs/
│   ├── staging/
│   │   ├── main.tf      # Environment-specific module calls
│   │   ├── variables.tf # Staging-specific values
│   │   └── backend.tf   # S3 state configuration for staging
│   └── prod/
│       ├── main.tf      # Environment-specific module calls
│       ├── variables.tf # Production-specific values
│       └── backend.tf   # S3 state configuration for prod
└── global/              # Shared ECR Repository & S3 State Bucket
```


## 4. Environment-Specific Differentiators
The following table outlines the scaling and resource differences between environments:

| Feature | Staging | Production |
| :--- | :--- | :--- |
| **Fargate Capacity** | 1 Task (fixed) | 2–4 Tasks (Auto-scaling) |
| **Task Size** | 0.5 vCPU / 1GB RAM | 1 vCPU / 2GB RAM |
| **Multi-AZ** | 2 Availability Zones | 3 Availability Zones |
| **Logs Retention** | 7 Days | 30 Days |
| **NAT Gateways** | 1 (Cost-saving) | 1 per AZ (High Availability) |

## 5. Security Requirements
*   **IAM (Principle of Least Privilege):**
    *   **Task Execution Role:** Permissions limited to ECR image pulling and CloudWatch logging (`logs:CreateLogStream`, `logs:PutLogEvents`).
    *   **Task Role:** Fine-grained access to specific S3 buckets (e.g., `s3:GetObject` on RAG prefixes) and specific Secrets Manager ARNs.
*   **Secrets Management:**
    *   No hardcoded environment variables.
    *   Secrets injected directly into the container via the ECS Task Definition using AWS Secrets Manager references.
*   **Network Isolation:** Fargate tasks reside in private subnets with no direct public ingress.
