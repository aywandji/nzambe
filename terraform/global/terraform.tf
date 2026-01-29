terraform {
  # Note: The global resources are bootstrapped first
  # State will be stored locally initially, then can be migrated to S3 after creation

  backend "s3" {
    bucket       = "nzambe-terraform-state"
    key          = "global/terraform.tfstate"
    region       = "us-west-2"
    use_lockfile = true
    encrypt      = true
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.28"
    }
  }

  required_version = ">= 1.2"
}
