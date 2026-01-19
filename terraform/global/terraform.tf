terraform {
  # Note: The global resources are bootstrapped first
  # State will be stored locally initially, then can be migrated to S3 after creation

  backend "s3" {
    bucket         = "nzambe-terraform-state"
    key            = "global/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "nzambe-terraform-locks"
    encrypt        = true
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.92"
    }
  }

  required_version = ">= 1.2"
}
