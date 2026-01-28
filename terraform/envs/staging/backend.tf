terraform {
  # Uncomment after running terraform in global/ first to create the S3 bucket
  # Then run: terraform init -migrate-state

  backend "s3" {
    bucket         = "nzambe-terraform-state"
    key            = "staging/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "nzambe-terraform-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.28"
    }
  }

  required_version = ">= 1.2"
}
