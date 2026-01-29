terraform {
  # Uncomment after running terraform in global/ first to create the S3 bucket
  # Then run: terraform init -migrate-state

  backend "s3" {
    bucket       = "nzambe-terraform-state"
    key          = "prod/terraform.tfstate"
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
