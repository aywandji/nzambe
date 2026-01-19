#####################################
# VPC Module - Multi-AZ Networking
# This module contains:
# - VPC
# - Internet Gateway for the public subnet
# - Public Subnet (for ALB and NAT Gateway(s)) and its corresponding Route Table
# - Private Subnet (for Fargate tasks) and its corresponding Route Table
# - NAT Gateway for the private subnet
#####################################

# Data source for available AZs
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  # Select the list of AZs based on var.az_count
  azs = slice(data.aws_availability_zones.available.names, 0, var.az_count)
}

#####################################
# VPC
#####################################
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.name_prefix}-vpc"
    Environment = var.environment
  }
}

#####################################
# Internet Gateway
#####################################
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${var.name_prefix}-igw"
    Environment = var.environment
  }
}

#####################################
# Public Subnets (for ALB)
#####################################
resource "aws_subnet" "public" {
  count                   = var.az_count
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.name_prefix}-public-${local.azs[count.index]}"
    Environment = var.environment
    Type        = "public"
  }
}

#####################################
# Private Subnets (for Fargate tasks)
#####################################
resource "aws_subnet" "private" {
  count             = var.az_count
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + var.az_count)
  availability_zone = local.azs[count.index]

  tags = {
    Name        = "${var.name_prefix}-private-${local.azs[count.index]}"
    Environment = var.environment
    Type        = "private"
  }
}

#####################################
# Elastic IPs for NAT Gateway(s)
#####################################
resource "aws_eip" "nat" {
  count  = var.nat_gateway_count
  domain = "vpc"

  tags = {
    Name        = "${var.name_prefix}-nat-eip-${count.index + 1}"
    Environment = var.environment
  }

  depends_on = [aws_internet_gateway.main]
}

#####################################
# NAT Gateway(s)
#####################################
resource "aws_nat_gateway" "main" {
  count         = var.nat_gateway_count
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name        = "${var.name_prefix}-nat-${local.azs[count.index]}"
    Environment = var.environment
  }

  depends_on = [aws_internet_gateway.main]
}

#####################################
# Route Table - Public
#####################################
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${var.name_prefix}-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "public" {
  count          = var.az_count
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

#####################################
# Route Table(s) - Private
#####################################
resource "aws_route_table" "private" {
  count  = var.az_count
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[min(count.index, var.nat_gateway_count - 1)].id
  }

  tags = {
    Name        = "${var.name_prefix}-private-rt-${local.azs[count.index]}"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "private" {
  count          = var.az_count
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}
