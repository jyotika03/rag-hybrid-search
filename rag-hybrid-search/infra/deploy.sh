#!/usr/bin/env bash
# Build, push to ECR, and deploy Project 6 to AWS via CloudFormation.
set -euo pipefail

PROJECT=rag-hybrid-search
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT}"
STACK="${PROJECT}-stack"

echo ">> Ensuring ECR repo (created by stack on first run; safe to ignore error)"
aws ecr describe-repositories --repository-names "$PROJECT" --region "$REGION" >/dev/null 2>&1 || \
  aws cloudformation deploy --stack-name "$STACK" --region "$REGION" \
    --template-file cloudformation.yaml --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides ContainerImage="public.ecr.aws/docker/library/python:3.11-slim" \
      OpenAIApiKey="${OPENAI_API_KEY:-placeholder}" || true

echo ">> Logging in to ECR"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo ">> Building and pushing image"
docker build -t "${ECR}:latest" ..
docker push "${ECR}:latest"

echo ">> Deploying stack"
aws cloudformation deploy \
  --stack-name "$STACK" --region "$REGION" \
  --template-file cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ContainerImage="${ECR}:latest" \
    OpenAIApiKey="${OPENAI_API_KEY:?set OPENAI_API_KEY}" \
    AnthropicApiKey="${ANTHROPIC_API_KEY:-}"

echo ">> Outputs:"
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs" --output table
