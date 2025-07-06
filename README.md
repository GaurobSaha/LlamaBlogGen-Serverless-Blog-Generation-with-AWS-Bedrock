"# LlamaBlogGen-Serverless-Blog-Generation-with-AWS-Bedrock" 

A fully serverless blog generation pipeline using AWS Bedrock, Lambda, API Gateway, and S3, powered by Meta’s LLaMA 3 (8B Instruct) model.
This project accepts a blog topic through a REST API, invokes the LLaMA 3 model via Bedrock to generate a 200-word blog, and automatically stores the output in an S3 bucket.
