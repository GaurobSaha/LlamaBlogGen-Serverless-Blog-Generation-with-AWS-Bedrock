# ----------------------------
# Section 1: Imports
# ----------------------------
import boto3  # AWS SDK for Python – allows calling Bedrock(invoke foundation models) and S3
import botocore.config  # For configuring retries and timeouts
import json  # For serializing/deserializing data
from datetime import datetime  # To generate timestamps for filenames

# ----------------------------
# Section 2: Blog Generation Function using LLaMA 3
# ----------------------------
def blog_generate_using_llama3(blogtopic: str) -> str:
    # LLaMA 3 expects this specific instruction format
    prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Write a 200-word blog on the topic: {blogtopic}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    # Body contains model parameters for generation
    body = {
        "prompt": prompt,           # Prompt input to the model
        "max_gen_len": 128,         # Max number of tokens to generate
        "temperature": 0.5,         # Controls randomness (0 = deterministic)
        "top_p": 0.9                # Top_p does nucleus sampling which controls diversity
    }

    try:
        # Create a Bedrock runtime client in us-east-1
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=botocore.config.Config(
                read_timeout=300,  # Waits up to 300s for model response
                retries={'max_attempts': 3}  # Retry on failure
            )
        )

        # Invoke the LLaMA 3 model with the JSON payload
        response = bedrock.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",  # ID of the model
            #body: JSON-formatted prompt and settings.
            body=json.dumps(body)  # Convert payload to JSON string
        )

        # Read the byte response from model
        response_content = response["body"].read() #read(): Reads the raw byte stream from AWS.
        response_data = json.loads(response_content)  # Convert bytes → dict
        blog_text = response_data["generation"]  # Extract generated text

        return blog_text

    except Exception as e:
        print(f"Error generating the blog: {e}")
        return ""  # If there's an error, return empty string

# ----------------------------
# Section 3: Save to S3 Function
#Saves the blog to an S3 bucket.
# ----------------------------
def save_blog_details_s3(s3_key, s3_bucket, generate_blog):
    s3 = boto3.client('s3')  # Create S3 client
    try:
        # Save object to S3
        s3.put_object(
            Bucket=s3_bucket,   # Target bucket name
            Key=s3_key,         # s3_key: File path (e.g., llama3-blogs/20250701_173000.txt)
            Body=generate_blog  # File content
        )
        print("Blog saved to S3")
    except Exception as e:
        print(f"Error saving blog to S3: {e}")

# ----------------------------
# Section 4: Lambda Handler Entry Point
# ----------------------------
# Lambda handler takes an event which is the input from API Gateway
# # and context which contains runtime information
#This is the entry point for AWS Lambda.
#AWS calls this function when a request hits the Lambda trigger (like API Gateway).

def lambda_handler(event, context):
    # Parse incoming JSON string from API Gateway
    event = json.loads(event['body'])  # Convert to Python dict
    blogtopic = event['blog_topic']    # Extract blog topic

    # Generate blog content via LLaMA 3
    generate_blog = blog_generate_using_llama3(blogtopic)

    if generate_blog:
        # Timestamp for filename uniqueness
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"llama3-blogs/{timestamp}.txt"  # Folder path + filename
        # Creates a timestamped filename like llama3-blogs/20250701_181200.txt.
        s3_bucket = 'awsbedrock-gaurob'         # Target bucket name

        # Save generated blog to S3
        save_blog_details_s3(s3_key, s3_bucket, generate_blog)
    else:
        print("Blog generation failed")

    # Final response back to the caller (e.g., API Gateway)
    return {
        'statusCode': 200,
        'body': json.dumps('Blog generation (LLaMA 3) completed.')
    }
