import os
from pathlib import Path
import boto3

# Initialize S3 client
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['bucketeer-9ed767dc-a15d-49e8-a0a5-0870a333bf2d'],
    aws_access_key_id=os.environ['AKIARVGPJVYVE7V2DYI6'],
    aws_secret_access_key=os.environ['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']
)

# Download model on startup
MODEL_PATH = "intent_model/pytorch_model.bin"
if not Path(MODEL_PATH).exists():
    Path("intent_model").mkdir(parents=True, exist_ok=True)
    s3.download_file(
        os.environ['BUCKETEER_BUCKET_NAME'],
        'models/pytorch_model.bin',  # Your model path in S3
        MODEL_PATH
    )
