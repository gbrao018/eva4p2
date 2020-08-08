import json
try:
    import unzip_requirements
except ImportError:
    pass
from requests_toolbelt.multipart import decoder
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from torchvision.models import mobilenet
from torch import nn

import boto3
import os
# import tarfile
import io
import base64
import json

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4p2-models'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobilenetv2.pt'
print('Downloading model...')

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        # get object from s3
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        # read it in memory
        bytestream = io.BytesIO(obj['Body'].read())
        print('Creating Bytestream')
        print('Loading Model')
        model = torch.jit.load(bytestream)
        print('Model Loaded')
    else:
        print('Model Path is a File')
except Exception as e:
    print(repr(e))
    raise (e)

def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise (e)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()

def classifyImage(event, context):
    try:
        print('converting headers to lowercase-start:')
        #for key in event['headers']:
        #    event['headers'][key.lower()] = event['headers'][key]
        print('headers-loowercase-conversion successful')
        content_type_header = event['headers']['content-type']
        #print(content_type_header)
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADEED')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print(prediction)
        filename = picture.headers[b'content-Disposition'].decode().split(';')[1].split('=')[1]
        print(filename)
        if len(filename) < 4:
            filename = picture.headers[b'content-Disposition'].decode().split(';')[2].split('=')[1]
        
        return {
            "statusCode": 200,
            "headers": {
                'content-type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted': prediction})
        }
    
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'content-type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }

