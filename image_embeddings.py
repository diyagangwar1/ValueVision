import pandas as pd
import tqdm
import json

import torch
import torch.nn as nn
import torchvision.models as models

import asyncio
import aiohttp
import requests
from PIL import Image
from io import BytesIO

import numpy as np
import re

# Cell 1
image_df = pd.read_csv('4 - Picture.csv')

image_df = image_df[image_df['picAddr'].str.match("http://i\d")]

price_df = pd.read_csv('1 - Propery_Basic.csv')[['ID', 'price']]


# Pretrained ResNet18 model
model = models.resnet18(pretrained=True)
# Remove the last layer
new_model = nn.Sequential(*list(model.children())[:-1]).double()


# Cell 2
# async code
async def fetch_image(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            img_bytes = await response.read()
            img = Image.open(BytesIO(img_bytes)).resize((224, 224))
            return np.array(img) / 255. 
        else:
            print(f"Failed to fetch {url}: {response.status}")
            return None

async def fetch_all_images(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, url) for url in urls]
        images = await asyncio.gather(*tasks)

        images = [img for img in images if img.shape == (224, 224, 3)]
        return images

# old sequential code
# def url_to_array(url):
#   response = requests.get(url)
#   if response.status_code == 200:
#     img = Image.open(BytesIO(response.content)).resize((224, 224))

#     return np.array(img)
#   else:
#     return None
    

# Cell 3
# compile image tensors / embeddings sequentially
num_listings = 1000  # len(price_df)

image_embeddings = []
ids = price_df['ID']
prices = price_df['price']


for idx in tqdm.tqdm(range(num_listings), desc="compiling image tensors"):
    listing_id = ids[idx]
    price = prices[idx]
    listing_images_urls = image_df.loc[image_df['proID'] == listing_id, 'picAddr'].to_list()
    num_images = len(listing_images_urls)
    if num_images > 0:
        arrs = asyncio.run(fetch_all_images(listing_images_urls))
        # correct num images
        num_images = len(arrs)

        if num_images > 0:
            i = 0
            while len(arrs) < 5:
                arrs.append(arrs[i])
                i = (i + 1) % num_images
    else:
        print(f"No images found for listing {listing_id}")
    try:
        np_batch = np.stack(arrs)
        batch = torch.from_numpy(np_batch).transpose(2, 3)
        batch = batch.transpose(1, 2)
        

        # find image embedding:
        embedding = torch.mean(new_model(batch), dim=0).flatten()

        image_embeddings.append([listing_id, embedding.tolist(), price])
    except:
        for arr in arrs:
            print(arr.shape)
