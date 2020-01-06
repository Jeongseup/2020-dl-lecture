import base64
import io
import json
import os
import re
import sys

import modelann
import numpy as np
from PIL import Image

# Default output
res = {"result": 0, "data": [], "error": ''}
try:
    if os.environ["REQUEST_METHOD"] == "POST":
        data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))
        # Convert data url to numpy array
        img_str = re.search(r'base64,(.*)', data).group(1)
        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        im = im.resize((28,28))
        # Normalize and invert pixel values
        arr = np.array(im)[:,:,0:1]
        arr = arr.astype(np.float64)
        H, W, L = arr.shape
        arr = arr.reshape(-1, W * H)
        arr = (255-arr) / 255.
        Nin = 784
        Nh = 256
        number_of_class = 10
        Nout = number_of_class
        model = modelann.ANN_seq_class(Nin, Nh, Nout)
        model.load_weights("models/modelann.tfl")
        # Predict class
        predictions = model.predict(arr)[0]
        # Return label data
        res['result'] = 1
        res['data'] = [float(num) for num in predictions]

except Exception as e:
    # Return error data
    res['error'] = str(e)
# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))