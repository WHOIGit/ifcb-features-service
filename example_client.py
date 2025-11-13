import json
import requests
from PIL import Image

def get_blob(roi_image):
    """Send an IFCB ROI image to the blob extraction service and return the blob mask image."""
    import base64
    import io

    # Convert PIL image to base64-encoded PNG
    buffered = io.BytesIO()
    roi_image.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Prepare the request payload
    payload = {
        "image_data": b64_image
    }

    # Send the request to the blob extraction service
    response = requests.post("http://localhost:8010/features/extract", json=payload)

    if response.status_code != 200:
        raise Exception(f"Blob extraction failed: {response.text}")

    # Decode the returned blob mask image
    result = response.json()
    blob_mask_image = Image.open(io.BytesIO(base64.b64decode(result['blob'])))
    features = result['features']

    return blob_mask_image, features

if __name__ == "__main__":
    # Example usage
    roi_image = Image.open("data/roi.png")
    blob_mask, features = get_blob(roi_image)
    blob_mask.save("data/blob.png")
    with open("data/features.json", "w") as f:
        json.dump(features, f, indent=2)