# Hosting YOLOv8 With FastAPI

## Introduction
In the ever-evolving landscape of computer vision and machine learning, two powerful technologies have emerged as key players in their respective domains: YOLO (You Only Look Once) and FastAPI. YOLO has gained fame for its real-time object detection capabilities, while FastAPI has earned a reputation as one of the most efficient and user-friendly web frameworks for building APIs. In this blog post, we'll explore the exciting synergy that can be achieved by hosting YOLOv8, a state-of-the-art YOLO variant, with FastAPI.

First, let's briefly introduce FastAPI. FastAPI is a Python web framework that simplifies the development of APIs with incredible speed and ease. It is designed for high-performance and productivity, offering automatic generation of interactive documentation and type hints, which are a boon for developers. With FastAPI, you can build robust APIs quickly, making it an ideal choice for integrating machine learning models, like YOLOv8, into web applications.

On the other side of the equation is YOLO, a groundbreaking object detection model that has become a cornerstone of computer vision applications. YOLO excels at identifying objects in images and video streams in real-time. YOLOv8 is the latest iteration, bringing even more accuracy and speed to the table. Combining the power of YOLOv8 with the efficiency of FastAPI opens up exciting possibilities for building interactive and efficient object detection applications.

In this blog post, we will dive into the process of hosting YOLOv8 with FastAPI, demonstrating how to create a web-based API that can analyze images. By the end of this guide, you'll have a solid understanding of how to leverage these two technologies to build your own object detection applications, whether it's for security, surveillance, or any other use case that demands accurate and speedy object detection. Let's embark on this journey of integrating the cutting-edge YOLOv8 with the versatile FastAPI framework for a truly powerful and responsive object detection experience.

## Directory Structure
First, I do always like to split my code across multiple files. In
my opinion, it just makes it easier to read for me. I would be doing
a disservice if I didn't accurately show you the structure layout
so you can understand the imports that are happening between files:

```shell
|____yolofastapi
| |____routers
| | |______init__.py
| | |____yolo.py
| |______init__.py
| |____schemas
| | |____yolo.py
| |____detectors
| | |______init__.py
| | |____yolov8.py
| |____main.py
```

At the top level, we have the `yolofastapi` directory which will be our
python application. Within there, there are a few directories:

1. `routers` - The endpoints / REST routers that our application will expose.
               If, for example, you wanted to add a new `GET` endpoint, you
               could add that in this directory.
2. `schemas` - This directory will show our request/response schemas that our
               routers will either expect or return. Pydantic makes the 
               serialization of these objects a breeze!
3. `detectors` - This is the fun stuff! We will put our `yolo` or other detection
                 models/wrappers in this directory. In our example, we will only
                 using `yolov8n`, but you could extend this to other detectors
                 or yolo versions as well.

So, if you see something like:

```python
from yolofastapi.routers import yolo
```

You know understand that it is importing the file at `yolofastapi/routers/yolo.py`!

## Schemas
We will only have a single response schema in our API. Essentially, it will
be a data-type class which returns a few things to the user:

1. The id of the uploaded image
2. The labels our detector found

For example, a response might look like

```json
{
  "id": 1,
  "labels": [
    "vase"
  ]
}
```

Which says "Hey, I detected vases in your image and gave it an 
image id of 1 (in case you want to download it later)!". 

In python, it's really easy to use `pydantic` to do a lot of this
heavy lifting for us:

```python
from pydantic import BaseModel
from typing import Set

class ImageAnalysisResponse(BaseModel):
    id: int
    labels: Set[str]
```

We just have to inherit from the `pydantic.BaseModel` class. Pydantic will
then go and do all of the necessary serialization when our API returns this
`ImageAnalysisResponse` to the user.

These will make a bit more sense when we begin to use them in our routers, so
let's get to it!

## Routers
I now want to dive in to the top level or the entrypoint of our API. The 
entrypoint of APIs are typically their routes (aka endpoints). Our API is
going to support two endpoints:

1. Upload an image and run yolo on it
2. Download an image that yolo annotated for us

Let's start with by creating our router:

```python
# For API operations and standards
from fastapi import APIRouter, UploadFile, Response, status, HTTPException
# Our detector objects
from yolofastapi.detectors import yolov8
# For encoding images
import cv2
# For response schemas
from yolofastapi.schemas.yolo import ImageAnalysisResponse

# A new router object that we can add endpoints to.
# Note that the prefix is /yolo, so all endpoints from
# here on will be relative to /yolo
router = APIRouter(tags=["Image Upload and analysis"], prefix="/yolo")
```

So, we do some standard imports. We will also make a new router
object which we will add our two endpoints to. Note that the router
will be prefixed with `/yolo`. For example, curl calls would be 

```shell
curl http://localhost/yolo
curl http://localhost/yolo/endpoint1
curl http://localhost/yolo/endpoint2
```

We will also keep a record of uploaded/annotated images in memory:

```python
# A cache of annotated images. Note that this would typically
# be some sort of persistent storage (think maybe postgres + S3)
# but for simplicity, we can keep things in memory
images = []
```

In a production setting, you would want some more robust/persistent storage.
A typical paradigm would be to push these to some sort of blob storage and then
keep their URLs in a database (think postgres). For simplicity, though, we will
just keep them in an indexible python array in memory.

Now, we can add our first endpoint!

```python
@router.post("/",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Successfully Analyzed Image."}
    },
    response_model=ImageAnalysisResponse,
)
async def yolo_image_upload(file: UploadFile) -> ImageAnalysisResponse:
    """Takes a multi-part upload image and runs yolov8 on it to detect objects

    Arguments:
        file (UploadFile): The multi-part upload file
    
    Returns:
        response (ImageAnalysisResponse): The image ID and labels in 
                                          the pydantic object
    
    Examlple cURL:
        curl -X 'POST' \
            'http://localhost/yolo/' \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'file=@image.jpg;type=image/jpeg'

    Example Return:
        {
            "id": 1,
            "labels": [
                "vase"
            ]
        }
    """
    contents = await file.read()
    dt = yolov8.YoloV8ImageObjectDetection(chunked=contents)
    frame, labels = await dt()
    success, encoded_image = cv2.imencode(".png", frame)
    images.append(encoded_image)
    return ImageAnalysisResponse(id=len(images), labels=labels)
```

Let's dissect this code. We are adding a new `POST` method to our
api at `/yolo/` (because the router is prefixed with `/yolo`). The route
will return an HTTP 201 with the response body of our `ImageAnalysisResponse`
schema. The route will also expect, as input, a multi-part upload of an image.
When we enter this function, we will first read the image and then pass it to 
our `YoloV8ImageObjectDetection` object (which we will discuss in the next section).
We then use the callable `YoloV8ImageObjectDetection` object to run our analysis,
encode the image in `png` format, and save it in our in-memory array. Finally, we 
return an `ImageAnalysisResponse` object with the id and any detected labels filled
out. At this point, we can successfully upload/analyze/save images in our application.

Let's add one more endpoint to download the images:

```python
@router.get(
    "/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Image ID Not Found."}
    },
    response_class=Response,
)
async def yolo_image_download(image_id: int) -> Response:
    """Takes an image id as a path param and returns that encoded
    image from the images array

    Arguments:
        image_id (int): The image ID to download
    
    Returns:
        response (Response): The encoded image in PNG format
    
    Examlple cURL:
        curl -X 'GET' \
            'http://localhost/yolo/1' \
            -H 'accept: image/png'

    Example Return: A Binary Image
    """
    try:
        return Response(content=images[image_id - 1].tobytes(), media_type="image/png")
    except IndexError:
        raise HTTPException(status_code=404, detail="Image not found") 
```

We are adding a new `GET` method to our router at `/yolo/<id>`. The route will
return an HTTP 200 if the image ID is in our array, otherwise, it will return a 404.
The body of the response will be a binary PNG image. The application code is a bit
easier here, as all we have to do is index our array and return the encoded image.

So, up to now, we have our two endpoints and are ready to dive into the most
heavy piece of code - our YOLO detector.

## Detectors

## Putting It All Together

## Running

## Testing

## References