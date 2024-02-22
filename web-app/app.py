import base64
import concurrent.futures
import os
from io import BytesIO

import cv2
import gradio as gr
import numpy as np
import requests
import supervision as sv
from inference_sdk import InferenceHTTPClient
from openai import OpenAI

CLIENT = InferenceHTTPClient(
    api_url="http://detect.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

openai_client = OpenAI()


def process_mask(region, task_id):
    region = cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite(f"region_{task_id}.jpg", region)

    # change channels
    region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    base64_image = base64.b64encode(
        BytesIO(cv2.imencode(".jpg", region)[1]).read()
    ).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read the text on the book spine. Only say the book cover title and author if you can find them. Say the book that is most prominent. Return the format [title] [author], with no punctuation.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    print(response.choices[0].message.content.rstrip("Title:").replace("\n", " "))

    return response.choices[0].message.content.rstrip("Title:").replace("\n", " ")


def process_book_with_google_books(book):
    response = requests.get(
        f"https://www.googleapis.com/books/v1/volumes?q={book}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response = response.json()

    isbn, author, link = "NULL", "NULL", "NULL"

    try:
        isbn = response["items"][0]["volumeInfo"]["industryIdentifiers"][0][
            "identifier"
        ]
        if (
            "volumeInfo" in response["items"][0]
            and "authors" in response["items"][0]["volumeInfo"]
        ):
            author = response["items"][0]["volumeInfo"]["authors"][0]
        link = response["items"][0]["volumeInfo"]["infoLink"]
    except:
        pass

    return isbn, author, link


# define function that accepts an image
def detect_books(image):
    # infer on a local image
    results = CLIENT.infer(image, model_id="open-shelves/6")
    results = sv.Detections.from_inference(results)

    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image, detections=results)

    masks_isolated = []

    polygons = [sv.mask_to_polygons(mask) for mask in results.mask]

    for mask in results.mask:
        masked_region = np.zeros_like(image)
        masked_region[mask] = image[mask]
        masks_isolated.append(masked_region)

    print("Calculated masks...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        tasks = [
            executor.submit(process_mask, region, task_id)
            for task_id, region in enumerate(masks_isolated)
        ]
        books = [task.result() for task in tasks]

    print("Processed books...")

    links = []

    isbns = []
    authors = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(process_book_with_google_books, book) for book in books
        ]

        for task in tasks:
            isbn, author, link = task.result()
            isbns.append(isbn)
            authors.append(author)
            links.append(link)

    print("Processed books with Google Books...")

    annotations = [
        {
            "title": title,
            "author": author,
            "isbn": isbn,
            "polygons": [polygon.tolist() for polygon in polygon_list],
            "xyxy": xyxy.tolist(),
            "link": link,
        }
        for title, author, isbn, polygon_list, xyxy, link in zip(
            books, authors, isbns, polygons, results.xyxy, links
        )
        if "sorry" not in title.lower() and "NULL" not in title
    ]

    width, height = image.shape[1], image.shape[0]

    svg = f"""<div class="image-container"><img src="image.jpeg" height="{height}" width="{width}">
        
        <svg width="{width}" height="{height}">"""

    for annotation in annotations:
        polygons = annotation["polygons"][0]
        svg += f"""<polygon points="{', '.join([f'{x},{y}' for x, y in polygons])}" fill="transparent" stroke="red" stroke-width="2"
        onclick="window.location.href='{annotation['link']}';"></polygon>"""

    svg += "</svg>"
    svg += """
        <style>
        .image-container {
            position: relative;
            height: HEIGHTpx;
            width: WIDTHpx;
        }
        .image-container img, .image-container svg {
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: auto;
        }
        .image-container svg {
            z-index: 1;
        }
        </style></div>""".replace(
        "HEIGHT", str(height)
    ).replace(
        "WIDTH", str(width)
    )

    return annotated_image, books, isbns, svg


iface = gr.Interface(
    fn=detect_books,
    description="""
    Use Open Shelves to detect books in an image. The model will return the annotated image with the detected books, the titles of the books, and the ISBNs of the books.
    
    [View the project source code](https://github.com/capjamesg/open-shelves).

    [View the dataset on which the book segmentation model was trained](https://universe.roboflow.com/capjamesg/open-shelves).""",
    inputs=gr.components.Image(label="Input Image"),
    # outputs should be an image and a list of text
    outputs=[
        gr.components.Textbox(label="Detected Books", info="The detected books."),
        gr.components.Image(label="Annotated Image", type="pil"),
        gr.components.Textbox(label="ISBNs", info="The ISBNs of the detected books."),
        gr.components.Textbox(label="SVG", info="Copy-paste this code onto a web page to create a clickable bookshelf. NB: This code doesn't scale to different screen resolutions."),
    ],
    title="Open Shelves",
    allow_flagging=False,
    theme="huggingface",
)
iface.launch()
