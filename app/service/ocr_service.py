import numpy as np
import imutils
import cv2
from PIL import Image
import os
import re
import requests
import uuid
import json
from io import BytesIO
from fastapi import HTTPException
from app.core.ocr_config import get_vietocr_predictor

detector = get_vietocr_predictor()

OCR_SCHEMA_DIR = 'ocr/schema/';
DOWLOADED_FILES_DIR = 'form-images/';
OCR_OUTPUT_DIR = 'ocr/ocr-output/';

def align_image(image, template, maxFeatures=500, keepPercent=0.2,
    debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the 'more similar' the features are)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
            matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow('Matched Keypoints', matchedVis)
        cv2.waitKey(0)
    
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype='float')
    ptsB = np.zeros((len(matches), 2), dtype='float')
    # loop over the top matches

    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, _) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    if debug :
        # resize both the aligned and template images so we can easily
        # visualize them on our screen
        aligned = imutils.resize(aligned, width=700)
        template = imutils.resize(template, width=700)
        # our first output visualization of the image alignment will be a
        # side-by-side comparison of the output aligned image and the
        # template
        stacked = np.hstack([aligned, template])
        
        overlay = template.copy()
        output = aligned.copy()
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        # show the two output image alignment visualizations
        cv2.imshow('Image Alignment Side By Side', stacked)
        cv2.imshow('Image Alignment Overlay', output)
        cv2.waitKey(0)

    # return the aligned image
    return aligned

def sharpen_image(image):
    rgb_planes = cv2.split(image)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        
    result_norm = cv2.merge(result_norm_planes)
    # tranform: convert into gray single channel
    gray_image = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def calculate_average_brightness(image, region):
    try:
        # Crop the region using the box position
        left, top, width, height = region
        cropped_image = image[top:top + height, left:left + width]

        # Convert the cropped region to numpy array
        region_array = np.array(cropped_image)

        # Calculate brightness for each pixel
        # Brightness = (Red + Green + Blue) / 3
        brightness = np.mean(region_array[..., :3])  # Only take R, G, B channels

        return brightness
    except Exception as e:
        print('[ocr-service]: read file fail:', e)
        return None

def retrieve_text(image, region):
    # Crop the region
    left, top, width, height = region
    cropped_image = image[top:top + height, left:left + width]

    # Convert cropped region to PIL Image (VietOCR requires PIL image)
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    text = detector.predict(cropped_pil)
    return text

def init_directories():
    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOWLOADED_FILES_DIR, exist_ok=True)

def get_files(downloading_urls):
    images = []
    try:
        for url in downloading_urls:
            # Send a GET request to the URL
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error if the request fails

            # Save image to memory (bytes array)
            with BytesIO(response.content) as image_data:
                # convert bytes array to OpenCV image
                img_array = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                images.append(img)
    except requests.RequestException as e:
        raise HTTPException(status_code=404, detail=f'Download failed: {str(e)}')
    
    return images

def scan_letter_by_letter(image_path, regions):
    text = ''
    for region_info in regions:
        char = retrieve_text(image_path, region_info['region'])
        text += char
    return text.rstrip()  # Remove trailing spaces

def get_checkbox_input(image, regions, brightness_threshold = 3):
    entries = []
    for region_info in regions:
        average_brightness = calculate_average_brightness(image, region_info['region'])
        if average_brightness is None:
            raise Exception('Read image file failed')
        if region_info['brightness'] - average_brightness > brightness_threshold:
            entries.append(region_info['entry'])
    return entries

async def do_ocr(image_urls, application_name):
    images = get_files(image_urls)

    schema_file_path = f"{OCR_SCHEMA_DIR}{application_name}.json"
    with open(schema_file_path, 'r', encoding='utf-8') as file:
        instruction = file.read()

    fields = json.loads(instruction)
    ocr_output = []

    for field in fields:
        image = images[int(field['page_number'])-1]
        if field['type'] == 'OCR_WORD':
            region = field['regions'][0]['region']
            ocr_text = retrieve_text(image, region)
            ocr_output.append({
                'name': field['name'],
                'field_type': field['type'],
                'data_type': field['data_type'],
                'text': ocr_text,
            })
        elif field.type == 'OCR_CHAR':
            ocr_text = scan_letter_by_letter(image, field['regions'])
            ocr_output.append({
                'name': field['name'],
                'field_type': field['type'],
                'data_type': field['data_type'],
                'text': ocr_text,
            })
        elif file.type == 'CHECK_BOX':
            entries = get_checkbox_input(image, field['regions'])
            ocr_output.append({
                'name': field['name'],
                'field_type': field['type'],
                'data_type': field['data_type'],
                'text': entries,
            })
    return ocr_output

init_directories();