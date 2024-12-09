import logging
import numpy as np
import imutils
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import torch
import cv2

logger = logging.getLogger("app")

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
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
            matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imwrite("matchedVis.jpg", matchedVis)
        # cv2.imshow("Matched Keypoints", matchedVis)
        # cv2.waitKey(0)
    
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
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
        # cv2.imshow("Image Alignment Side By Side", stacked)
        cv2.imwrite("sideBySide.jpg", stacked)
        # cv2.imshow("Image Alignment Overlay", output)
        cv2.imwrite("overlay.jpg", output)
        # cv2.waitKey(0)

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

def retrieve_text(image_path):
    # load model
    config = Cfg.load_config_from_name('vgg_transformer') 
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['predictor']['beamsearch'] = False
    config['cnn']['pretrained'] = False

    detector = Predictor(config)
    image = cv2.imread(image_path)

    image = sharpen_image(image)

    cv2.imwrite("test.png", image)
    image_from_array = Image.fromarray(image)
    text = detector.predict(image_from_array)

    print("Kết quả OCR:", text)

if __name__ == "__main__":
    image = cv2.imread("temp/2.jpeg")
    template = cv2.imread("base-form/drop-out-school-application/1.jpg")
    image = align_image(image, template)

    image = sharpen_image(image)
    cv2.imwrite("demo.png", image)

