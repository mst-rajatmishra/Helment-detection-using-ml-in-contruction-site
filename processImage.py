import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

from skinDetector import SkinDetector

def openCloseMask(mask, iterations=2):
    # Create structural element
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

    # Perform opening and closing on the image to remove blobs and fill gaps
    newMask = mask.copy()
    for _ in range(iterations):
        newMask = cv2.morphologyEx(newMask, cv2.MORPH_OPEN, shape)
        newMask = cv2.morphologyEx(newMask, cv2.MORPH_CLOSE, shape)

    return newMask

def getContours(binary_img):
    # Find contours
    contours, _ = cv2.findContours(binary_img, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by size
    newContours = sorted(contours, key=cv2.contourArea, reverse=True)
    return newContours

def getSkinMask(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    skinD = SkinDetector(image)
    skinD.findSkin() 

    skinMask = skinD.getMask()
    skinMask = openCloseMask(skinMask)  
    return skinMask

def preProcess(img):
    image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    smoothImg = cv2.fastNlMeansDenoising(image, h=6)   # Noise removal
    return smoothImg

def combineBoundingBox(box1, box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[2], box2[2])
    h = max(box1[3], box2[3])

    return (x, y, w, h)

def touchingRect(box1, box2):
    return (
        box1[0] < box2[0] + box2[2] and
        box1[0] + box1[2] > box2[0] and
        box1[1] < box2[1] + box2[3] and
        box1[1] + box1[3] > box2[1]
    )

def containsRect(box1, box2):
    x, y, w, h = box1
    x2, y2, w2, h2 = box2
    return (
        (x >= x2 and x <= x2 + w2 and y >= y2 and y <= y2 + h2) or
        (x <= x2 and x >= x2 + w2 and y <= y2 and y >= y2 + h2)
    )

def getFaces(img, skinMask):
    image = img.copy()
    contours = getContours(skinMask)

    newRects = []
    largestArea = cv2.contourArea(contours[0])

    # Discard irrelevant contours (5x smaller than the biggest area contours)
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area > largestArea * 0.20:
            newRects.append(cv2.boundingRect(contours[c]))

    # Merge boxes into one
    mergedRects = []
    for i in range(len(newRects)):
        for j in range(i + 1, len(newRects)):
            if touchingRect(newRects[i], newRects[j]):
                newBox = combineBoundingBox(newRects[i], newRects[j])
                if newBox not in newRects:
                    mergedRects.append(newBox)
                newRects.append(newBox)

    # Nullify rect if it's a child of another rect
    for i in range(len(mergedRects)):
        for j in range(i + 1, len(mergedRects)):
            if containsRect(mergedRects[i], mergedRects[j]):
                area = mergedRects[i][2] * mergedRects[i][3]
                area1 = mergedRects[j][2] * mergedRects[j][3]
                if area > area1:
                    mergedRects[j] = (0, 0, 0, 0)
                elif area1 > area: 
                    mergedRects[i] = (0, 0, 0, 0)

    # Fallback to base face rects if no merged boxes
    if not mergedRects:
        mergedRects = newRects

    # Remove any rectangles directly below
    for i in range(len(mergedRects)):
        for j in range(i + 1, len(mergedRects)):
            curr = mergedRects[i]
            comp = mergedRects[j]
            if comp[0] >= curr[0] and comp[0] + comp[2] <= curr[0] + curr[2] and comp[1] >= curr[1]:
                mergedRects[j] = (0, 0, 0, 0)

    faces = []
    for r in mergedRects:
        if r != (0, 0, 0, 0):
            x, y, w, h = r
            newY = max(0, y - int(1.2 * h))
            left = max(0, x - int(w * 0.2))
            width = w + int(w * 0.5)
            height = int(2.2 * h)
            faces.append((left, newY, width, height))

    return faces

def processHelmet(img):
    h, w = img.shape[:2]
    area = h * w

    hsvImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    helmetColors = [
        ((56, 3, 133), (116, 255, 241)),  # Green
        ((15, 0, 180), (115, 37, 236))    # White
    ]

    isHelmet = False
    for color in helmetColors:
        try: 
            lower, upper = color

            helmet_mask = cv2.inRange(hsvImage, lower, upper)
            finalMask = openCloseMask(helmet_mask, 4)

            rect = cv2.boundingRect(getContours(finalMask)[0]) + finalMask.std()
            helmetArea = rect[2] * rect[3]

            percentage = float(helmetArea / area) * 100

            if percentage >= 39.0:
                isHelmet = True
        except:
            pass
    return isHelmet

def getHelmets(img, skinMask, faces):
    image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    roi_img = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(skinMask))
    roi_img[np.all(roi_img >= 250, axis=2)] = 0
    roi_img[np.all(roi_img <= 150, axis=2)] = 0

    for f in faces:
        faceArea = roi_img[f[1]:f[1] + f[3], f[0]:f[0] + f[2]]
        isHelmet = processHelmet(faceArea)

        if isHelmet:
            cv2.rectangle(image, f, color=(0, 255, 0), thickness=6)
        else:
            cv2.rectangle(image, f, color=(255, 0, 0), thickness=6)

    return image

def process(img):
    preImg = preProcess(img)
    skinMask = getSkinMask(preImg)
    foundFaces = getFaces(preImg, skinMask)
    helmetImg = getHelmets(img, skinMask, foundFaces)

    return helmetImg
