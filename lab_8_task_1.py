import cv2 as cv
import sys

img = cv.imread('variant-10.jpg')

if img is None:
    sys.exit("Could not read the image.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 150, 255, 0)
cv.imshow('thresh', thresh)

cv.waitKey(0)
cv.destroyAllWindows()

