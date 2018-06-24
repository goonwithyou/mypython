import cv2 as cv

image = cv.imread('data/penguin.jpg')
cv.imshow('title', image)
resize = cv.resize(image, (200, 200))
cv.imshow('resize', resize)

cv.waitKey()

import sklearn.mixture as sm

help(sm.distribute_covar_matrix_to_match_covariance_type)
import sklearn.covariance
