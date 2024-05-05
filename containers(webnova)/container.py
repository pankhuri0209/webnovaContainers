import cv2
import numpy as np

original = cv2.imread("../input_queue/containers_1.png", cv2.IMREAD_GRAYSCALE)
image_to_compare = cv2.imread("../input_queue/containers_2.png", cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, desc1 = sift.detectAndCompute(original, None)
kp2, desc2 = sift.detectAndCompute(image_to_compare, None)

# Initialize BFMatcher
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.match(desc1, desc2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Set a threshold for matching distance
threshold = 0.7 * matches[-1].distance

# Identify keypoints with significant differences
differences = []
for match in matches:
    if match.distance > threshold:
        differences.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))

# Draw circles around keypoints with significant differences on the original image
for point1, point2 in differences:
    x1, y1 = np.int32(point1)
    x2, y2 = np.int32(point2)
    cv2.circle(original, (x1, y1), 5, (0, 0, 255), 2)
    cv2.circle(image_to_compare, (x2, y2), 5, (0, 0, 255), 2)

# Save images with differences marked
cv2.imwrite("../output_queue/original_with_differences.png", original)
cv2.imwrite("../output_queue/compared_with_differences.png", image_to_compare)

# Write number of significant differences to output file
with open("../output_queue/results.txt", "w") as f:
    f.write("Number of significant differences: " + str(len(differences)))

print("Number of significant differences:", len(differences))
