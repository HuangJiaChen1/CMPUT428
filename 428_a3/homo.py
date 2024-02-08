import cv2
import numpy as np
from matplotlib import pyplot as plt


def DLT(pts,pts_new,n):
    A = np.zeros((2 * n, 9))
    for i in range(n):
        [x, y] = pts[i]
        x_prime, y_prime = pts_new[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x_prime * x, x_prime * y, x_prime]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime]
    return A
# pts = [(1,0)]
# pts_new = [(2,0)]
# A = DLT(pts,pts_new,1)
# U,S,VT = np.linalg.svd(A)
# H = VT[-1].reshape(3,3)
# print(H)
# old = np.float32([1,0,1])
# new = np.dot(H,old)
# print(new/new[2])
# def click_event1(event, x, y, flags, params):
#     # callback function for the first image window
#     if event == cv2.EVENT_LBUTTONDOWN:
#         pts.append((x, y))
#         cv2.circle(key1, (x, y), 5, (255, 0, 0), -1)
#         cv2.imshow('old', key1)
#
# def click_event2(event, x, y, flags, params):
#     # callback function for the second image window
#     if event == cv2.EVENT_LBUTTONDOWN:
#         pts_new.append((x, y))
#         cv2.circle(key3, (x, y), 5, (0, 255, 0), -1)
#         cv2.imshow('new', key3)


def normalize_points(pts):
    centroid = np.mean(pts, axis=0)
    translated_pts = pts - centroid

    avg_dist = np.mean(np.sqrt((translated_pts[:, 0] ** 2) + (translated_pts[:, 1] ** 2)))
    scale = np.sqrt(2) / avg_dist
    # normalized_pts = translated_pts * scale
    # T = np.array([
    #     [scale, 0, -scale * centroid[0]],
    #     [0, scale, -scale * centroid[1]],
    #     [0, 0, 1]
    # ])
    T = np.array([
        [centroid[0]+centroid[1], 0, centroid[0]/2],
        [0, centroid[0]+centroid[1], centroid[1]/2],
        [0, 0, 1]
    ])
    T = np.linalg.inv(T)
    normalized_pts = []
    for points in pts:
        points = np.append(points,1)
        print(points)
        normalized = np.dot(T,points)
        normalized = normalized / normalized[2]
        normalized_pts.append(normalized[:2])
    return normalized_pts, T

# pts = []
# pts_new = []
# key1 = cv2.imread('key1.jpg')
# key2 = cv2.imread('key2.jpg')
# key3 = cv2.imread('key3.jpg')
# cv2.imshow('old',key1)
# cv2.imshow('new',key3)
# cv2.setMouseCallback('old',click_event1)
# cv2.setMouseCallback('new',click_event2)
# cv2.waitKey(0)
# print(pts)
# print(pts_new)
# n = len(pts)
# print(n)
# pts = np.array(pts, dtype=np.float32)
# pts_new = np.array(pts_new, dtype=np.float32)
# pts,T_old = normalize_points(pts)
# pts_new, T_new = normalize_points(pts_new)
# A = DLT(pts,pts_new,n)
# U,S,VT = np.linalg.svd(A)
# H = VT[-1].reshape(3,3)
# H_denorm = np.dot(np.dot(np.linalg.inv(T_new),H),T_old)
# print(H_denorm)
# # pts = np.array(pts, dtype=np.float32)
# # pts_new = np.array(pts_new, dtype=np.float32)
# # M = cv2.getPerspectiveTransform(pts,pts_new)
# # dst = cv2.warpPerspective(key1,M,(key1.shape[0],key1.shape[1]))
# dst1 = cv2.warpPerspective(key1,H_denorm,(key1.shape[0],key1.shape[1]))
# # cv2.imshow('dst',dst)
# # cv2.waitKey(0)
# cv2.imshow('dst1',dst1)
# cv2.waitKey(0)


points = []
height = 10
width = 10
trackers = []
four_points = [(0,0),(0,0),(0,0),(0,0)]
def select_point(event, x, y, flags, param):
    # On left mouse button click, record the point and initialize a tracker
    global first_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        bbox = (int(x-width/2), int(y-height/2), width, height)
        tracker = cv2.TrackerCSRT.create()
        tracker.init(first_frame,bbox)
        trackers.append(tracker)
        if len(points) == 4:
            print("All trackers initialized.")
cap = cv2.VideoCapture(0)
_,_ = cap.read()
_,_ = cap.read()
ret_val, first_frame = cap.read()
cv2.namedWindow("First Frame")
cv2.setMouseCallback("First Frame", select_point)
cv2.imshow('First Frame', first_frame)
cv2.waitKey()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    i = 0
    # Update and draw trackers
    for tracker in trackers:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        success, bbox = tracker.update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0]+ bbox[2]), int(bbox[1]+bbox[3]))
            x = (p1[0]+p2[0])/2
            y = (p1[1]+p2[1])/2
            point = (int(x),int(y))
            cv2.circle(frame,point,5,(255,0,0),2)
            four_points[i] = point
        i+=1
    pts = np.array(points, dtype=np.float32)
    pts_new = np.array(four_points, dtype=np.float32)
    pts = np.array(pts, dtype=np.float32)
    pts_new = np.array(pts_new, dtype=np.float32)
    pts,T_old = normalize_points(pts)
    pts_new, T_new = normalize_points(pts_new)
    A = DLT(pts_new,pts,4)
    U,S,VT = np.linalg.svd(A)
    H = VT[-1].reshape(3,3)
    M = np.dot(np.dot(np.linalg.inv(T_old),H),T_new)
    dst1 = cv2.warpPerspective(frame, M, (frame.shape[0], frame.shape[1]))
    cv2.imshow('tracking',frame)
    cv2.imshow('homo',dst1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()