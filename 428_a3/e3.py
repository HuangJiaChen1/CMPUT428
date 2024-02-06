import cv2
import numpy as np

trackers = []
points = []
height = int(input("Height: "))
width = int(input("Width: "))
def select_point(event, x, y, flags, param):
    # On left mouse button click, record the point and initialize a tracker
    global first_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        bbox = (int(x-width/2), int(y-height/2), width, height)
        trackers.append(bbox)
        if len(points) == 4:
            print("All trackers initialized.")
def track(iter,x,y,w,h,last_dim,p,template):
    while iter <= max_iter:
        X = np.arange(x, x + w, dtype=np.float32) + p[0]
        Y = np.arange(y, y + h, dtype=np.float32) + p[1]
        X, Y = np.meshgrid(X, Y)
        warp = np.array([[1,0],[0,1]])
        I = cv2.remap(frame_gray, X, Y, cv2.INTER_LINEAR)
        dim = np.float32(I) - np.float32(template)

        if np.linalg.norm(last_dim - dim) <= 0.001:
            break
        last_dim = dim

        dy, dx = np.gradient(np.float32(I))
        A = np.dot(np.hstack((dx.reshape(-1, 1), dy.reshape(-1, 1))),warp)
        b = -dim.reshape(-1, 1)
        u = np.dot(np.linalg.pinv(A), b)
        p += u
        iter += 1
    return p


def midpoint(pt1, pt2, pt3, pt4):
    pt1 += (1,)
    pt2 += (1,)
    pt3 += (1,)
    pt4 += (1,)
    print(pt1)
    diagonal1 = np.cross(pt1,pt3)
    diagonal2 = np.cross(pt2,pt4)
    homo_midpoint = np.cross(diagonal1,diagonal2)
    midpoint_norm = homo_midpoint / homo_midpoint[2]
    midpoint = midpoint_norm[:2]
    return midpoint

# Setup video capture
cap = cv2.VideoCapture(0)



_,_ = cap.read()
_,_ = cap.read()
ret_val, first_frame = cap.read()
cv2.namedWindow("First Frame")
cv2.setMouseCallback("First Frame", select_point)
cv2.imshow('First Frame', first_frame)
cv2.waitKey()
p1 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
p2 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
p3 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
p4 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
ps = [p1,p2,p3,p4]
templates = []
last_dims = []
four_points = [0,0,0,0]
for i in range(4):
    x = trackers[i][0]
    y = trackers[i][1]
    w = trackers[i][2]
    h = trackers[i][3]
    X = np.arange(x, x + w, dtype=np.float32)
    Y = np.arange(y, y + h, dtype=np.float32)
    X, Y = np.meshgrid(X, Y)

    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    template = cv2.remap(first_frame_gray, X, Y, cv2.INTER_LINEAR)
    last_dim = np.zeros_like(template)
    templates.append(template)
    last_dims.append(last_dim)
max_iter = 50
while True:
    ret, frame = cap.read()
    if not ret:
        break
    i = 0
    # Update and draw trackers
    for tracker in trackers:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        u = np.array([0, 0])
        iter = 0
        x = tracker[0]
        y = tracker[1]
        w = tracker[2]
        h = tracker[3]
        last_dim = last_dims[i]
        template = templates[i]

        motion_vector = track(iter, x, y, w, h, last_dim, ps[i],template)
        ps[i] = motion_vector
        point = (x + int(motion_vector[0])+int(width/2), y + int(motion_vector[1])+int(height/2))
        four_points[i] = point
        cv2.rectangle(frame, (x + int(motion_vector[0]), y + int(motion_vector[1])), (x + int(motion_vector[0]) + w, y + int(motion_vector[1]) + h), (0, 255, 0), 2)

        i+=1
    # print(type(points))
    mp = midpoint(four_points[0],four_points[1],four_points[2],four_points[3])
    mp = (int(mp[0]), int(mp[1]))
    print(mp)
    cv2.circle(frame, mp, 5, (0, 255, 0), -1)
    cv2.imshow('Tracking', frame)
    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
