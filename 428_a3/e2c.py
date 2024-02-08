import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

trackers = []
points = []
height = int(input("Height: "))
width = int(input("Width: "))
def cross_prod(a, b):
    result = [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]
    result = np.float32(result)
    return result
def select_point(event, x, y, flags, param):
    # On left mouse button click, record the point and initialize a tracker
    global first_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append((x, y))
        bbox = (int(x-width/2), int(y-height/2), width, height)
        tracker = cv2.TrackerCSRT.create()
        tracker.init(first_frame,bbox)
        trackers.append(tracker)
        if len(points) == 8:
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
p5 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
p6 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
p7 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
p8 = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
ps = [p1,p2,p3,p4,p5,p6,p7,p8]
templates = []
last_dims = []
eight_points = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
max_iter = 50
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
            cv2.rectangle(frame, p1,p2,(255,0,0),2,1)
            x = (p1[0]+p2[0])/2
            y = (p1[1]+p2[1])/2
            point = (int(x),int(y),1)
            eight_points[i] = point
        i+=1
    # print(type(points))
    for i in range(len(eight_points)):
        eight_points[i] += (1,)
    # print(eight_points[0])
    line1 = cross_prod(eight_points[0],eight_points[1])
    # print(line1)
    line2 = cross_prod(eight_points[2],eight_points[3])
    line3 = cross_prod(eight_points[4],eight_points[5])
    line4 = cross_prod(eight_points[6],eight_points[7])
    cv2.line(frame,eight_points[0][:2],eight_points[1][:2],(0,255,0),2)
    cv2.line(frame, eight_points[2][:2], eight_points[3][:2], (0, 255, 0), 2)
    cv2.line(frame, eight_points[4][:2], eight_points[5][:2], (0, 255, 0), 2)
    cv2.line(frame, eight_points[6][:2], eight_points[7][:2], (0, 255, 0), 2)
    inf1 = cross_prod(line1,line2)
    inf2 = cross_prod(line3,line4)
    epar = cross_prod(line2,line3)
    epar = epar/ epar[2]
    y1 ,y2,y3,y4 = np.float32(eight_points[2:6])
    ell = np.add(np.dot(y1,cross_prod(y3,y4)) , np.dot(y2,cross_prod(y3,y4)))
    ell =ell/ell[2]
    # pil_im = Image.fromarray(frame)
    # draw = ImageDraw.Draw(pil_im)
    # font = ImageFont.truetype("Roboto-Regular.ttf", 50)
    # draw.text((0, 0), f'{epar}', font=font)
    print(epar)
    print(ell)
    print(np.linalg.norm(epar))
    print(np.linalg.norm(ell))
    cv2.imshow('Tracking', frame)
    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
