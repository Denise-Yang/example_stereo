import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray, set_axes_equal

baseDir = 'data' #data directory
resDir = 'submit'
useLowRes = False #enable lowres for debugging
dX = 558.8 #calibration plane length in x direction
dY = 303.2125 #calibration plane length in y direction
frogStart = [311,324]
frogEnd = [809,630]
vOffset = 250 
hOffset = 190
yOffset = 650

upBound = 4000.
loBound = -2000.

#For extracting images from video 
def extract_images(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*250))    
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite( pathOut + "%d.jpg" % count, image)     
        count = count + 1

def get_shadow_im(ims):
    max = np.max(ims, axis=0)
    min = np.min(ims, axis=0)
    return (max + min)/2

"""
x1 y1 1     a/c     0
x2 y2 1     b/c     0
x3 y3 1     1       0
"""
def get_edge(im,h,w, vOffset, hOffset):
    #get both positive and negative edge and make sure accurate for all frames
    im_shifted = np.roll(im,-1,axis=1)
    im_shifted[:,w-1]  = im[:,w-1]
    edge = np.where(im_shifted - im > .045, 1, 0)
    
    points = np.transpose(np.nonzero(edge))
    pointsx = points[:,1] + hOffset
    pointsy = points[:,0] + vOffset
    points = np.dstack((pointsy,pointsx))[0]
    rows,_ = np.shape(points)
    ones = np.array([np.ones(rows)]).T
    
    A1 = np.append(points, ones, 1)
    A = points
    b = np.ones(rows)*-1
    s = np.linalg.lstsq(A,b, rcond=None)[0]
    
    m = -s[0]/s[1]
    c = -1/s[1]
    # if (h > 95):
    #     x1 = points[0,1]
    #     print(x1)
    #     x2 =  points[len(points)-1,1]

    #     xs,ys = [x1, x2], [m*x1+c,m*x2+c]
    #     # x1= [points[0,1], points[len(points)-1,1]]
    #     # y1= [points[0,0], points[len(points)-1,0]]
    #     # plt.imshow(edge)
    #     plt.plot(xs,ys, marker="o")
    #     plt.show()
    return (m,c)


    
def cam2Plane(point, translate, rotation):
    #might need to take transpose here as well
    return np.matmul(rotation.T, (np.array(point)-translate.flatten()))


def plane2Cam(point, translate, rotation):
    R_i = np.linalg.inv(rotation.T)
    return translate.flatten() + np.matmul(R_i, point.T)

def loadCallibrationValues():
    with np.load(os.path.join(baseDir, "intrinsic_calib.npz")) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    with np.load(os.path.join(baseDir, "extrinsic_calib.npz")) as X:
        t_h, m_h, t_v, m_v = [X[i] for i in ('t_h', 'm_h','t_v', 'm_v')]
    print(t_h, "\n", t_v)
    return mtx, dist, t_h, m_h, t_v, m_v

def getShadowPlane(v_line, h_line, mtx, dist, t_h, m_h, t_v, m_v,idx, delta_im,h):

    hm,hc = h_line
    vm,vc = v_line
    yv0 = 0
    yv1 = 100
    yh0 = yOffset
    yh1 = yOffset+100
    p1 = [ hm*(yh0-yOffset)+hc, yh0]
    p2 = [ hm*(yh1-yOffset)+hc, yh1]
    p3 = [ vm*yv0+vc, yv0]
    p4 = [ vm*yv1+vc, yv1]
    pts = np.array([p1,p2,p3,p4])

    # if (idx > 70):

    #     xs,ys = [vm*yv1+vc,vm*yv0+vc],  [yv1, yv0]
    #     xh,yh = [hm*(yh1-yOffset)+hc,hm*(yh0-yOffset)+hc],  [yh1, yh0]
    #     plt.imshow(delta_im)
    #     plt.plot(xs,ys, c='b', marker="o")
    #     plt.plot(xh,yh, c='r', marker="o")
    #     plt.show()


    #Use pixel2Ray to convert to cam world aka the dir
    cam_rays = pixel2ray(pts, mtx, dist)
    #convert pt and ray to H for p1,p2
    #calculate intersection where Z=0 to get P1, P2
    oh = cam2Plane([0,0,0], t_h, m_h) #camera origin w respect to horizontal plane
    ov = cam2Plane([0,0,0], t_v, m_v) #camera origin w respect to vertical plane
    Ps = []
    for i in range(4):
        r0 = cam_rays[i]
        if i < 2:

            r = np.matmul(m_h.T, r0[0]) #cam2Plane(r0, t_h, m_h)
            t = -oh[2]/r[2]
            Ps.append(plane2Cam(oh.T - r*t, t_h, m_h))
        else:
            r = np.matmul(m_v.T, r0[0])
            t = -ov[2]/r[2] 
            Ps.append(plane2Cam(ov.T - r*t, t_v, m_v))
    # if(idx > 40):
    #     y0 = 0
    #     y1 = 100
    #     y2 = 650
    #     y3 = 750
    #     xs,ys = [vm*y1+vc,vm*y0+vc],  [y1, y0]
    #     xh,yh = [hm*(y2)+hc,hm*(y3)+hc],  [y2, y3],
    #     z = [0,0]
    #     plt.imshow(delta_im)
    #     print(i)
    #     plt.plot(xs,ys, c='b', marker="o")
    #     plt.plot(xh,yh, c='r', marker="o")
    #     plt.show()
    #     # drawPlane(Ps[0].T[0], Ps[1].T[0],Ps[2].T[0],Ps[3].T[0])
    #     p1 = Ps[0].T[0]
    #     p2 = Ps[1].T[0]
    #     p3 = Ps[2].T[0]
    #     p4 = Ps[3].T[0]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zs=[p1[2], p2[2]], c='b') 
    #     ax.plot([p2[0],p3[0]], [p2[1], p3[1]], zs=[p2[2], p3[2]], c='r') 
    #     ax.plot([p3[0], p4[0]], [p3[1], p4[1]], zs=[p3[2], p4[2]], c='m') 
    #     ax.plot([p4[0], p1[0]], [p4[1], p1[1]], zs=[p4[2], p1[2]], c='y') 
    #     ax.plot(xs,ys,zs=z, c='g')
    #     ax.plot(xh,yh,zs=z, c='c')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.show()
    return Ps

def show3d(oneDArray,im):
    oneArray  = np.array(oneDArray)
    mask = np.copy(oneDArray)
    oneDArray = oneDArray[(mask[:,2]<upBound) * (mask[:,2]>loBound)]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    im = im[(mask[:,2]<upBound) * (mask[:,2]>loBound)]
    ax.scatter3D(oneDArray[:,0], oneDArray[:,1], oneDArray[:,2],c=oneDArray[:,2],cmap='gray')
    plt.title("Reconstruction")
    plt.show()
    

def drawPlane(p1,p2,p3,p4):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zs=[p1[2], p2[2]], c='b')
    ax.plot([p2[0],p3[0]], [p2[1], p3[1]], zs=[p2[2], p3[2]], c='r')
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], zs=[p3[2], p4[2]], c='m')
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], zs=[p4[2], p1[2]], c='y')

    # ax.scatter3D(intersect[0], intersect[1],intersect[2], c=intersect[2], marker="s", label='first')
    plt.legend(loc='upper left')
    plt.show()
    

def writePlanesAndNorms(Ps):
    Ps = np.array(Ps)
    P1 = Ps[:,0]
    P2 = Ps[:,1]
    P3 = Ps[:,2]
    P4 = Ps[:,3]
    u1 = P2-P1
    u2 = P4-P3
    n = np.cross(u1,u2)
    mag = np.array(np.linalg.norm(n,axis=1))
    h = mag.shape
    div = np.repeat(mag, 3).reshape((h[0],3))
    n_norm  = n/div
    np.savez(os.path.join(resDir, "shadowParameters.npz"), normals=n_norm, p1=P1)

def getShadowPlaneIntersection(rays, planes, shadow_time, indices):
    Y = indices[:,:,0]
    X = indices[:,:,1]
    shadows = shadow_time[Y,X]
    Ps = np.take(planes, shadows, axis=0) #np.array(np.take(planes, indices))
    h,w,pts,dim = Ps.shape
    Ps = Ps.reshape((h,w,pts,dim))
    rs = rays[Y,X] #np.array(np.take(rays, indices))
    

    P1 = Ps[:,:,0]
    P2 = Ps[:,:,1]
    P3 = Ps[:,:,2]
    P4 = Ps[:,:,3]
    
    u1 = P2-P1
    u2 = P4-P3
    n = np.cross(u1,u2)
    mag = np.array(np.linalg.norm(n,axis=2))
    h,w = mag.shape
    div = np.repeat(mag, 3).reshape((h,w,3))
    n_norm  = n/div

    times = np.sum(P1*n_norm, axis=2)/np.sum(rs*n_norm, axis=2)
    times = np.repeat(times, 3).reshape(h,w,3)
    # for i in range(100,200):
    #     for j in range(200,300):
    #         drawPlane(P1[i][j],P2[i][j],P3[i][j],P4[i][j])
    return rs*times
    


def createThresholdMask(ims):
    diff = np.max(ims, axis=0) - np.min(ims, axis=0)
    
    return (diff > .200) * (diff < .6)
    # return np.where((ims > 50 ) * (ims < 140), 1, 0)

def main():
    ims = []
    total_frames = 166
    for i in range(1,166+1):
        if i < 10:
            im_no = "00" + str(i)
        elif i < 100:
            im_no = "0" + str(i)
        else:
            im_no  = str(i)
        im = rgb2gray(skimage.io.imread("data/frog/" + im_no+ ".jpg"))
        ims.append(np.array(im))
    print("...done loading")
    
    shadow = get_shadow_im(ims)
    ims = np.array(ims)
    h,w = ims[0].shape
   
    frames = []

    mtx, dist, t_h, m_h, t_v, m_v = loadCallibrationValues()

    shadow_planes = []
    #get shadow planes
    for i in range(len(ims)):
        delta_im = ims[i] - shadow 
        frames.append(delta_im)
        #Get the unobstructed horizontal regions
        cropped_h =  delta_im[yOffset:h, hOffset:837]
        hh,wh = cropped_h.shape
        h_edge = get_edge(cropped_h,i,wh, 0,hOffset)

        # # Get the unobstructed vertical regions
        cropped_v =  delta_im[0:320, vOffset:780]
        hv,wv = cropped_v.shape
        v_edge = get_edge(cropped_v, i, wv, 0,vOffset)
        shadow_planes.append(getShadowPlane(v_edge, h_edge, mtx, dist, t_h, m_h, t_v, m_v,i,delta_im,h))
        

    
    shadow_planes_flat= np.array(shadow_planes).reshape((166*4,3))
  
    # show3d(shadow_planes_flat, None)
    # np.savez(os.path.join(resDir, "shadowPlanePoints.npz"), planes=shadow_planes)


    #get shadow time
    shadow_vals = np.zeros((h,w))
    zero_mask = np.zeros((h,w))
    shadow_time = np.ones((h,w))*300
    for i in range(10,len(frames)):
        if i == len(frames)-1:
            break
        zero_crossing = (frames[i]< 0) + (frames[i+1] > 0)
        zero_mask += (frames[i]< 0)
        diffs = frames[i+1] - frames[i] 
        shadow_time =  np.where((zero_crossing ==0) &(diffs < shadow_vals), i , shadow_time)
        shadow_vals = np.where((zero_crossing ==0) &(diffs < shadow_vals), diffs , shadow_vals)
    zero_mask = np.where((zero_mask>30), 0, 1)
    shadow_time = shadow_time * zero_mask
    shadow_time = np.where(shadow_time == 300, 0, (shadow_time).astype("int64"))



    rows = np.repeat(np.arange(h).reshape((h,1)),w).reshape((h,w))
    cols  = np.tile(np.arange(w),(h,1))

    #get cropped indices
    idx =  np.dstack((rows,cols))
    cropped_idx = idx[frogStart[1]:frogEnd[1], frogStart[0]:frogEnd[0]]
    pixels_flat = idx.flatten().reshape((h*w,2)).astype("float32")
    rays_flat = pixel2ray(pixels_flat,mtx,dist)
    rays = np.reshape(rays_flat,(h,w,3))


    

    cropped_ims = ims[:,frogStart[1]:frogEnd[1], frogStart[0]:frogEnd[0]]
    shadow_time_cropped = shadow_time[frogStart[1]:frogEnd[1], frogStart[0]:frogEnd[0]]
    threshold_mask = createThresholdMask(cropped_ims).flatten()
    ch,cw = cropped_ims[0].shape
    dims = threshold_mask.shape
    #use cropped indices, get shadow time valye then use that valuye to get the plane, then proceed as normal
    intersections = getShadowPlaneIntersection(rays, shadow_planes, shadow_time, cropped_idx)
    writePlanesAndNorms(shadow_planes)
    intersections_flat = intersections.flatten().reshape((dims[0],3))
    validims = cropped_ims[0].flatten()
    valid_intersections = intersections_flat[threshold_mask]
    valid_colors = validims[threshold_mask]
    show3d(valid_intersections, valid_colors)
  




    # plt.imshow(shadow_time, cmap= "jet", interpolation=None)
    # plt.show()


  
if __name__ == "__main__":
    main()




    ##code graveyard
        #stuff for drawing shaodw edges on image
        # xs,ys = [vm*x1+vc,vm*x2+vc],  [x1, x2]
        # xh,yh = [hm*(x1-yOffset)+hc,hm*(x2-yOffset)+hc],  [x1, x2]

        # # x1= [points[0,1], points[len(points)-1,1]]
        # # y1= [points[0,0], points[len(points)-1,0]]
        # plt.imshow(delta_im)
        # print(i)
        # plt.plot(xs,ys, c='b', marker="o")
        # plt.plot(xh,yh, c='r', marker="o")
        # plt.show()