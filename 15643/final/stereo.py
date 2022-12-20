import numpy as np
import cv2
import time
import skimage
from skimage import io, color, measure, draw
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from cp_hw5 import integrate_poisson, integrate_frankot, load_sources
from scipy import optimize
from sklearn.neighbors import NearestNeighbors

do_poisson = True
radius = 0
circle_threshold = [100,255]

#image cropping for watter bottle
sphere_cropX = [1230,1930]
sphere_cropY = [1970,2630]
obj_cropX = [2405,4923]
obj_cropY = [1444,2376]
img_name = "bottle"

#image cropping for peanut butter jar
# img_name = "peanut"
# sphere_cropX = [3967,4639]
# sphere_cropY = [1837,2503]
# obj_cropX = [1360,2700]
# obj_cropY = [283,2690]


def show3D(Z):
    H, W = Z.shape
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    # set 3D figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # add a light and shade to the axis for visual effect
    # (use the ‘-’ sign since our Z-axis points down)
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.cm.gray)
    # display a surface
    # (control surface resolution using rstride and cstride)
    surf = ax.plot_surface(x, y, -Z, facecolors=color_shade, rstride=5, cstride=5)
    # turn off axis
    plt.axis('off')
    plt.show()

def show(B,gray):
    if gray:
        plt.imshow(B, cmap='gray')
    else:
        plt.imshow(B)
    plt.show()

# Reshapes a 3xN matrix where N is the number of normals 
# to (h,w,3)
def extract_normals(B,h,w):
    # Code was from hw5 and has been removed 
    return B, B

# Finds a matrix Q such that QB is integrable, where B is a 
# matrix of pseudo normals 
def make_integrable(B,h,w):
    # Code was from hw5 and has been removed 
    return B


# Finds dx and dy of the normals 
# and applies poisson integration
def integrate(BQ,h,w, flip):
    # Code was from hw5 and has been removed 
    return BQ


def get_normals(sphere_im, obj_im, x0, y0, r):
    #first try with coloredd images
    print("starting matching point")
    start = time.perf_counter()
    h,w,d = sphere_im.shape
    neigh = NearestNeighbors(n_neighbors=1,algorithm='kd_tree')
    neigh.fit(sphere_im.reshape(h*w,d))
    h0,w0,d0 = obj_im.shape
    sample_pts = obj_im.reshape((h0*w0,d0))
    nearest_pts =  neigh.kneighbors(sample_pts, n_neighbors=1, return_distance=False)
    stop = time.perf_counter()
    Xs = np.mod(nearest_pts,w) 
    Ys = (nearest_pts/w).astype("int64") 

    X2 = np.power(Xs - x0 ,2)
    Y2 = np.power(Ys - y0,2)
    R2 = np.power(r,2) 
    Zs  = np.where(X2 + Y2 <= R2, np.sqrt(R2 - X2 - Y2), 0)
    print(f"neighbors took {stop - start:0.4f} seconds")
    return np.vstack((Xs.T,Ys.T,Zs.T))



#extract sphere from image
def extract_sphere(img):
    # Code modified from https://stackoverflow.com/questions/63001988/how-to-remove-background-of-images-in-python

    # threshold input image as mask
    mask = cv2.inRange(img,110,170)

    # negate mask
    mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
    return mask


def fit_sphere(img):
    img = np.where(img < 250, 0, 1)
    label, num = measure.label(img,return_num=True)
    regions = measure.regionprops(label)
    bubble = regions[0]

    y0, x0 = bubble.centroid
    r = bubble.major_axis_length/2

    def cost(params):
        x0, y0, r = params
        coords = draw.disk((y0, x0), r, shape=img.shape)
        template = np.zeros_like(img)
        template[coords] = 1
        return -np.sum(template == img)

    x0, y0, r = optimize.fmin(cost, (x0, y0, r))

    f, ax = plt.subplots()
    circle = plt.Circle((x0, y0), r)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.add_artist(circle)
    plt.show()
    print(x0,y0,r)
    return x0,y0,r



def main():
    ims = []
    h,w,c = skimage.io.imread("data/"+img_name+ str( 1)+".jpg").shape
    avg = np.zeros((h,w))
    for i in range(1,9):
        im = skimage.io.imread("data/"+img_name+ str(i)+".jpg")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ims.append(im)
        avg += im
    avg = avg/8
    all_ims = np.dstack(ims)
    
 
    obj_im = all_ims[obj_cropY[0]:obj_cropY[1],obj_cropX[0]:obj_cropX[1]]
    plt.imshow(avg)
    plt.show()
    sphere_samples =  all_ims[sphere_cropY[0]:sphere_cropY[1],sphere_cropX[0]:sphere_cropX[1]]
    
    sphere_im =  avg[sphere_cropY[0]:sphere_cropY[1],sphere_cropX[0]:sphere_cropX[1]]
    mask = extract_sphere(sphere_im)
    ##average all the sphere images together to extract the sphere dims
    x0,y0,r = fit_sphere(mask)
    normals = get_normals(sphere_samples, obj_im, x0, y0, r)
    h0,w0 = obj_im.shape[:2]
    normals, a, N = make_integrable(normals, h0,w0)
    extract_normals( normals, h0,w0)
    integrate(normals,h0,w0, True)




if __name__ == '__main__':
    main()