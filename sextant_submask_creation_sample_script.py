import h5py
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel, regionprops
import math

def keys(f):
    return [key for key in f.keys()]


# function to visualize T2 images, whole gland mask and sextant submasks
def mask_view(T2,whole,sextants,filepath):
    #determine z bounds of prostate mask
    z = np.any(whole, axis=(0, 1))
    zmin, zmax = np.where(z)[0][[0, -1]]

    #choose zslices
    zrange=np.linspace((zmin-2),(zmax+2),num=12).astype(np.int32)
    zrange=np.clip(zrange,0,(T2.shape[2]-1))

    #crop display view to cropped area near prostate in xy plane
    s = np.any(whole, axis=(1, 2))
    c = np.any(whole, axis=(0, 2))
    smin, smax = np.where(s)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    max_trv_diam=np.max([(smax-smin),(cmax-cmin)])
    xmin = int((smax - smin) / 2 + smin - (1.2 * max_trv_diam / 2))
    xmax = int((smax - smin) / 2 + smin + (1.2 * max_trv_diam / 2))
    ymin = int((cmax - cmin) / 2 + cmin - (1.2 * max_trv_diam / 2))
    ymax = int((cmax - cmin) / 2 + cmin + (1.2 * max_trv_diam / 2))
    xrange=np.array([xmin,xmax])
    xrange = np.clip(xrange, 0, (T2.shape[0] - 1))
    yrange=np.array([ymin,ymax])
    yrange = np.clip(yrange, 0, (T2.shape[0] - 1))

    fig, axes=plt.subplots(ncols=12, nrows=8, figsize=(36, 24), sharex=True, sharey=True)
    for i in list(range(12)):
        slic = zrange[i];
        ax0 = axes[0, i];
        ax1 = axes[1, i];
        ax2 = axes[2, i];
        ax3 = axes[3, i];
        ax4 = axes[4, i];
        ax5 = axes[5, i];
        ax6 = axes[6, i];
        ax7 = axes[7, i];
        ax0.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax0.set_title('T2 input: z={}'.format(slic));
        ax1.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax1.imshow(whole[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='Blues',alpha=0.5);
        ax1.set_title('Whole Gland');
        ax2.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax2.imshow(sextants[0,xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='Oranges',alpha=0.5);
        ax2.set_title('Sextant 0');
        ax3.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax3.imshow(sextants[1,xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='Oranges',alpha=0.5);
        ax3.set_title('Sextant 1');
        ax4.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax4.imshow(sextants[2,xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='Oranges',alpha=0.5);
        ax4.set_title('Sextant 2');
        ax5.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax5.imshow(sextants[3,xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='Oranges',alpha=0.5);
        ax5.set_title('Sextant 3');
        ax6.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax6.imshow(sextants[4,xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='Oranges',alpha=0.5);
        ax6.set_title('Sextant 4');
        ax7.imshow(T2[xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='gray');
        ax7.imshow(sextants[5,xrange[0]:xrange[1],yrange[0]:yrange[1], slic], cmap='Oranges',alpha=0.5);
        ax7.set_title('Sextant 5');
    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()
    plt.close()

# function to process sextant masks from whole gland mask
def sextant_mask(whole, T2_pixelspacing, z_spacing):
    # create right & left mask
    s = np.any(whole, axis=(1, 2))
    smin, smax = np.where(s)[0][[0, -1]]
    rightleft=int((smax+smin)/2)
    right_mask=np.zeros(whole.shape,dtype=int)
    right_mask[0:rightleft,:,:]=1
    left_mask=1-right_mask

    #create saggital mip, fit ellipse and define major axis and its orientation
    sagittal_mip=np.amax(whole,axis=0)
    sagittal_mip=sagittal_mip.astype(int)
    sagittal_mip_zoom = ndimage.zoom(sagittal_mip, [T2_pixelspacing, z_spacing], mode='nearest') # resample to 1x1x1
    props=regionprops(sagittal_mip_zoom)
    y0, x0 = props[0]['centroid']
    orientation=props[0]['orientation']
    eccentricity = props[0]['eccentricity']
    major_axis_length=props[0]['major_axis_length']

    #set bounds on orientation
    if orientation <-0.7:
        orientation=-0.7
    elif orientation>0:
        orientation=0

    #calculate points on major axis that are intercepts for division into thirds
    x_apex_mid_boundary = x0 - math.cos(orientation) * (1/6) * major_axis_length
    y_apex_mid_boundary = y0 + math.sin(orientation) * (1/6) * major_axis_length

    x_mid_base_boundary = x0 + math.cos(orientation) * (1/6) * major_axis_length
    y_mid_base_boundary = y0 - math.sin(orientation) * (1/6) * major_axis_length

    #define line perpindicular to major axis for division of prostate into thirds
    y_apex_mid_boundary_extent1 = y_apex_mid_boundary + math.sin(orientation+math.pi/2) * (1/2) * major_axis_length

    y_mid_base_boundary_extent1 = y_mid_base_boundary + math.sin(orientation+math.pi/2) * (1/2) * major_axis_length

    # create right & left mask
    s = np.any(whole, axis=(1, 2))
    smin, smax = np.where(s)[0][[0, -1]]
    rightleft = int((smax + smin) / 2)
    right_mask = np.zeros(whole.shape, dtype=int)
    right_mask[0:rightleft, :, :] = 1
    left_mask = 1 - right_mask

    # create saggital mip, fit ellipse and define major axis and its orientation
    sagittal_mip = np.amax(whole, axis=0)
    sagittal_mip = sagittal_mip.astype(int)
    sagittal_mip_zoom = ndimage.zoom(sagittal_mip, [T2_pixelspacing, z_spacing], mode='nearest')  # resample to 1x1x1
    props = regionprops(sagittal_mip_zoom)
    y0, x0 = props[0]['centroid']
    orientation = props[0]['orientation']
    eccentricity = props[0]['eccentricity']
    major_axis_length = props[0]['major_axis_length']

    # set bounds on orientation
    if orientation < -0.7:
        orientation = -0.7
    elif orientation > 0:
        orientation = 0

    # calculate points on major axis that are intercepts for division into thirds
    x_apex_mid_boundary = x0 - math.cos(orientation) * (1 / 6) * major_axis_length
    x_mid_base_boundary = x0 + math.cos(orientation) * (1 / 6) * major_axis_length

    y_apex_mid_boundary = y0 + math.sin(orientation) * (1 / 6) * major_axis_length
    y_mid_base_boundary = y0 - math.sin(orientation) * (1 / 6) * major_axis_length
    y_apex_mid_boundary_extent1 = y_apex_mid_boundary + math.sin(orientation + math.pi / 2) * (
                1 / 2) * major_axis_length
    y_mid_base_boundary_extent1 = y_mid_base_boundary + math.sin(orientation + math.pi / 2) * (
                1 / 2) * major_axis_length

    # unzoom x_apex_mid_boundary and x_mid_base_boundary (in z dimension)
    x_apex_mid_boundary_unzoom = x_apex_mid_boundary/z_spacing
    x_mid_base_boundary_unzoom = x_mid_base_boundary / z_spacing
    y_apex_mid_boundary_unzoom = y_apex_mid_boundary/T2_pixelspacing
    y_mid_base_boundary_unzoom = y_mid_base_boundary/T2_pixelspacing
    y_apex_mid_boundary_extent1_unzoom = y_apex_mid_boundary_extent1/T2_pixelspacing
    y_mid_base_boundary_extent1_unzoom = y_mid_base_boundary_extent1/T2_pixelspacing

    # #save image of ellipse that was fit, centroid, long axis and boundaries between apex, mid and base
    # plt.imshow(sagittal_mip, cmap=plt.cm.gray)
    # plt.plot(x_apex_mid_boundary_unzoom, y_apex_mid_boundary_unzoom, '.g', markersize=3)
    # plt.plot((x_apex_mid_boundary_unzoom, x_apex_mid_boundary_unzoom), (y_apex_mid_boundary_unzoom,
    #                                                                     y_apex_mid_boundary_extent1_unzoom),
    #         '-r', linewidth=1)
    # plt.plot(x_mid_base_boundary_unzoom, y_mid_base_boundary_unzoom, '.g', markersize=3)
    # plt.plot((x_mid_base_boundary_unzoom, x_mid_base_boundary_unzoom), (y_mid_base_boundary_unzoom,
    #                                                                     y_mid_base_boundary_extent1_unzoom),
    #         '-r', linewidth=1)
    # plt.title(('orientation:'+"{0:.2f}".format(orientation)+'\n eccentricity:'+"{0:.2f}".format(eccentricity)))
    # plt.tight_layout()

    #create submask for apex
    a=np.array([y_apex_mid_boundary_unzoom,x_apex_mid_boundary_unzoom])
    b=np.array([y_apex_mid_boundary_extent1_unzoom,x_apex_mid_boundary_unzoom])
    apex_mid_boundary_mask=np.empty(sagittal_mip.shape,dtype=int)
    for index,value in np.ndenumerate(apex_mid_boundary_mask):
        if np.cross((index-a),(b-a)) >0:
            apex_mid_boundary_mask[index]=1
        else:
            apex_mid_boundary_mask[index]=0
    apex_mask=sagittal_mip*apex_mid_boundary_mask

    # create submask for base
    a = np.array([y_mid_base_boundary_unzoom, x_mid_base_boundary_unzoom])
    b = np.array([y_mid_base_boundary_extent1_unzoom, x_mid_base_boundary_unzoom])
    mid_base_boundary_mask = np.empty(sagittal_mip.shape, dtype=int)
    for index, value in np.ndenumerate(mid_base_boundary_mask):
        if np.cross((index - a), (b - a)) < 0:
            mid_base_boundary_mask[index] = 1
        else:
            mid_base_boundary_mask[index] = 0
    base_mask = sagittal_mip * mid_base_boundary_mask

    # create mask for mid gland
    mid_mask = sagittal_mip * (1 - mid_base_boundary_mask) * (1 - apex_mid_boundary_mask)

    # broadcast 2D sag masks to 3D
    apex_mask_3D = np.broadcast_to(apex_mask, whole.shape)
    mid_mask_3D = np.broadcast_to(mid_mask, whole.shape)
    base_mask_3D = np.broadcast_to(base_mask, whole.shape)

    # create sextant specific masks
    left_apex_mask = np.multiply(whole, np.multiply(left_mask, apex_mask_3D))
    right_apex_mask = np.multiply(whole, np.multiply(right_mask, apex_mask_3D))
    left_mid_mask = np.multiply(whole, np.multiply(left_mask, mid_mask_3D))
    right_mid_mask = np.multiply(whole, np.multiply(right_mask, mid_mask_3D))
    left_base_mask = np.multiply(whole, np.multiply(left_mask, base_mask_3D))
    right_base_mask = np.multiply(whole, np.multiply(right_mask, base_mask_3D))

    sextants = np.stack(
        (left_apex_mask, left_mid_mask, left_base_mask, right_apex_mask, right_mid_mask, right_base_mask), axis=0)

    sextants=np.stack((left_apex_mask,left_mid_mask,left_base_mask,right_apex_mask,right_mid_mask,right_base_mask),axis=0)

    return sextants


# load in sample data
f=h5py.File('sample_data.hdf5','r')
T2 = f['T2'][:, :, :]
whole = f['whole_gland_mask'][:, :, :]

#calculate sextant masks
sextants = sextant_mask(whole, f.attrs['T2_pixelspacing'], f.attrs['z_spacing'])

#visualize sextant masks
mask_view(T2, whole, sextants, 'sample_submask_visualization.png')
