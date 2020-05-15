# coding: UTF-8
'''
Generate uv position map of 300W_LP.
'''
import os, sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
import argparse
from time import time
import matplotlib.pyplot as plt
import dlib

sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return uv_coords

def get_colors(image, vertices):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    [h, w, _] = image.shape
    vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w -1)  # x
    vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h -1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:,1], ind[:,0], :] # n x 3
    
    return colors

def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y] 
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

def render_texture_simple(vertices, colors, triangles, h, w, c = 3, BG = None):
    ''' render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width    
    '''
    # initial 
    if BG is None:
        image = np.zeros((h, w, c))
    else:
        image = np.array(BG)
    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
    tri_tex = (colors[:, triangles[0,:]] + colors[:,triangles[1,:]] + colors[:, triangles[2,:]])/3.

    for i in range(triangles.shape[1]):
        tri = triangles[:, i] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u,v], vertices[:2, tri]): 
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image
def write_obj_with_colors(obj_name, vertices, triangles):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

def transform_vertices(M, vertices):
    v_size = vertices.size()
    R = M[:, :2]
    t = M[:, 2]
    vertices2 = vertices.clone()
    vertices2 = vertices2.float()
    vertices2[:2, :] = R.mm(vertices2[:2, :]) + t.repeat(v_size[1], 1).t()
    return vertices2

def run_posmap_300W_LP(bfm, image_path, mat_path, save_folder, idx=0, uv_h=256, uv_w=256, image_h=256, image_w=256):
    # 1. load image and fitted parameters
    image_name = image_path.strip().split('/')[-1]
    image_ori = io.imread(image_path)
    image = image_ori/255.
    [h, w, c] = image.shape
    #cv.imshow('ori',image_ori)
    info = sio.loadmat(mat_path)
    shape_para = info['Shape_Para'].astype(np.float32)
    exp_para = info['Exp_Para'].astype(np.float32)
    pose_para = info['new_pose'].T.astype(np.float32)
    tp = bfm.get_tex_para('random')
    colors = bfm.generate_colors(tp)
    # print('colb: ',colors)
    colors = np.minimum(np.maximum(colors, 0), 1)

    # ----------------------------if use dlib
    # dlib_landmark_model = './models/shape_predictor_68_face_landmarks.dat'
    # face_regressor = dlib.shape_predictor(dlib_landmark_model)
    # face_detector = dlib.get_frontal_face_detector()
    # rects = face_detector(image_ori, 1)
    # pts = face_regressor(image_ori, rects[0]).parts()
    # pts = np.array([[pt.x, pt.y] for pt in pts]).T
    # x = pts.T
    # fitted_sp, fitted_ep, s, angles, t = bfm.fit(x, bfm.kpt_ind, max_iter = 100, isShow = False)
    # ---------------------------------------

    # generate shape
    vertices = bfm.generate_vertices(shape_para, exp_para)
    
    # # transform mesh
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]
    #print('angle: ',angles)
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1
    
    # --------------show render pic
    # colors = get_colors(image_ori,image_vertices)
    # print('colorso: ',colors)
    # ccc = []
    # for i in range(len(colors)):
    #     #print('aa ',float(colors[i][0]) / float(255.0))
    #     a = float(colors[i][0]) / float(255.0)
    #     b = float(colors[i][1]) / float(255.0)
    #     c = float(colors[i][2]) / float(255.0)
    #     ccc.append([a,b,c])
    # #colors = np.minimum(np.maximum(colors, 0), 1)
    # c_color = np.array(ccc)
    # fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, c_color, h, w)
    # cv.imshow('fitti',fitted_image)
    # cv.waitKey(0)
    #-----------------------

    # 3. crop image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0,
                       bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top) / 2
    size = int(old_size * 1.5)
    # random pertube. you can change the numbers
    marg = old_size * 0.1
    t_x = np.random.rand() * marg * 2 - marg
    t_y = np.random.rand() * marg * 2 - marg
    center[0] = center[0] + t_x
    center[1] = center[1] + t_y
    size = size * (np.random.rand() * 0.2 + 0.9)

    # crop and record the transform parameters
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))

    # transform face position(image vertices) along with 2d facial image
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2] * tform.params[0, 0]  # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
    #write_obj_with_colors('./300w.obj',position,bfm.triangles)
    # 4. uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c=3)
    
    ####only for verify
    # uv_texture_map_rec = cv.remap(cropped_image, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT,borderValue=(0))
    # all_colors = np.reshape(uv_texture_map_rec, [256**2, -1])
    # face_ind = np.loadtxt('./data/face_ind.txt').astype(np.int32)
    # triangles = np.loadtxt('./data/triangles.txt').astype(np.int32)
    # all_vertices_gt = np.reshape(uv_position_map, [256**2, -1])
    # vertices_gt = all_vertices_gt[face_ind, :]
    # text_c = all_colors[face_ind, :]
    # print('vtx: ',vertices_gt.shape)
    # print('text_c: ',text_c.shape)
    # print('triangles: ',triangles.shape)
    # pic = render_texture_simple(vertices_gt.T,text_c.T,triangles.T,256,256)
    # cv.imshow('fittia',pic)
    # cv.imshow('cropped_image',cropped_image)
    # cv.waitKey(0)
    ## -----------------------


    io.imsave('{}/{}'.format(save_folder, image_name), np.squeeze(cropped_image))
    np.save('{}/{}'.format(save_folder, image_name.replace('jpg', 'npy')), uv_position_map)


def generate_batch_sample(input_dir, save_folder):
    uv_h = uv_w = 256

    # load uv coords
    global uv_coords
    uv_coords = face3d.morphable_model.load.load_uv_coords('./examples/Data/BFM/Out/BFM_UV.mat')  #
    uv_coords = process_uv(uv_coords, uv_h, uv_w)

    # load bfm
    bfm = MorphabelModel('./examples/Data/BFM/Out/BFM.mat')

    base = 0
    path_w = open('/media/weepies/Seagate Backup Plus Drive/3DMM/train_path_afw.txt','w')
    print('input: ',input_dir)
    for idx, item in enumerate(os.listdir(input_dir)):
        print('dealing')
        if 'jpg' in item:
            ab_path = os.path.join(input_dir, item)
            img_path = ab_path
            mat_path = ab_path.replace('jpg', 'mat')

            run_posmap_300W_LP(bfm, img_path, mat_path, save_folder, idx + base)
            name_p = image_name = img_path.strip().split('/')[-1]
            num_name = name_p.split('.jpg')
            path_w.write('/media/weepies/Seagate Backup Plus Drive/3DMM/posnet/300W_HELEN/'+name_p+'*'+ 
                '/media/weepies/Seagate Backup Plus Drive/3DMM/posnet/300W_HELEN/'+num_name[0]+'.txt'+'\n')
            # print("Number {} uv_pos_map was generated!".format(idx))
    path_w.close()


if __name__ == '__main__':
    save_folder = '/media/weepies/Seagate Backup Plus Drive/3DMM/posnet/300W_HELEN/'
    input_dir = '/media/weepies/Seagate Backup Plus Drive/3DMM/posnet/synthesize/'
    generate_batch_sample(input_dir,save_folder)