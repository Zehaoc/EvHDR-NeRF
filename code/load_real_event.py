import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from glob import glob

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'input_images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'input_images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'input_images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'input_images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'input_images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    if os.path.exists(os.path.join(basedir, 'poses_bounds_exps.npy')):
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds_exps.npy'))
        poses = poses_arr[:, :-3].reshape([-1, 3, 5]).transpose([1,2,0])
        bds = poses_arr[:, -3:-1].transpose([1,0])
        exps = poses_arr[:, -1:].transpose([1,0])
    else:
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        bds = poses_arr[:, -2:].transpose([1,0])
        _, img_num = bds.shape
        exps = np.full((1, img_num), 0.33333)
        
    # print('bds', bds.shape)
    # print('poses', poses.shape)
    # print('exps', exps.shape)
    img0 = [os.path.join(basedir, 'input_images', f) for f in sorted(os.listdir(os.path.join(basedir, 'input_images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'input_images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds, exps
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])

    return poses, bds, exps, imgs          
   

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt


def poses_avg(poses):
    bottom = np.reshape([0,0,0,1.], [1,4])
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), bottom], 0)
    
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    bottom = np.reshape([0,0,0,1.], [1,4])
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), bottom], 0))
    return render_poses
    

# def recenter_poses(poses):
#     poses_ = poses+0
#     bottom = np.reshape([0,0,0,1.], [1,4])
#     c2w = poses_avg(poses)
#     poses = np.linalg.inv(c2w) @ poses
#     poses_[:,:3,:4] = poses[:,:3,:4]
#     poses = poses_

#     return poses

def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

def load_syn_llff_data(basedir, half_res=False, testskip=1, bd_factor=0.75, max_exp=1, min_exp=1, near_depth=4.0, rand_seed=1, render_size=30):
    np.random.seed(rand_seed)
    splits = ['train', 'test']
    metas = {}
    exps_metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
        with open(os.path.join(basedir, 'exposure_{}.json'.format(s)), 'r') as fp:
            exps_metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_exps = []
    counts = [0]
    num_exps = 5
    for s in splits:
        meta = metas[s]
        exps_meta = exps_metas[s]
        imgs = []
        poses = []
        exps = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            if s == 'train':
                idx = np.random.choice([0, 2, 4]) # randomly select an exposure from {t_1, t_3, t_5} for each input view
                fname = os.path.join(basedir, frame['file_path'] + '_%d.png' % idx)
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
                exps.append(np.float(exps_meta[frame['file_path'] + '_%d.png' % idx]))
            if s == 'test':
                for i in range(num_exps):
                    fname = os.path.join(basedir, frame['file_path'] + '_%d.png' % i)
                    imgs.append(imageio.imread(fname))
                    poses.append(np.array(frame['transform_matrix']))
                    exps.append(np.float(exps_meta[frame['file_path'] + '_%d.png' % i]))

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        exps = np.array(exps).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_exps.append(exps)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(2)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    sc = 1. if bd_factor is None else 1./(near_depth * bd_factor)
    poses[:, :3, 3] *= sc
    near_depth *= sc
    poses = recenter_poses(poses)
    exps = np.concatenate(all_exps, 0).reshape([-1, 1])
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    zdelta = near_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), render_size, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2

    # Generate poses and exposures for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    render_exps = np.linspace(min_exp, max_exp, N_views//2) # the exposure denotes exposure value (EV) 
    render_exps = 2 ** render_exps
    render_exps = np.concatenate([render_exps, render_exps[::-1]])
    render_exps = np.reshape(render_exps, [-1, 1]).astype(np.float32)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, exps, render_poses, render_exps, [H, W, focal], i_split

def _load_event_data(basedir):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds_event.npy'))
    print('poses_arr', poses_arr.shape)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    
    # print('poses', poses.shape)
    event_lis = sorted(glob(os.path.join(basedir, 'event_npy/*.npy')))
    event_frame_lis = []
    for i in range(len(event_lis)):
        im_name = event_lis[i]
        event_frame = np.load(im_name, allow_pickle=True)
        event_frame_lis.append(event_frame)
    event_frame_lis = np.array(event_frame_lis)
    event_frame_lis = event_frame_lis[:, :, :, np.newaxis]
    print('event_frame_lis', event_frame_lis.shape)
    return poses, event_frame_lis
            

def load_real_event_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, max_exp=1, min_exp=1, near_depth=4.0, render_size=30):
    poses, bds, exp, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    # print('poses', poses.shape)
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)

    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    exp = np.moveaxis(exp, -1, 0).astype(np.float32) # [1, N]
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    # sc = 1. if bd_factor is None else 1./(near_depth * bd_factor)
    # poses[:, :3, 3] *= sc
    # near_depth *= sc
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    blur_event = np.zeros_like(images)[...,0:1]
    print('blur_event', blur_event.shape)
    # print(a)
    H = 260
    W = 346
    event_poses, event = _load_event_data(basedir)
    event_poses = np.concatenate([event_poses[:, 1:2, :], 
                            -event_poses[:, 0:1, :], 
                            event_poses[:, 2:, :]], 1)
    event_poses = np.moveaxis(event_poses, -1, 0).astype(np.float32)
    
    event_poses[:, :3, 3] *= sc
    i_ll = poses.shape[0]

    all_poses = np.concatenate((poses, event_poses), axis=0)
    all_poses = recenter_poses(all_poses)
    poses = all_poses[:i_ll]
    event_poses = all_poses[i_ll:]

    
    # event_poses_prev = event_poses.copy()
    # event_poses_prev = event_poses_prev[:-1]
    # event_poses = event_poses[1:]
    # event_poses = event_poses[4::5]
    # event_poses_prev = event_poses_prev[::5]

    # event = merge_event(event)
      
    # print('event', event.shape)
    # num_event = event.shape[0]
    # H = 260
    # W = 346
    # color_mask = np.zeros((H, W, 3))
    # color_mask[0::2, 0::2, 0] = 1  # r
    # color_mask[0::2, 1::2, 1] = 1  # g
    # color_mask[1::2, 0::2, 1] = 1  # g
    # color_mask[1::2, 1::2, 2] = 1  # b
    # color_mask = color_mask[np.newaxis, :, :, :]
    # color_masks = np.tile(color_mask, (num_event, 1, 1, 1)) 
    # e_exp = exp[0] * np.ones((num_event, 1))
    # print('event_poses_prev', event_poses_prev.shape)
    # print('event_poses', event_poses.shape)
    # print('color_mask', color_mask.shape)
    
    event_poses_prev = event_poses.copy()
    event_poses_prev = event_poses_prev[:-1]
    event_poses = event_poses[1:]
    
    
    num_event = event.shape[0]
    color_mask = np.zeros((H, W, 3))
    color_mask[0::2, 0::2, 0] = 1  # r
    color_mask[0::2, 1::2, 1] = 1  # g
    color_mask[1::2, 0::2, 1] = 1  # g
    color_mask[1::2, 1::2, 2] = 1  # b
    color_mask = color_mask[np.newaxis, :, :, :]
    color_masks = np.tile(color_mask, (num_event, 1, 1, 1))    
    e_exp = exp[0] * np.ones((num_event, 1))
    # print('event', event.shape)
    # print('event_poses_prev', event_poses_prev.shape)
    # print('event_poses', event_poses.shape)
    # print('color_mask', color_masks.shape)
    # print('exp', exp.shape)
    # print('e_exp', e_exp.shape)
    # print(a)
    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    zdelta = bds.min() * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), render_size, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    
    hwf = poses[0,:3,-1]
    H, W, focal = hwf
    # Generate poses and exposures for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    render_exps = np.linspace(min_exp, max_exp, N_views//2) # the exposure denotes exposure value (EV) 
    render_exps = 2 ** render_exps
    render_exps = np.concatenate([render_exps, render_exps[::-1]])
    render_exps = np.reshape(render_exps, [-1, 1]).astype(np.float32)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)    

    return images, poses, bds, exp, render_poses, render_exps, i_test, blur_event, event_poses, event_poses_prev, event, e_exp, color_masks



def merge_event(event):
    new_event = []
    for i in range(0, event.shape[0], 5):
        print(i, event.shape[0])
        sub_event = event[i:i + 5]
        m = np.sum(sub_event, axis=0)
        # print(m.shape)
        new_event.append(m)
    # print('event', event.shape)
    return np.array(new_event)
