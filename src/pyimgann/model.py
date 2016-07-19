import numpy as np
import logging
import pathlib as pl
import cPickle as pkl
from collections import defaultdict
from sortedcontainers import SortedSet
from PyQt4.QtCore import QObject, pyqtSignal

log = logging.getLogger("pyimgann.model")
log.setLevel(logging.DEBUG)

class PointSet2D(object):
    def __init__(self):
        self.xs_ = defaultdict(SortedSet)
        
    def add(self, pt):
        self.xs_[pt[0]].add(pt[1])
        
    def remove(self, pt):
        x = self.xs_[pt[0]]
        if x:
            x.discard(pt[1])
            
    def points(self):
        pts = []
        for x,v in self.xs_.iteritems():
            for y in v:
                pts.append(np.array([x,y]))
        return pts

class ImageAnnotationModel(object):
    def __init__(self):
        self.project_name = "new project"

    def save(self):
        log.info("save() is not implemented")

    def load(self, filename):
        log.info("load() is not implemented")

class Correspondence(object):
    def __init__(self, a, b):
        self.pts_ = np.array([a,b])
    
    def __len__(self):
        return len(self.pts_)

    def __getitem__(self, i):
        return self.pts_[i]

    def __setitem__(self, i, v):
        self.pts_[i] = v
        return v

    def __hash__(self):
        return tuple([tuple(self.pts_[0]),
                     tuple(self.pts_[1])]).__hash__()
    
    def __eq__(self, o):
        return np.all(self.pts_ == o.pts_)

def gen_pairs(images, skip, offset=0):
    pairs = []
    count = len(images)
    left = offset
    right = offset + skip
    while right < count:
        pairs.append((images[left],images[right]))
        left = right
        right += skip
    return pairs

def load_images(d, pat, skip):
    log.debug("loading images from {0}".format(str(d)))
    dpath = pl.Path(d)
    assert dpath.exists() and dpath.is_dir()
    images = sorted(dpath.glob(pat))
    pairs = gen_pairs(images, skip)
    return images, pairs

def new_correspondence_project(name, image_path, skip, pat="*.png"):
    imgs, pairs = load_images(image_path, pat, skip)
    return {'name': name,
            'image_path': image_path,
            'images': imgs,
            'kps': defaultdict(set),
            'pairs': pairs,
            'skip': skip,
            'correspondences': defaultdict(set),
            'pat': pat}

def get_kps(proj, index=None):
    pair_index = index or proj.get('index',0)
    img_pair = proj['pairs'][pair_index]
    return proj['kps'][img_pair[0]], proj['kps'][img_pair[1]]

def get_correspondences(proj, index=None):
    pair_index = index or proj.get('index',0)
    img_pair = proj['pairs'][pair_index]
    return pair_index, proj['correspondences'][img_pair]

def corr_filename(basedir, left, right):
    """ Return the correspondence filename given the basedir and left/right
    image filenames """
    leftid = left.stem
    rightid = right.stem
    filename = "correspondences_{0}_{1}.csv".format(leftid,rightid)
    return basedir / filename

def write_correspondences(corrs, basedir = pl.Path(".")):
    for k,v in corrs.iteritems():
        left, right = k
        filename = corr_filename(basedir, left, right)
        # correspondences should be Nx4 matrix for N correspondences
        np.savetxt(str(filename), v, fmt='%d', delimeter=',')

def read_correspondences(path):
    C = np.loadtxt(str(path), dtype=np.int32, delimeter=',')
    return C

def save_correspondence_project(proj, filename, save_all_corrs=False):
    savepath = pl.Path(filename)
    if savepath.parent.exists():
        with open(str(savepath),"w") as f:
            pkl.dump(proj, f)
        if save_all_corrs:
            if 'correspondences' in proj:
                corrs = proj['correspondences']
                write_correspondences(corrs)
    else:
        raise IOError("Cannot save to " + filename + ", since the parent doesn't exist")

def load_correspondence_project(filename, load_all_corrs=False):
    loadpath = pl.Path(filename)
    if loadpath.exists():
        with open(str(loadpath),"r") as f:
            proj = pkl.load(f)        
        if load_all_corrs:
            basedir = loadpath.parent
            #corr_paths = sorted(basedir.glob("*.csv"))
            corrs = {}
            for img_pair in proj['pairs']:
                filename = corr_filename(basedir, img_pair[0], img_pair[1])
                C = read_correspondences(filename)
                corrs[img_pair] = C
            proj['correspondences'] = corrs
        return proj
    else:
        raise IOError("File not found: " + str(loadpath))

class ImagePair(QObject):
    item_added = pyqtSignal(tuple)
    item_removed = pyqtSignal(tuple)

    def __init__(self):
        self.corrs_ = []

    def set_images(self, a, b):
        self.image_a = a
        self.image_b = b

    def append(self, corr):
        self.corrs_.append(corr)
        self.item_added.emit(corr)

    def remove(self, idx):
        it = self.corrs_[idx]
        del self.corrs_[idx]
        self.item_removed(it)

class CorrespondenceModel(ImageAnnotationModel):
    def __init__(self, project_name):
        super(CorrespondenceModel,self).__init__()
        self.image_dir_ = None
        self.num_images = 0
        self.images = []
        self.current_pair = 0
        self.skip_frames = 5

    @property
    def image_dir(self):
        return self.image_dir_
    
    @image_dir.setter
    def image_dir(self, d):
        self.image_dir_ = d
        imgs, pairs = load_images(self.image_dir_, "*.png", self.skip_frames)
        self.images = imgs
        self.pairs = pairs

    def export_correspondences(self, filename):
        log.info("export_correspondences unimplemented")

