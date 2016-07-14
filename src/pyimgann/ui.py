import traceback
import sys
import pathlib as pl

from PyQt4.QtCore import Qt, QRect, QLine, QMargins, \
     QDir, pyqtSignal, QRectF, QPointF
from PyQt4.QtGui import QApplication, QLabel, QWidget, QImage, QPainter, \
     QColor, QPixmap, QGridLayout, QLabel, QGraphicsView, QGraphicsScene, \
     QMainWindow, QPalette, QMenu, QAction, QFileDialog, QScrollArea, \
     QGraphicsItemGroup, QGraphicsLineItem, QGraphicsRectItem, QGraphicsPolygonItem, \
     QGraphicsEllipseItem, QListView, QDockWidget, QPolygonF, QPushButton, QHBoxLayout

import numpy as np
from skimage.io import imread
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries, slic, felzenszwalb, quickshift, join_segmentations
from skimage import exposure
import cv2

import qimage2ndarray as qn

class Annotation(object):
    """ Basic annotation that can handle points, lines, and polygons """
    def __init__(self, desc="", color=(255,0,0), pts=[]):
        self.desc_ = desc
        self.color_ = color
        self.pts_ = pts
        self.item_ = None

    @property
    def qcolor(self):
        return QColor(self.color_[0],
                      self.color_[1],
                      self.color_[2])

    def get_item(self):
        if self.item_ is None:
            num_pts = len(self.pts_)
            if num_pts == 1:
                # draw a point                
                self.item_ = QGraphicsEllipseItem(pts[0][0]-1,
                                                  pts[0][1]-1,
                                                  3,3,
                                                  self.qcolor)

            elif num_pts == 2:
                self.item_ = QGraphicsLineItem(pts[0][0], pts[0][1],
                                               pts[1][0], pts[1][1],
                                               self.qcolor)
            else:
                poly = QPolygonF()
                for p in self.pts_:
                    poly.append(QPointF(p[0],p[1]))
                self.item_ = QGraphicsPolygonItem(poly, self.qcolor)
        return self.item_
        
class DualImageView(QGraphicsView):
    VERTICAL = 0
    HORIZONTAL = 1

    # no argument signal
    images_changed = pyqtSignal()
    annotations_changed = pyqtSignal()
    
    image_a_click = pyqtSignal(int,int)
    image_b_click = pyqtSignal(int,int)    

    def __init__(self, main_win):
        super(QGraphicsView,self).__init__(main_win)
        self.parent_ = main_win
        self.main_win_ = main_win
        
        self.setStyleSheet("QGraphicsView { border: none; }")
        
        self.scene_ = QGraphicsScene(0,0,0,0,self.parent_)
        self.image_item_ = self.scene_.addPixmap(QPixmap())
        self.image_item_.setPos(0,0)
        self.ann_group_ = QGraphicsItemGroup()
        self.ann_group_.setPos(0,0)
        self.scene_.addItem(self.ann_group_)
        self.setScene(self.scene_)
        
        # TODO: handle orientation
        self.orientation_ = DualImageView.VERTICAL
        self.images_ = [None, None]
        self.composite_ = None
        self.annotations_ = []
        self.dim_ = 0
        
        self.images_changed.connect(self.on_images_changed)

    def on_images_changed(self):
        imga = self.images_[0]
        imgb = self.images_[1]
        width = max(imga.shape[1],imgb.shape[1])
        heighta = imga.shape[0]
        heightb = imgb.shape[0]
        height = heighta + heightb
        self.dim_ = heighta
        # this assumes rgb images :-(
        comp = np.empty((height,width,imga.shape[2]),dtype=imga.dtype)
        comp[0:heighta,:imga.shape[1],:] = imga
        comp[heighta:(heighta+heightb),:imgb.shape[1],:] = imgb
        self.composite_ = comp
        qimg = qn.array2qimage(self.composite_)
        pix = QPixmap.fromImage(qimg)
        self.image_item_.setPixmap(pix)
        self.scene_.setSceneRect(0,0, width, height)
        self.repaint()
        
    def on_annotations_changed(self):
        if self.composite_:
            self.scene_.removeItem(self.ann_group_)
            self.ann_group_ = QGraphicsItemGroup()
            self.ann_group_.setPos(0,0)
            for a in self.annotations_:
                self.ann_group_.addToGroup(a.get_item())
            self.scene_.addItem(self.ann_group_)

    def transform_raw_pt(self, ev):
        pt = self.mapToScene(ev.x(), ev.y())
        return (int(pt.x()), int(pt.y()))

    def clear(self):
        self.images_ = [None,None]
        self.composite_ = None
        self.annotations_ = None
        self.images_changed.emit()
        self.annotations_changed.emit()
        
    def set_images(self, img_pair):
        self.images_ = img_pair
        self.images_changed.emit()

    def set_annotations(self, anns):
        self.annotations_ = anns
        self.annotations_changed.emit()

    def paintEvent(self, ev):
        painter = QPainter(self.viewport())
        painter.fillRect(0,0,self.viewport().width(),self.viewport().height(), QColor(0,0,0))
        painter.end()
        QGraphicsView.paintEvent(self, ev)

    def add_annotation(self, ann):
        self.annotations_.append(ann)
        self.annotations_changed.emit()
        
    def remove_last_annotation(self):
        del self.annotations_[-1]
        self.annotations_changed.emit()

    def mouseClickEvent(self, ev):
        img_local = self.transform_raw_pt(ev)
        if img_local[1] < self.dim_:
            self.image_a_click.emit(img_local[0],img_local[1])
        else:
            self.image_b_click.emit(img_local[0],img_local[1])

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.dual_img_ = DualImageView(self)
        self.corr_list_ = QListView(self)
        self.next_button_ = QPushButton("&Next",self)
        self.prev_button_ = QPushButton("&Previous",self)
        self.status_msg_ = QLabel("Status...",self)
        
        self.create_layout()
        self.create_menu()

        self.setWindowTitle("Image Annotation")
            
    def select(self, name):
        try:
            return self.__dict__[name + "_"]
        except:
            None

    def dock(self, widget, where, title=None):
        if title:
            d = QDockWidget(title, self)
        else:
            d = QDockWidget(self)
        d.setWidget(widget)
        self.addDockWidget(where, d)
        return d

    def create_layout(self):
        self.setCentralWidget(self.dual_img_)        
        self.dock(self.corr_list_, Qt.LeftDockWidgetArea, title="Correspondences")
        wpanel = QWidget()
        hpanel = QHBoxLayout(wpanel)
        hpanel.addWidget(self.prev_button_)
        hpanel.addWidget(self.next_button_)
        hpanel.addWidget(self.status_msg_)
        hpanel.addStretch()
        wpanel.setLayout(hpanel)
        bot = self.dock(wpanel, Qt.BottomDockWidgetArea)
        bot.setFeatures(bot.features() & QDockWidget.NoDockWidgetFeatures)

    def create_menu(self):
        self.file_ = QMenu("&File", self)
        self.edit_ = QMenu("&Edit", self)
        self.options_ = QMenu("&Options", self)
        
        self.menuBar().addMenu(self.file_)
        self.menuBar().addMenu(self.edit_)
        self.menuBar().addMenu(self.options_)
