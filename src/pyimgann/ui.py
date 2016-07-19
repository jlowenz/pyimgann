import traceback
import sys
import os
import pathlib as pl
import logging

from PyQt4.QtCore import Qt, QRect, QLine, QMargins, \
     QDir, pyqtSignal, QRectF, QPointF, QObject
from PyQt4.QtGui import QApplication, QLabel, QWidget, QImage, QPainter, \
     QColor, QPixmap, QGridLayout, QLabel, QGraphicsView, QGraphicsScene, \
     QMainWindow, QPalette, QMenu, QAction, QFileDialog, QScrollArea, \
     QGraphicsItemGroup, QGraphicsLineItem, QGraphicsRectItem, QGraphicsPolygonItem, \
     QGraphicsEllipseItem, QListView, QDockWidget, QPolygonF, QPushButton, QHBoxLayout, \
     QSpinBox, QDialogButtonBox, QLineEdit, QSplitter, QDialog, QFormLayout, QTableView, \
     QGraphicsItem

import numpy as np
from skimage.io import imread
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries, slic, felzenszwalb, quickshift, join_segmentations
from skimage import exposure
import cv2

import qimage2ndarray as qn

log = logging.getLogger('pyimgann.ui')
log.setLevel(logging.DEBUG)

class Annotation(QObject):
    BASE_COLOR = (255,0,0)
    SELECTED_COLOR = (0,255,0)

    changed = pyqtSignal()

    """ Basic annotation that can handle points, lines, and polygons """
    def __init__(self, desc="", color=BASE_COLOR, pts=[]):
        super(Annotation,self).__init__()
        self.desc_ = desc
        self.color_ = color
        self.pts_ = np.array(pts)
        self.radius_ = 4
        self.index = -1
        self.item_ = None

    def __hash__(self):
        return tuple(map(tuple, self.pts_)).__hash__()

    def __eq__(self, o):
        #log.debug("__eq__: {0} {1}".format(self.pts_,o.pts_))
        return (self.pts_ == o.pts_).all()

    def select(self):
        self.color_ = Annotation.SELECTED_COLOR
        self.item_.setPen(self.qcolor)
        self.changed.emit()
        
    def deselect(self):
        self.color_ = Annotation.BASE_COLOR
        self.item_.setPen(self.qcolor)
        self.changed.emit()

    @property
    def is_point(self):
        return len(self.pts_) == 1
    
    @property
    def is_line(self):
        return len(self.pts_) == 2

    @property
    def is_polygon(self):
        return len(self.pts_) > 2

    @property
    def points(self):
        return self.pts_
    
    @property
    def qcolor(self):
        if len(self.color_) == 3:
            return QColor(self.color_[0],
                          self.color_[1],
                          self.color_[2])
        else:
            return QColor(self.color_[0],
                          self.color_[1],
                          self.color_[2],
                          self.color_[3])

    @property
    def item(self):
        if self.item_ is None:
            num_pts = len(self.pts_)
            if num_pts == 1:
                # draw a point          
                item = QGraphicsEllipseItem(self.pts_[0][0]-self.radius_,
                                            self.pts_[0][1]-self.radius_,
                                            2*self.radius_,
                                            2*self.radius_)
                item.setBrush(self.qcolor)
            elif num_pts == 2:
                item = QGraphicsLineItem(self.pts_[0][0], self.pts_[0][1],
                                         self.pts_[1][0], self.pts_[1][1])
            else:
                poly = QPolygonF()
                for p in self.pts_:
                    poly.append(QPointF(p[0],p[1]))
                item = QGraphicsPolygonItem(poly)
            item.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
            item.setPen(self.qcolor)
            item.setEnabled(True)
            item.setActive(True)
            self.item_ = item
        return self.item_
        
class DualImageView(QGraphicsView):
    VERTICAL = 0
    HORIZONTAL = 1
    IMAGE_A = 0
    IMAGE_B = 1

    # no argument signal
    images_changed = pyqtSignal()
    annotations_changed = pyqtSignal()
    annotation_selected = pyqtSignal(int)
    no_selection = pyqtSignal()

    # click/point signals
    image_a_click = pyqtSignal(int,int)
    image_b_click = pyqtSignal(int,int)    

    # keyboard
    key_event = pyqtSignal(int)

    def __init__(self, main_win):
        super(QGraphicsView,self).__init__(main_win)
        self.parent_ = main_win
        self.main_win_ = main_win
        self.setInteractive(True)

        self.setStyleSheet("QGraphicsView { border: none; }")
        
        self.scene_ = QGraphicsScene(0,0,0,0,self.parent_)
        self.image_item_ = self.scene_.addPixmap(QPixmap())
        self.image_item_.setPos(0,0)
        #self.ann_group_ = QGraphicsItemGroup()
        #self.ann_group_.setPos(0,0)
        #self.scene_.addItem(self.ann_group_)
        self.setScene(self.scene_)
        self.scene_.selectionChanged.connect(self.on_selection_changed)
        
        # TODO: handle orientation
        self.orientation_ = DualImageView.VERTICAL
        self.images_ = [None, None]
        self.composite_ = None
        self.annotations_ = []
        self.dim_ = 0
        self.offset_ = np.array([0,0])
        self.cancel_click_ = False
        
        self.images_changed.connect(self.on_images_changed)
        self.annotations_changed.connect(self.on_annotations_changed)

    def on_selection_changed(self):
        log.debug("on_selection_changed")
        selected = self.scene_.selectedItems()
        if len(selected) > 0:
            self.cancel_click_ = True
            selected = self.scene_.selectedItems()[0]
            idx = -1
            for a in self.annotations_:
                idx += 1
                if a.item == selected:
                    log.debug(" emitting selection {0}".format(idx))
                    self.annotation_selected.emit(idx)
        else:
            self.no_selection.emit()

    @property
    def image_b_offset(self):
        return np.array([0,self.dim_],dtype=np.int32)

    def point_in_image(self, p):
        if p[1] < self.dim_:
            return 0
        else:
            return 1

    def point_to_image(self, which, p):
        if which == DualImageView.IMAGE_B:
            return p - self.image_b_offset
        return p

    def image_to_view(self, which, p):
        if which == DualImageView.IMAGE_B:
            return p + self.image_b_offset
        return p

    def on_images_changed(self):
        imga = self.images_[0]
        imgb = self.images_[1]
        width = max(imga.shape[1],imgb.shape[1])
        heighta = imga.shape[0]
        heightb = imgb.shape[0]
        height = heighta + heightb
        self.dim_ = heighta
        self.offset_ = np.array([0,heighta])
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
        #log.debug("on_annotations_changed")
        # self.scene_.removeItem(self.ann_group_)
        # self.ann_group_ = QGraphicsItemGroup()
        # self.ann_group_.setHandlesChildEvents(False)
        # self.ann_group_.setPos(0,0)
        # for a in self.annotations_:
        #     log.debug(" adding item")
        #     self.ann_group_.addToGroup(a.get_item())
        # self.scene_.addItem(self.ann_group_)
        self.repaint()

    def transform_raw_pt(self, ev):
        pt = self.mapToScene(ev.x(), ev.y())
        return np.array([int(pt.x()), int(pt.y())], dtype=np.int32)

    def clear(self):
        self.images_ = [None,None]
        self.composite_ = None
        self.annotations_ = None
        self.images_changed.emit()
        self.annotations_changed.emit()
        
    def set_images(self, img_pair):
        self.images_ = img_pair
        self.images_changed.emit()

    # @property
    # def annotations(self):
    #     return self.annotations_
    
    # @annotations.setter
    # def annotations(self, anns):
    #     self.annotations_ = anns
    #     self.annotations_changed.emit()

    # def set_annotations(self, anns):
    #     self.annotations_ = anns
    #     self.annotations_changed.emit()

    def paintEvent(self, ev):
        painter = QPainter(self.viewport())
        painter.fillRect(0,0,self.viewport().width(),self.viewport().height(), QColor(0,0,0))
        painter.end()
        QGraphicsView.paintEvent(self, ev)

    def annotation(self, idx):
        return self.annotations_[idx]

    def clear_annotations(self):
        for a in self.annotations_:
            self.scene_.removeItem(a.item)
        self.annotations_ = []
        self.annotations_changed.emit()

    def add_annotation(self, ann):
        ann.changed.connect(self.on_annotations_changed)
        self.annotations_.append(ann)
        self.scene_.addItem(ann.item)
        self.annotations_changed.emit()
        return len(self.annotations_) - 1
        
    def remove_last_annotation(self):
        self.scene_.removeItem(self.annotations_[-1].item)
        del self.annotations_[-1]
        self.annotations_changed.emit()

    def remove_annotation(self, idx):
        self.scene_.removeItem(self.annotations_[idx].item)
        del self.annotations_[idx]
        self.annotations_changed.emit()

    def mousePressEvent(self, ev):
        super(DualImageView,self).mousePressEvent(ev)
        if self.cancel_click_:
            return
        log.debug("mouse pressed: " + str(ev))
        self.img_local_pt = self.transform_raw_pt(ev)

    def mouseReleaseEvent(self, ev):
        super(DualImageView,self).mouseReleaseEvent(ev)
        if self.cancel_click_:
            self.cancel_click_ = False
            return
        log.debug("mouse released: " + str(ev))
        rel_pt = self.transform_raw_pt(ev)
        delta = rel_pt - self.img_local_pt
        if abs(delta[0]) < 3 and abs(delta[1] < 3):
            # it was a successful click
            self.mouseClicked(self.img_local_pt)
        else: 
            # recognize this as a rectangle drag
            self.mouseDragged(self.img_local_pt, delta)

    def mouseDragged(self, pt, delta):
        log.debug("mouse dragged: {0}, {1}".format(pt,delta))

    def mouseClicked(self, pt):
        log.debug("mouse clicked: {0}".format(pt))
        if pt[1] < self.dim_:
            self.image_a_click.emit(pt[0],pt[1])
        else:
            self.image_b_click.emit(pt[0],pt[1] - self.dim_)

    # handle the keyboard events here!
    def keyPressEvent(self, ev):
        pass

    def keyReleaseEvent(self, ev):
        k = ev.key()
        self.key_event.emit(k)

class QFileField(QWidget):
    # itemSelected is emitted when a valid file/dir is chosen
    itemSelected = pyqtSignal()
    
    def __init__(self, basepath=os.getcwd(), msg = "Choose", select_dir=False, parent=None):
        super(QFileField,self).__init__(parent)
        self.select_dir_ = select_dir
        self.basepath_ = basepath
        self.filepath_ = None
        self.msg_ = msg

        self.text_ = QLineEdit(self)
        self.browse_ = QPushButton("Select",self)
        self.browse_.clicked.connect(self.on_browse)

        layout = QHBoxLayout(self)
        layout.addWidget(self.text_)
        layout.addWidget(self.browse_)
        layout.setMargin(0)
        self.setLayout(layout)

    def on_browse(self):
        # open a file selection dialog
        if self.select_dir_:
            path = QFileDialog.getExistingDirectory(self, self.msg_, self.basepath_)
        else:
            path = QFileDialog.getOpenFileName(self, self.msg_, self.basepath_)
        if path is not None:
            self.filepath_ = str(path)
            self.text_.setText(self.filepath_)
            self.itemSelected.emit()
    
    @property
    def path(self):
        return self.filepath_

    @path.setter
    def path(self, p):
        # todo check validity
        self.filepath_ = p

class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super(NewProjectDialog,self).__init__(parent)
        self.setSizeGripEnabled(True)
        self.create_layout()
        
    def create_layout(self):
        # select directory containing images
        # configure the skip-frames
        # cancel or create project
        self.project_name_ = QLineEdit()
        self.filefield_ = QFileField(msg="Select Image Directory...", select_dir=True)
        self.skip_ = QSpinBox()
        self.skip_.setRange(0,50)
        self.skip_.setValue(5)
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignVCenter)
        layout.addRow("Project Name:", self.project_name_)
        layout.addRow("Image Directory:", self.filefield_)
        layout.addRow("Skip images:", self.skip_)
        
        dbb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dbb.accepted.connect(self.on_accept)
        dbb.rejected.connect(self.on_reject)
        layout.addRow(dbb)
        self.setLayout(layout)
    
    @property
    def name(self):
        return self.project_name_.text()
        
    @property
    def path(self):
        return self.filefield_.path
    
    @property
    def skip(self):
        return self.skip_.value()

    def on_accept(self):
        # todo: check file validity
        self.accept()
        
    def on_reject(self):
        self.reject()    
        
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.dual_img_ = DualImageView(self)
        self.corr_list_ = QTableView(self)
        self.pair_list_ = QListView(self)
        self.next_button_ = QPushButton("&Next",self)
        self.prev_button_ = QPushButton("&Previous",self)
        self.status_msg_ = QLabel("Status...",self)
        
        self.corr_list_.verticalHeader().setVisible(False)
        self.corr_list_.setSelectionBehavior(QTableView.SelectRows)

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
        
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.corr_list_)
        splitter.addWidget(self.pair_list_)
        self.dock(splitter, Qt.LeftDockWidgetArea, title="Correspondences")

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
