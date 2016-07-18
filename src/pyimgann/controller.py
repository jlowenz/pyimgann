import os
import pyimgann.ui as ui
import logging
import traceback as tb
from functools import partial
import pathlib as pl
import cPickle as pkl
from skimage.io import imread
from PyQt4.QtCore import pyqtSignal, QObject, QRect
from PyQt4.QtGui import QAction, QStandardItemModel, QStandardItem, QDialog, \
     QItemSelectionModel, QUndoCommand, QUndoStack, QFileDialog, QHeaderView

import numpy as np
from transitions import Machine
import pyimgann.model as mdl

log = logging.getLogger("pyimgann.controller")
log.setLevel(logging.DEBUG)

class AnnotationController(QObject):
    def __init__(self, ui):
        super(AnnotationController,self).__init__()
        self.ui_ = ui

def to_model(items, model, formatter):
    for i in items:
        model.appendRow(formatter(i))

def show_images(img_pair, ui):
    imga = imread(str(img_pair[0]))
    imgb = imread(str(img_pair[1]))
    ui.dual_img.set_images((imga,imgb))

def corr_formatter(corr):
    left = QStandardItem("{0}".format(corr[0,:]))
    right = QStandardItem("{0}".format(corr[1,:]))
    return [left,right]

def img_pair_formatter(proj, ipair):
    left = ipair[0].stem
    right = ipair[1].stem
    return QStandardItem("{0}, {1}".format(left,right))

def draw_annotation(view, pts=np.array([]), color=(255,0,0), desc=""):
    ann = ui.Annotation(pts=np.array([pts[0],
                                      pts[1]+view.image_b_offset]), 
                        color=color, desc=desc)
    ann.index = view.add_annotation(ann)
    # anns = view.annotations
    # anns.append(ann)
    # view.annotations = anns

def draw_keypoint(view, pt, color=(255,0,0,128), desc=""):
    ann = ui.Annotation(pts=[pt], color=color, desc=desc)
    ann.index = view.add_annotation(ann)

def show_keypoints(apts, bpts, ui):
    log.debug("show keypoints")
    for a in apts:
        draw_keypoint(ui.dual_img, a)
    for b in bpts:
        draw_keypoint(ui.dual_img, b + ui.dual_img.image_b_offset)

def load_annotations(corrs, ui):
    log.debug("load_annotations")
    to_model(corrs, ui.corr_model, corr_formatter)
    w = ui.corr_view.width()
    ui.corr_view.horizontalHeader().setResizeMode(QHeaderView.Stretch)    
    # cnt = ui.corr_view.horizontalHeader().count()
    # for i in range(cnt):
    #     ui.corr_view.setColumnWidth(i, w/2)
    for c in corrs:
        draw_annotation(ui.dual_img, pts=c)

def load_frame(proj, ui, idx):
    tb.print_stack()
    log.debug("load frame {0}".format(idx))
    ui.clear()
    # update the index    
    count = len(proj['pairs'])
    img_pair = proj['pairs'][idx]
    # load the image and keypoints
    show_images(img_pair, ui)
    akps, bkps = mdl.get_kps(proj, idx)
    show_keypoints(akps, bkps, ui)
    # update the pair correspondences
    _, corrs = mdl.get_correspondences(proj, idx)
    load_annotations(corrs, ui)
    ui.status_field.setText("Loaded frame {0}".format(idx))

def load_project(proj, ui):
    ui.clear(clear_pairs=True)
    ui.status_field.setText("Loading project: " + proj['name'])
    # load the list data
    to_model(proj['pairs'], ui.pair_model, partial(img_pair_formatter, proj))
    # to_model(proj['correspondences'], ui.corr_model)
    # load the current image pair
    pair_index = proj.get('index',0)
    proj['index'] = pair_index
    #load_frame(proj, ui, pair_index)
    ui.select_pair(pair_index)
    ui.status_field.setText("{0} project ready".format(proj['name']))


class AddCorrespondenceCmd(QUndoCommand):
    def __init__(self, proj, view, pta, ptb, desc="", color=""):
        super(AddCorrespondenceCmd,self).__init__()
        self.proj = proj
        self.view = view
        self.pts = np.array([pta,ptb])
        self.desc = desc
        self.color = color
        self.setText("add correspondence")
        
    def redo(self):
        log.debug("Add correspondence")
        pair_index, corr_set = mdl.get_correspondences(self.proj)
        akps, bkps = mdl.get_kps(self.proj, pair_index)
        self.corr_index = len(corr_set)
        corr_set.append(self.pts)
        akps.add(tuple(self.pts[0]))
        bkps.add(tuple(self.pts[1]))
        self.ann = ui.Annotation(pts=np.array([self.pts[0],
                                          self.pts[1] + self.view.image_b_offset]))
        self.ann.index = self.view.add_annotation(self.ann)
        
    def undo(self):
        pair_index, corr_set = mdl.get_correspondences(self.proj)
        akps, bkps = mdl.get_kps(self.proj, pair_index)
        apt,bpt = corr_set[self.corr_index]
        del corr_set[self.corr_index]
        akps.discard(tuple(apt))
        bkps.discard(tuple(bpt))
        self.view.remove_annotation(self.ann.index)

class CorrespondenceController(AnnotationController):
    project_changed = pyqtSignal()
        
    states = ['no_project', 'new_project', 'open_project', 'clean_project', \
              'dirty_project', 'point_a', 'point_b', 'exiting']

    def __init__(self, ui):
        super(CorrespondenceController,self).__init__(ui)

        self.machine = Machine(model=self, states=CorrespondenceController.states,
                               initial='no_project')
        self.machine.on_enter_dirty_project('save_dirty')

        # no project
        self.machine.add_transition(trigger='on_new_project', 
                                    source=['no_project','clean_project'],
                                    dest='new_project',
                                    conditions='do_new_project')
        self.machine.add_transition('on_open_project',
                                    ['no_project','clean_project'],
                                    'clean_project',
                                    conditions='do_open_project')

        # clean project
        self.machine.add_transition('on_close_project',
                                    'clean_project',
                                    'no_project',
                                    conditions='do_close_project')

        self.machine.add_transition('on_save_project',
                                    ['new_project','dirty_project'],
                                    'clean_project',
                                    conditions='do_save_project')

        # point_a
        self.machine.add_transition('on_image_a_point',
                                    ['clean_project','new_project'],
                                    'point_a')
        self.machine.add_transition('on_image_a_point',
                                    'point_a',
                                    'point_a')
        self.machine.add_transition('on_image_b_point',
                                    'point_a',
                                    'dirty_project',
                                    conditions='add_correspondence')

        # point_b
        self.machine.add_transition('on_image_b_point',
                                    ['clean_project','new_project'],
                                    'point_b')
        self.machine.add_transition('on_image_b_point',
                                    'point_b',
                                    'point_b')
        self.machine.add_transition('on_image_a_point',
                                    'point_b',
                                    'dirty_project',
                                    conditions='add_correspondence')

        self.machine.add_transition('on_exit',
                                    '*', 'exiting',
                                    before='check_save',
                                    after='do_exit',
                                    conditions='safe_to_exit')

        self.project_changed.connect(self.on_project_changed)
        self.current_project = None
        self.current_filename = None
        self.dual_img = ui.select('dual_img')
        self.dual_img.image_a_click.connect(self.image_a_clicked)
        self.dual_img.image_b_click.connect(self.image_b_clicked)
        self.corr_view = ui.select('corr_list')
        self.corr_model = QStandardItemModel(self.corr_view)
        self.corr_view.setModel(self.corr_model)
        self.pair_view = ui.select('pair_list')
        self.pair_model = QStandardItemModel(self.pair_view)
        self.pair_view.setModel(self.pair_model)
        self.status_field = ui.select('status_msg')
        ui.select('next_button').clicked.connect(self.on_next_pair)
        ui.select('prev_button').clicked.connect(self.on_prev_pair)

        self.pair_view.selectionModel().selectionChanged.connect(self.pair_selected)
        self.dual_img.annotation_selected.connect(self.annotation_selected)
        self.dual_img.no_selection.connect(self.clear_selection)

        self.a_point = None
        self.b_point = None
        self.selection = None

        self.undo_stack = QUndoStack()

        self.file_menu = self.ui_.select('file')
        self.edit_menu = self.ui_.select('edit')
        self.options_menu = self.ui_.select('options')
        self.create_actions()

    def clear(self, clear_pairs=False):
        self.undo_stack.clear()
        self.a_point = None
        self.b_point = None
        self.selection = None
        self.dual_img.clear_annotations()
        self.corr_model.clear()
        if clear_pairs:
            self.pair_model.clear()

    def check_save(self, checked):
        if self.state == 'dirty_project' or self.state == 'new_project':
            return self.save_dirty()
        return True

    def safe_to_exit(self, checked):
        return self.check_save(checked)

    def add_correspondence(self):
        self.undo_stack.push(AddCorrespondenceCmd(self.current_project, self.dual_img,
                                                  self.a_point, self.b_point))
        return True

    def clear_selection(self):
        self.selection[1].deselect()
        self.selection = None

    def annotation_selected(self, idx):
        log.debug("annotation_selected")
        if self.selection:
            # if another item was selected
            self.selection[1].deselect()
        ann = self.dual_img.annotation(idx)
        self.selection = (idx, ann)
        ann.select()

    def pair_selected(self, item_selections):
        if len(item_selections.indexes()) == 0:
            return
        else:
            pair_idx = item_selections.indexes()[0].row()
            self.current_project['index'] = pair_idx
            load_frame(self.current_project, self, pair_idx)

    def select_pair(self, idx):
        # set the selection and trigger the load
        mdl_idx = self.pair_model.index(idx,0)
        #self.pair_view.setSelection(QRect(0,idx,1,1),QItemSelectionModel.Select)
        self.pair_view.setCurrentIndex(mdl_idx)

    def on_next_pair(self):
        log.debug("next image pair")
        # update the index
        count = len(self.current_project['pairs'])
        idx = self.current_project['index'] + 1
        self.current_project['index'] = idx = min(idx, count - 1)
        #load_frame(self.current_project, self, idx)
        self.select_pair(idx)
        
    def on_prev_pair(self):
        log.debug("previous image pair")
        idx = max(0, self.current_project['index'] - 1)
        self.current_project['index'] = idx
        #load_frame(self.current_project, self, idx)
        self.select_pair(idx)

    def image_a_clicked(self, x, y):
        log.debug("image A clicked: {0}".format((x,y)))
        self.a_point = np.array([x,y])
        self.on_image_a_point()

    def image_b_clicked(self, x, y):
        log.debug("image B clicked: {0}".format((x,y)))
        self.b_point = np.array([x,y])
        self.on_image_b_point()

    def on_project_changed(self):
        log.debug("project changed")
        self.do_save_project_.setEnabled(True)

    # the "extra" argument is from Qt triggered() signal
    def do_new_project(self, checked):
        log.debug("new project: {0}".format(checked))
        npd = ui.NewProjectDialog(self.ui_)
        if npd.exec_() == QDialog.Accepted:
            name = npd.name
            path = pl.Path(npd.path)
            skip_images = npd.skip
            self.current_project = mdl.new_correspondence_project(name, path, skip_images)
            load_project(self.current_project, self)
            return True
        return False

    def do_open_project(self, checked):
        log.debug("open project")
        fn = QFileDialog.getOpenFileName(self.ui_, "Open file", os.getcwd(), "*.pya")
        self.current_filename = str(fn)
        if self.current_filename:
            self.current_project = mdl.load_correspondence_project(self.current_filename)
            load_project(self.current_project, self)
            return True
        return False
    
    def do_close_project(self, checked):
        log.debug("close project")
        self.current_project = None
        self.corr_model.clear()
        self.pair_model.clear()
        return True

    def do_save_project(self, checked):
        log.debug("save project")
        return self.save()
    
    def save_dirty(self):
        log.debug("saving dirty project")
        if self.save():
            self.to_clean_project()
            return True
        log.error("Failed to save project")
        return False

    def save(self):
        log.debug("current filename: {0}".format(self.current_filename))
        if self.current_filename is None:
            imgpath = self.current_project['image_path']
            fn = QFileDialog.getSaveFileName(self.ui_, "Save File As", str(imgpath), "*.pya")
            self.current_filename = str(fn)
        if self.current_filename:
            mdl.save_correspondence_project(self.current_project, self.current_filename)
            self.do_save_project_.setEnabled(False)
            return True
        return False            

    def do_exit(self, checked):
        log.debug("exit")
        # check whether the project should be saved
        # save it
        # shutdown the application
        self.ui_.close()

    def create_actions(self):
        self.do_new_project_ = QAction("&New Project...", self.ui_)
        self.do_new_project_.setShortcut("Ctrl+N")        
        self.do_new_project_.triggered.connect(self.on_new_project)
        self.file_menu.addAction(self.do_new_project_)

        self.do_open_project_ = QAction("&Open Project...", self.ui_)
        self.do_open_project_.setShortcut("Ctrl+O")
        self.do_open_project_.triggered.connect(self.__dict__['on_open_project'])
        self.file_menu.addAction(self.do_open_project_)

        self.do_close_project_ = QAction("&Close Project", self.ui_)
        self.do_close_project_.setShortcut("Ctrl+C")
        self.do_close_project_.triggered.connect(self.__dict__['on_close_project'])
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.do_close_project_)

        self.do_save_project_ = QAction("&Save", self.ui_)
        self.do_save_project_.setShortcut("Ctrl+S")
        self.do_save_project_.triggered.connect(self.__dict__['on_save_project'])
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.do_save_project_)

        self.do_exit_ = QAction("E&xit", self.ui_)
        self.do_exit_.setShortcut("Ctrl+Q")
        self.do_exit_.triggered.connect(self.__dict__['on_exit'])
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.do_exit_)
        
        self.edit_menu.addAction(self.undo_stack.createUndoAction(self.ui_))
        self.edit_menu.addAction(self.undo_stack.createRedoAction(self.ui_))
