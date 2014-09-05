#!/usr/bin/python2.4
#  
#  For more information, please see: http://software.sci.utah.edu
#  
#  The MIT License
#  
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
#  
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#  
#    File   : SurfsToTets.py
#    Author : Martin Cole
#    Date   : Mon May 15 08:42:14 2006

import pygtk
pygtk.require('2.0')

import gtk, gtk.glade, gobject
import gtk.gtkgl
import sys, os
import time
import thread
import threading
import re

from OpenGL.GL import *
from OpenGL.GLU import *

import segnrrd
import glob


# import the scirun python api.
import pysci

 
path_exp = re.compile("(.+/)(.+)")

def do_tetgen(stobj, in1, in2, out) :
        stobj.status = 0;
        if pysci.tetgen_2surf(in1, in2, out) :
            stobj.pbar_.set_text("Completed")
            print "SUCCESS!"
            stobj.status = 1
        else :
            print "Failed :("
            stobj.status = 2
            stobj.pbar_.set_text("Failed...")
        # progress bar is done.
        stobj.pbar_state = 2

global vthread
vthread = None

## class Viewer(threading.Thread) :
##     def __init__(self, glconfig):
##         threading.Thread.__init__(self, None, None, "Viewer-1")
##         self.glconfig = glconfig
##         self.drawing_area = None
        
##     def run(self) :
##         try :
##             print "running"
        
##             self.drawing_area = SimpleDrawingArea(self.glconfig)
##             self.drawing_area.set_size_request(1366, 768)
##             self.drawing_area.show()

##             while (self.drawing_area.sci_context == None) :
##                 print "not set yet"
##                 time.sleep(.1)
            
##             pysci.run_viewer_thread(self.drawing_area.sci_context)
##         except Exception, ex:
##             print "caught exception in viewer thread"

##         print "Viewer done - python"

class EM(threading.Thread) :
    def __init__(self):
        threading.Thread.__init__(self, None, None, "EventManager-1")
    def run(self) :
        try :
            pysci.run_event_manager_thread()
        except Exception, ex:
            print "caught exception in EventManager thread"

        print "EventManager done - python"

# Update the value of the progress bar so that we get
# some movement

def progress_timeout(pbobj):
    #gtk.threads_enter()
    if pbobj.pbar_state  == 1 :
        pbobj.pbar_.pulse()
    elif pbobj.pbar_state  == 2 :
        # Calculate the value of the progress bar using the
        # value range set in the adjustment object
        pbobj.pbar_.set_fraction(1.0)
        pbobj.pbar_state = 0
    # As this is a timeout function, return TRUE so that it
    # continues to get called
    #gtk.threads_leave()
    return True

class SimpleDrawingArea(gtk.DrawingArea, gtk.gtkgl.Widget):
  """OpenGL drawing area."""

  def __init__(self, glconfig):
    print "called __init__"
    gtk.DrawingArea.__init__(self)
    self.sci_context = None
    # Set OpenGL-capability to the drawing area
    self.set_gl_capability(glconfig)

    # Connect the relevant signals.
    self.connect_after('realize',   self._on_realize)
        
  def _on_realize(self, *args):
    print "called _on_realize"
    # Obtain a reference to the OpenGL drawable
    # and rendering context.
    gldrawable = self.get_gl_drawable()
    glcontext = self.get_gl_context()
        
        
    self.sci_context = pysci.CallbackOpenGLContext()
    self.sci_context.set_pymake_current_func(self.make_current)
    self.sci_context.set_pyswap_func(self.swap)
    self.sci_context.set_pywidth_func(self.width)
    self.sci_context.set_pyheight_func(self.height)
    self.sci_context.set_pyrelease_func(self.release)


  def make_current(self) :
    #print "making current"
    t = threading.currentThread()
    #print "active: %d" % threading.activeCount()
    #print t
    gldrawable = self.get_gl_drawable()
    glcontext = self.get_gl_context()
    #print " - - - "
    # OpenGL begin.
    if not gldrawable.gl_begin(glcontext):
      print "gl_begin failed"
      return
        
    if not gldrawable.make_current(glcontext) :
      print "make current failed"
      return 0;

##         data = glGetIntegerv(GL_MAX_LIGHTS);
##         gldrawable.gl_end()
##         print "GL_MAX_LIGHTS = %d" % data
        #print "make current success"
    return 1

  def swap(self) :
    gldrawable = self.get_gl_drawable()

    if gldrawable.is_double_buffered():
      gldrawable.swap_buffers()
    else:
      print "Error: not double buffered drawable, cannot swap"

  def width(self) :
    gldrawable = self.get_gl_drawable()
    t = gldrawable.get_size()
    return t[0]

  def height(self) :
    gldrawable = self.get_gl_drawable()
    t = gldrawable.get_size()
    return t[1]

  def release(self) :
    gldrawable = self.get_gl_drawable()
    gldrawable.gl_end()


def get_pysci_modifier_mask(event, mask) :
  if event.state & gtk.gdk.SHIFT_MASK :
    mask  |= pysci.EventModifiers.SHIFT_E
  if event.state & gtk.gdk.LOCK_MASK :
    mask  |= pysci.EventModifiers.CAPS_LOCK_E
  if event.state & gtk.gdk.CONTROL_MASK :
    mask  |= pysci.EventModifiers.CONTROL_E
  if event.state & gtk.gdk.MOD1_MASK :
    mask  |= pysci.EventModifiers.ALT_E
  if event.state & gtk.gdk.MOD2_MASK :
    mask  |= pysci.EventModifiers.M1_E
  if event.state & gtk.gdk.MOD3_MASK :
    mask  |= pysci.EventModifiers.M2_E
  if event.state & gtk.gdk.MOD4_MASK :
    mask  |= pysci.EventModifiers.M3_E
  if event.state & gtk.gdk.MOD5_MASK :
    mask  |= pysci.EventModifiers.M4_E
  return mask

def get_pysci_button_num(event) :
  if event.state & gtk.gdk.BUTTON1_MASK :
    return 1
  if event.state & gtk.gdk.BUTTON2_MASK :
    return 2
  if event.state & gtk.gdk.BUTTON3_MASK :
    return 3
  if event.state & gtk.gdk.BUTTON4_MASK :
    return 4
  if event.state & gtk.gdk.BUTTON5_MASK :
    return 5
    
def get_pysci_pointer_modifier_mask(which, mask) :
  if which == 1 :
    mask  |= pysci.PointerEvent.BUTTON_1_E
  if which == 2 :
    mask  |= pysci.PointerEvent.BUTTON_2_E
  if which == 3 :
    mask  |= pysci.PointerEvent.BUTTON_3_E
  if which == 4 :
    mask  |= pysci.PointerEvent.BUTTON_4_E
  if which == 5 :
    mask  |= pysci.PointerEvent.BUTTON_5_E
  return mask
        

class SurfsToTets:
  def __init__(self):
    gtk.glade.set_custom_handler(self.get_custom_handler)
    self.xml_ = gtk.glade.XML('tetmesh.glade')
    self.xml_.signal_autoconnect(self)
    self.main_dialog_ = self.xml_.get_widget("main_dialog")
    self.main_dialog_.show()
    self.file_ = None
    self.shortcuts_added_ = 0
    self.pbar_state = 0
    self.pbar_ = self.xml_.get_widget("progress")
    self.sbar_ = self.xml_.get_widget("status_bar")
    print self.sbar_
    print "^^^^^"
    
    self.sbar_ctx_ = self.sbar_.get_context_id("Main Notification Area")
    print self.sbar_ctx_

    
    self.drawing_area = None
    self.list_view_ = self.xml_.get_widget("model_treeview")
    self.mat_list_view_ = self.xml_.get_widget("material_treeview")
    self.init_model_treeview()
    self.init_material_treeview()
    self.open_file_handler_ = None

    # Add a timer callback to update the value of the progress bar
    self.timer = gobject.timeout_add (100, progress_timeout, self)


  def get_custom_handler(self, glade, function_name, widget_name,
			 str1, str2, int1, int2):
    """
    Generic handler for creating custom widgets, used to
    enable custom widgets.

    The custom widgets have a creation function specified in design time.
    Those creation functions are always called with str1,str2,int1,int2 as
    arguments, that are values specified in design time.

    This handler assumes that we have a method for every custom widget
    creation function specified in glade.

    If a custom widget has create_foo as creation function, then the
    method named create_foo is called with str1,str2,int1,int2 as arguments.
    """

    print function_name
    print widget_name
    print str1
    print str2
    print int1
    print int2
    print glade
    handler = getattr(self, function_name)
    return handler(str1, str2, int1, int2)

  def init_model_treeview(self) :
	  
    ls = gtk.ListStore(gobject.TYPE_STRING, gobject.TYPE_STRING,
		       gobject.TYPE_STRING)

    self.list_view_.set_model(ls)
    self.list_view_.set_headers_visible(True)

    ifact = gtk.IconFactory()
    ifact.lookup("visible")

    renderer = gtk.CellRendererPixbuf()
    #renderer.connect('edited', self.on_model_treeview_cell_edited, 0)
    #renderer.set_property('editable', True)
    column = gtk.TreeViewColumn("Visible", renderer, stock_id=0)
    self.list_view_.append_column(column)

    renderer = gtk.CellRendererText()
    column = gtk.TreeViewColumn("Name", renderer, text=1)
    self.list_view_.append_column(column)

    renderer = gtk.CellRendererText()
    column = gtk.TreeViewColumn("FID", renderer, text=2)
    self.list_view_.append_column(column)

  def init_material_treeview(self) :

    ls = gtk.ListStore(int, str, float)
     
    self.mat_list_view_.set_model(ls)
    self.mat_list_view_.set_headers_visible(True)

    renderer = gtk.CellRendererText()
    renderer.connect('edited', self.on_material_treeview_cell_edited, 0)
    renderer.set_property('editable', True)#gtk.CELL_RENDERER_MODE_EDITABLE)
    column = gtk.TreeViewColumn("Material Index", renderer, text=0)
    self.mat_list_view_.append_column(column)
        
    renderer = gtk.CellRendererText()
    renderer.connect('edited', self.on_material_treeview_cell_edited, 1)
    renderer.set_property('editable', True)
    column = gtk.TreeViewColumn("Material Name", renderer, text=1)
    self.mat_list_view_.append_column(column)

    renderer = gtk.CellRendererText()
    renderer.connect('edited', self.on_material_treeview_cell_edited, 2)
    renderer.set_property('editable', True)
    column = gtk.TreeViewColumn("Isovalue", renderer, text=2)
    self.mat_list_view_.append_column(column)


  # this selects a model from the visible list, for picking purposes.
  def on_model_treeview_button_release_event(self, a, event) :
    
    sel = self.list_view_.get_selection()
    l = sel.get_selected()[0]
    it = sel.get_selected()[1]

    if it == None : return
 
##     print l.get_path(it)
##     print l.get_value(it, 1)
##     print l.get_value(it, 2)


    if event.button == 3 :
      ls = self.list_view_.get_model()
      # right click means toggle visibility
      tog = gtk.STOCK_YES
      if l.get_value(it, 0) == "gtk-yes" :
	tog = gtk.STOCK_NO
      ls.set(it, 0, tog)
      fid = l.get_value(it, 2)
      pysci.toggle_field_visibility(int(fid))
      


    # create an event to notify tools that the selection target is
    # the selected field id.
    fid = -1
    fid = l.get_value(it, 2)
    self.show_status_msg("%s is selected." % l.get_value(it, 1))
    pysci.selection_target_changed(int(fid))


  def get_file(self, pattern) :

    chooser = self.xml_.get_widget("chooser")
    filt = gtk.FileFilter()
    filt.add_pattern(pattern)
    chooser.set_filter(filt)
    if not self.shortcuts_added_ :
	self.shortcuts_added_ = 1
	cdir = os.getcwd()
	chooser.add_shortcut_folder(cdir)
	if os.environ.has_key("SCIRUN_DATA") :
	    path = os.environ["SCIRUN_DATA"]
	    chooser.add_shortcut_folder(path)

    chooser.show()

  def get_dir(self) :
    chooser = self.xml_.get_widget("dir_chooser")
    chooser.show()

  def on_open_fld_clicked(self, button) :
    self.open_file_handler_ = self.handle_load_srfld
    self.get_file("*.fld")


  def on_chooser_cancel_clicked(self, button) :
    chooser = self.xml_.get_widget("chooser")
    chooser.hide()

  def on_dir_chooser_cancel_clicked(self, button) :
    chooser = self.xml_.get_widget("dir_chooser")
    chooser.hide()

  def handle_load_srfld(self) :
      # show the field in the viewer
      fld_id = pysci.load_field(self.file_)
      if fld_id == 0 :
	print "Error loading file: %s" % self.file_
	return
      pysci.show_field(fld_id)
      fstr = self.file_
      mo = path_exp.match(fstr)
      if mo != None :
        fstr = mo.group(2)

      ls = self.list_view_.get_model()
      ls.insert(0, (gtk.STOCK_YES, fstr, fld_id))

      
  def on_chooser_open_clicked(self, button) :
    chooser = self.xml_.get_widget("chooser")
    if chooser.get_filename() != None :
      self.file_ = chooser.get_filename()
      self.open_file_handler_()
      chooser.hide()

  def on_dir_chooser_open_clicked(self, button) :
    chooser = self.xml_.get_widget("dir_chooser")
    if chooser.get_filename() != None :
      self.file_ = chooser.get_filename()
      self.open_file_handler_()
      chooser.hide()


  def on_quit_activate(self, button) :
    global vthread
    pysci.terminate();
    time.sleep(.5)
    sys.exit(0)


  def create_ogl_window(self, str1, str2, int1, int2) :
    print "creation function"
    
    # Query the OpenGL extension version.
    print "OpenGL extension version - %d.%d\n" % gtk.gdkgl.query_version()

    # Configure OpenGL framebuffer.
    # Try to get a double-buffered framebuffer configuration,
    # if not successful then try to get a single-buffered one.
    display_mode = (gtk.gdkgl.MODE_RGBA    |
		    gtk.gdkgl.MODE_DEPTH  |
		    gtk.gdkgl.MODE_DOUBLE)
    try:
      glconfig = gtk.gdkgl.Config(mode=display_mode)
    except gtk.gdkgl.NoMatches:
      display_mode &= ~gtk.gdkgl.MODE_DOUBLE
      glconfig = gtk.gdkgl.Config(mode=display_mode)

    print "is RGBA:",                 glconfig.is_rgba()
    print "is double-buffered:",      glconfig.is_double_buffered()
    print "is stereo:",               glconfig.is_stereo()
    print "has alpha:",               glconfig.has_alpha()
    print "has depth buffer:",        glconfig.has_depth_buffer()
    print "has stencil buffer:",      glconfig.has_stencil_buffer()
    print "has accumulation buffer:", glconfig.has_accum_buffer()
    print

    self.drawing_area = SimpleDrawingArea(glconfig)
    self.drawing_area.set_size_request(1366, 768)
    self.drawing_area.show()

    # not sure why self.drawing_area becomes None later but it does,
    # store off the reference so we can get at it in da_
    self.da_ = self.drawing_area
    return self.drawing_area

  def on_ogl_window_expose_event(self, window_widget, even) :
    #print "ogl_window_expose_event"
    time.sleep(.1)

  def on_ogl_window_configure_event(self, window_widget, even) :
    #print "ogl_window_configure_event"
    time.sleep(.1)


  def on_eventbox_map_event(self, widget, event) :
    #print self.da_
    pysci.run_viewer_thread(self.da_.sci_context)

  def on_eventbox_key_press_event(self, window_widget, event) :
    e = pysci.KeyEvent()
    e.set_time(long(event.time))
    # translate gdk modifiers to sci modifiers
    mask = e.get_modifiers()
    e.set_modifiers(get_pysci_modifier_mask(event, mask))
    e.set_keyval(event.keyval)
    e.set_key_string(event.string)
    e.set_key_state(pysci.KeyEvent.KEY_PRESS_E)
    pysci.add_key_event(e)
    #print e
    return True

  def on_eventbox_enter_notify_event(self, window_widget, event) :
    #print "... enter notify! ...."
    #print event.type
    #print long(event.time)
    #print event.state
    return True

  def on_eventbox_motion_notify_event(self, window_widget, event) :
    e = pysci.PointerEvent()
    e.set_time(long(event.time))
    e.set_pointer_state(pysci.PointerEvent.MOTION_E)
    e.set_x(int(event.x))
    e.set_y(int(event.y))
    #e.set_which(event.button)
    # translate gdk modifiers to sci modifiers
    mask = e.get_modifiers()
    e.set_modifiers(get_pysci_modifier_mask(event, mask))
    n = get_pysci_button_num(event)
    state = e.get_pointer_state()
    e.set_pointer_state(get_pysci_pointer_modifier_mask(n, state))
    pysci.add_pointer_event(e)
    return True

  def on_eventbox_button_press_event(self, window_widget, event) :
    e = pysci.PointerEvent()
    e.set_time(long(event.time))
    e.set_pointer_state(pysci.PointerEvent.BUTTON_PRESS_E)
    e.set_x(int(event.x))
    e.set_y(int(event.y))
    # translate gdk modifiers to sci modifiers
    mask = e.get_modifiers()
    e.set_modifiers(get_pysci_modifier_mask(event, mask))
    state = e.get_pointer_state()
    e.set_pointer_state(get_pysci_pointer_modifier_mask(event.button,
							state))
    pysci.add_pointer_event(e)

    return True

  def on_eventbox_button_release_event(self, window_widget, event) :
    e = pysci.PointerEvent()
    e.set_time(long(event.time))
    e.set_pointer_state(pysci.PointerEvent.BUTTON_RELEASE_E)
    e.set_x(int(event.x))
    e.set_y(int(event.y))
    # translate gdk modifiers to sci modifiers
    mask = e.get_modifiers()
    e.set_modifiers(get_pysci_modifier_mask(event, mask))
    state = e.get_pointer_state()
    e.set_pointer_state(get_pysci_pointer_modifier_mask(event.button,
							state))
    pysci.add_pointer_event(e)
    return True

  def on_del_fld_clicked(self, a) :
    print "delete a model from the list"	

  def on_rm_faces_clicked(self, a) :
    e = pysci.CommandEvent("OpenGLViewer")
    e.set_command("delete faces")
    pysci.add_command_event(e)

  def on_add_face_clicked(self, a) :
    e = pysci.CommandEvent("OpenGLViewer")
    e.set_command("add face")
    pysci.add_command_event(e)

  def on_select_nodes_toggled(self, a) :
    etype = None
    if a.get_active() :
      etype = pysci.TMNotifyEvent.RESUME_E
    else :
      etype = pysci.TMNotifyEvent.SUSPEND_E


    e = pysci.TMNotifyEvent("FBPickTool", etype,
			    "OpenGLViewer")
    e.set_tool_mode("nodes")
    pysci.add_tm_notify_event(e)


  def on_select_faces_toggled(self, a) :
    etype = None
    if a.get_active() :
      etype = pysci.TMNotifyEvent.RESUME_E
    else :
      etype = pysci.TMNotifyEvent.SUSPEND_E


    e = pysci.TMNotifyEvent("FBPickTool", etype,
			    "OpenGLViewer")
    e.set_tool_mode("faces")
    pysci.add_tm_notify_event(e)

  def on_test_mesh_clicked(self, a) :
    sel = self.list_view_.get_selection()
    it = sel.get_selected()[1]
    if it == None :
      self.show_status_msg("No mesh currently selected.")
      return
    
    msg = "Testing selected mesh with TetGen."
    self.show_status_msg(msg)


  def show_status_msg(self, msg) :
    self.sbar_.pop(self.sbar_ctx_)
    self.sbar_.push(self.sbar_ctx_, msg)
    
  #sel_cb is the check button in the menu.
  def on_show_model_manip_activate(self, sel_cb) :
    sel_tb = self.xml_.get_widget("model_manip")
    if sel_cb.get_active() :
      sel_tb.show()
    else :
      sel_tb.hide()

  def on_add_mat_button_clicked(self, w) :
    ls = self.mat_list_view_.get_model()
    ls.insert(0, (0, "Material Name", 0.5))

  def on_rm_mat_button_clicked(self, w) :
    ls = self.mat_list_view_.get_model()
    sel = self.mat_list_view_.get_selection()

    it = sel.get_selected()[1]
    ls.remove(it)


  def on_gen_surf_button_clicked(self, w) :
    if self.proj_dir_ == None :
      return

    ls = self.mat_list_view_.get_model()
    for i in ls :
      print "%d %s %f" % (i[0], i[1], i[2])
      idx = i[0]
      t = (i[0], i[2])
      nrrd = self.input_nrrd_
      cur = os.getcwd()
      os.chdir(self.proj_dir_)
      segnrrd.make_solo_material(idx, nrrd)
      segnrrd.uniform_resample_nnrd(idx)      
      segnrrd.pad_nrrd("mat%d.resamp.nhdr" % idx, idx)
      segnrrd.make_afront_nhdr("mat%d.resamp.pad.nhdr" % idx)
      segnrrd.afront_isosurface("afront.mat%d.resamp.pad.nhdr" % idx, t)
      segnrrd.make_trisurf(idx)
      os.chdir(cur)




  def handle_input_nrrd(self) :
    self.input_nrrd_ = self.file_
    ent = self.xml_.get_widget("in_nrrd_ent")
    ent.set_text(self.input_nrrd_)
    segnrrd.inspect_nrrd(self.input_nrrd_)
    ent = self.xml_.get_widget("num_mats_ent")
    ent.set_text("%d" % segnrrd.maxval)
    
  def on_input_activate(self, a):
    self.open_file_handler_ = self.handle_input_nrrd
    self.get_file("*.nrrd")

  def handle_new_proj(self) :
    self.proj_dir_ = self.file_
    ent = self.xml_.get_widget("proj_dir_ent")
    ent.set_text(self.proj_dir_)

  def handle_open_proj(self) :
    self.proj_dir_ = self.file_
    ent = self.xml_.get_widget("proj_dir_ent")
    ent.set_text(self.proj_dir_)
    #also load all the surfaces from this dir that have been generated.
    
    files = glob.glob("%s/mat*.ts.fld" % self.proj_dir_)
    for f in files :
      fld_id = pysci.load_field(f)
      if fld_id == 0 :
	print "Error loading file: %s" % f
	return
      pysci.show_field(fld_id)
      ls = self.list_view_.get_model()
      ls.insert(0, (gtk.STOCK_YES, f, fld_id))
    

  def on_new_proj_activate(self, a):
    self.open_file_handler_ = self.handle_new_proj
    self.get_dir()

  def on_open_proj_activate(self, a):
    self.open_file_handler_ = self.handle_open_proj
    self.get_dir()

  def on_material_treeview_cell_edited(self, cell, pathstr, new, udat):
    col = udat
    ls = self.mat_list_view_.get_model()
    iter = ls.get_iter_from_string(pathstr)
    if col == 0 :
      ls.set(iter, col, int(new))
    elif col == 1 :
      ls.set(iter, col, new)
    else :
      ls.set(iter, col, float(new))

      
  def on_model_treeview_cell_edited(self, cell, pathstr, new, udat):
    col = udat
    ls = self.list_view_.get_model()
    iter = ls.get_iter_from_string(pathstr)

    print "ok toggle visibility"



if __name__ == "__main__" :
  gtk.threads_init()

  env = []
  for k in os.environ.keys() :
    estr = "%s=%s" % (k, os.environ[k])
    env.append(estr)

  pysci.init_pysci(env)
  test = SurfsToTets()
  try :
    gtk.main()
  except() :
    print "boo"
