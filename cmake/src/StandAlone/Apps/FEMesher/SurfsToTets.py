#!/usr/bin/python2.4
#  
#  For more information, please see: http://software.sci.utah.edu
#  
#  The MIT License
#  
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
#  
#  License for the specific language governing rights and limitations under
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

from OpenGL.GL import *
from OpenGL.GLU import *


# import the scirun python api.
import pysci

 
#import gtk.gtkgl

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

class Viewer(threading.Thread) :
    def __init__(self, glconfig):
        threading.Thread.__init__(self, None, None, "Viewer-1")
        self.glconfig = glconfig
        self.drawing_area = None
        
    def run(self) :
        try :
            print "running"
        
            self.drawing_area = SimpleDrawingArea(self.glconfig)
            self.drawing_area.set_size_request(1366, 768)
            self.drawing_area.show()

            while (self.drawing_area.sci_context == None) :
                print "not set yet"
                time.sleep(.1)
            
            pysci.run_viewer_thread(self.drawing_area.sci_context)
        except Exception, ex:
            print "caught exception in viewer thread"

        print "Viewer done - python"

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

        pysci.run_viewer_thread(self.sci_context)

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
        self.file_ = None
        self.active_chooser_ = None
        self.shortcuts_added_ = 0
        self.pbar_state = 0
        self.pbar_ = self.xml_.get_widget("progress")
        self.drawing_area = None
        
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
        
    def get_file(self, pattern) :
            
        chooser = self.xml_.get_widget("chooser")
        filt = gtk.FileFilter()
        filt.add_pattern(pattern)
        chooser.set_filter(filt)
        if not self.shortcuts_added_ :
            self.shortcuts_added_ = 1
            print os.curdir
            cdir = os.getcwd()
            chooser.add_shortcut_folder(cdir)
            if os.environ.has_key("SCIRUN_DATA") :
                path = os.environ["SCIRUN_DATA"]
                chooser.add_shortcut_folder(path)

        chooser.show()

    def on_in1_open_clicked(self, button) :
        self.get_file("*.fld")
        self.active_chooser_ = "in1"

    def on_in2_open_clicked(self, button) :
        self.get_file("*.fld")
        self.active_chooser_ = "in2"

    
    def on_out_save_clicked(self, button) :
        chooser = self.xml_.get_widget("chooser")
        chooser.show()
        self.active_chooser_ = "out"

    def on_chooser_cancel_clicked(self, button) :
         chooser = self.xml_.get_widget("chooser")
         chooser.hide()
         
    def on_chooser_open_clicked(self, button) :
        chooser = self.xml_.get_widget("chooser")
        if chooser.get_filename() != None :
            self.file_ = chooser.get_filename()
            chooser.hide()
            ent = None
            if (self.active_chooser_ == "in1") :
                ent = self.xml_.get_widget("input1-ent")
            elif (self.active_chooser_ == "in2") :
                ent = self.xml_.get_widget("input2-ent")
            else :
                ent = self.xml_.get_widget("output-ent")
            ent.set_text(self.file_)
            self.active_chooser_ = None


    def on_quit_button_clicked(self, button) :
        global vthread
        pysci.terminate();
        time.sleep(.1)
        sys.exit(0)

            
    def on_execute_button_clicked(self, button) :
        in1 = self.xml_.get_widget("input1-ent").get_text()
##         in2 = self.xml_.get_widget("input2-ent").get_text()
##         out = self.xml_.get_widget("output-ent").get_text()

##         self.pbar_.set_fraction(0.0)
##         self.pbar_.set_text("Working...")
##         self.pbar_state = 1
##         thread.start_new_thread(do_tetgen, (self, in1, in2, out))
        fld_id = pysci.load_field(in1)
        pysci.show_field(fld_id)
                         
        
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
        self.drawing_area.set_size_request(800, 450)
        self.drawing_area.show()
##         v = Viewer(glconfig);
##         v.start();
##         global vthread
##         vthread = v
        return self.drawing_area

    def on_ogl_window_expose_event(self, window_widget, even) :
        print "ogl_window_expose_event"
        time.sleep(.1)

    def on_ogl_window_configure_event(self, window_widget, even) :
        print "ogl_window_configure_event"
        time.sleep(.1)

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
	print e
        return True

    def on_eventbox_enter_notify_event(self, window_widget, event) :
        print "... enter notify! ...."
        print event.type
        print long(event.time)
        print event.state
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

    def on_add_b_clicked(self, a) :
	print "on_add_b_clicked"
	# first arg: is the name of the tool,
	# second arg: the notify state,
	# third arg: target just our viewer window tool manager for the event
	e = pysci.TMNotifyEvent("GeomPickTool", pysci.TMNotifyEvent.START_E,
				"OpenGLViewer")
	pysci.add_tm_notify_event(e)
	
    def on_remove_b_clicked(self, a) :
	print "on_remove_b_clicked"
	
    def on_nodes_tb_toggled(self, a) :
	print "on_nodes_tb_toggled"
	
    def on_edges_tb_toggled(self, a) :
	print "on_edges_tb_toggled"
	
    def on_faces_tb_toggled(self, a) :
	print "on_faces_tb_toggled"
	
    def on_elems_tb_toggled(self, a) :
	print "on_elems_tb_toggled"


    #sel_cb is the check button in the menu.
    def on_show_selection_activate(self, sel_cb) :
	print "on_show_selection_activate"
	sel_tb = self.xml_.get_widget("selection_tb")
	if sel_cb.get_active() :
		sel_tb.show()
	else :
		sel_tb.hide()
	
	

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
