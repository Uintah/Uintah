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



catch {rename EditTransferFunc2 ""}

itcl_class Volume_Visualization_EditTransferFunc2 {
    inherit Module
    constructor {config} {
	set name EditTransferFunc2
	set_defaults
    }

    method set_defaults {} {
	global $this-rgbhsv
	set $this-rgbhsv 1
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    create_gl
	    return
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 20 -pady 20
        create_gl
    }

    method create_gl {} {
        set w .ui[modname]
        if {[winfo exists $w.f.gl]} {
            raise $w
        } else {
            set n "$this-c needexecute"
	    
            frame $w.f.gl
            pack $w.f.gl -padx 2 -pady 2
            # create an OpenGL widget
            opengl $w.f.gl.gl -geometry 512x256 -doublebuffer true -direct true \
                -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2
            # every time the OpenGL widget is displayed, redraw it
            bind $w.f.gl.gl <Expose> "$this-c expose"
            #bind $w.f.gl.gl <Configure> "$this-c resize"
            bind $w.f.gl.gl <ButtonPress> "$this-c mouse push %x %y %b"
            bind $w.f.gl.gl <ButtonRelease> "$this-c mouse release %x %y %b"
            bind $w.f.gl.gl <Motion> "$this-c mouse motion %x %y"
            bind $w.f.gl.gl <Destroy> "$this-c closewindow"
            # place the widget on the screen
            pack $w.f.gl.gl -fill both -expand 1
        }
    }
}
