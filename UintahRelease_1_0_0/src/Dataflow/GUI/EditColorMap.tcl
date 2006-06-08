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



itcl_class SCIRun_Visualization_EditColorMap {
    inherit Module
    constructor {config} {
	set name EditColorMap
	set_defaults
    }
    method set_defaults {} {
	global $this-rgbhsv
	set $this-rgbhsv 1

	global $this-resolution
	set $this-resolution 256

	trace variable $this-alphas_pickle w "$this-c unpickle"
	trace variable $this-rgbhsv w "$this-c toggle-hsv"
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    doGL
	    return
	}

	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w

	global $this-rgbhsv
	global $this-linespline

	set $this-rgbhsv 0
	set $this-linespline 0
	
	doGL
    }

    method doGL {} {
        set w .ui[modname]
        
        if {[winfo exists $w.f.gl1]} {
            raise $w
        } else {
	    set n "$this-c needexecute"

	    # HSB/RGB switching
#	    make_labeled_radio $w.f.types "Color Space" $n left \
#		    $this-rgbhsv { { "RGB" 0 }  { "HSV" 1 } }
	    
#	    pack $w.f.types -side top -anchor w

            # Resolution
	    scale $w.f.s2 -from 2 -to 256 -state normal \
		    -orient horizontal  -variable $this-resolution \
		    -label "Resolution"
	    pack $w.f.s2 -expand yes -fill x -pady 2 -padx 2

	    Tooltip $w.f.s2 "Sets the number of unique colors used in the color map."

	    bind $w.f.s2 <ButtonRelease> "$this-c needexecute"

	    frame $w.f.gl1
	    pack $w.f.gl1 -padx 2 -pady 2

            # Create an OpenGL widget.
	    $this-c setgl 0

	    # Every time the OpenGL widget is displayed, redraw it.
	    bind $w.f.gl1.gl <Expose> "$this-c expose 0"
	    bind $w.f.gl1.gl <ButtonPress> "$this-c mouse down %x %y %b"
	    bind $w.f.gl1.gl <ButtonRelease> "$this-c mouse release %x %y %b"
	    bind $w.f.gl1.gl <Motion>        "$this-c mouse motion %x %y"
	    bind $w.f.gl1.gl <Destroy> "$this-c closewindow"

	    # Place the widget on the screen.
	    pack $w.f.gl1.gl -fill both -expand 1
	}

        if {[winfo exists $w.f.gl3]} {
            raise $w
        } else {
            # Initialize geometry and placement of the widget.
	    frame $w.f.gl3
	    pack $w.f.gl3 -padx 2 -pady 2

            # Create an OpenGL widget.
	    $this-c setgl 1

            # Every time the OpenGL widget is displayed, redraw it.
            bind $w.f.gl3.gl <Expose> "$this-c expose 1"

            # Place the widget on the screen.
            pack $w.f.gl3.gl -fill both -expand 1
        }
    }
}
