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


# GUI for NIMRODConverter module
# by Allen R. Sanderson
# May 2003

catch {rename PCS_Visualization_Silhouettes ""}

itcl_class PCS_Visualization_Silhouettes {
    inherit Module
    constructor {config} {
        set name Silhouettes
        set_defaults

        global $this-build_field
        global $this-build_geom
	global $this-color-r
	global $this-color-g
	global $this-color-b
        global $this-autoexecute

        set $this-build_field 0
        set $this-build_geom  1
	set $this-color-r 0.4
	set $this-color-g 0.2
	set $this-color-b 0.9
        set $this-autoexecute 1
    }

    method set_defaults {} {
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

	toplevel $w

	iwidgets::labeledframe $w.opt -labelpos nw -labeltext "Options"
	set opt [$w.opt childsite]
	
	frame $opt.left
	checkbutton $opt.left.auto -text "Execute automatically" \
	    -variable $this-autoexecute

	checkbutton $opt.left.field -text "Build Output Field" \
	    -variable $this-build_field

	checkbutton $opt.left.geom -text "Build Output Geometry" \
	    -variable $this-build_geom

	pack $opt.left.field -side top -anchor nw -pady 5
	pack $opt.left.geom  -side top -anchor nw -pady 5
	pack $opt.left.auto  -side top -anchor nw -pady 5

	frame $opt.right
	
	addColorSelection $opt.right $this-color

	pack $opt.left $opt.right -side left -pady 5  -padx 10

	pack $w.opt -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    method raiseColor {swatch color} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     SciRaise $window.color
	     return
	 } else {
	     makeColorPicker $window.color $color \
		     "$this setColor $swatch $color" \
		     "destroy $window.color"
	 }
    }

    method setColor {swatch color} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $swatch config -background [format #%04x%04x%04x $ir $ig $ib]
         $this-c needexecute
    }

    method addColorSelection {frame color} {
	 #add node color picking 
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]
	 
	 frame $frame.colorFrame
	 frame $frame.colorFrame.swatch -relief ridge -borderwidth \
		 4 -height 0.8c -width 1.0c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmmd "$this raiseColor $frame.colorFrame.swatch $color"
	 button $frame.colorFrame.set_color \
		 -text "Default Color" -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.swatch -side left
	 pack $frame.colorFrame -side left -padx 3 -pady 3

    }

}
