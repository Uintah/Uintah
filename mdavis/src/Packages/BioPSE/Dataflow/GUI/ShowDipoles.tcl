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


##
 #  ShowDipoles.tcl: Set theta and phi for the dipole
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   June 1999
 #
 #  Copyright (C) 1999 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename BioPSE_Visualization_ShowDipoles ""}

itcl_class BioPSE_Visualization_ShowDipoles {
    inherit Module
    constructor {config} {
        set name ShowDipoles
        set_defaults
    }
    method set_defaults {} {
	global $this-widgetSizeGui_
	global $this-scaleModeGui_
	global $this-showLastVecGui_
	global $this-showLinesGui_
	global $this-num-dipoles
	global $this-force-field-reset

	set $this-widgetSizeGui_ 1
	set $this-scaleModeGui_ normalize
	set $this-showLastVecGui_ 0
	set $this-showLinesGui_ 1
	set $this-num-dipoles 0
	set $this-force-field-reset 1
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
	global $v
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
        wm minsize $w 150 30
        frame $w.f
	global $this-widgetSizeGui_
	make_entry $w.f.s "Widget Size:" $this-widgetSizeGui_ "$this-c widget_scale"
	frame $w.f.r -relief sunken -bd 2
	global $this-scaleModeGui_
	radiobutton $w.f.r.fixed -text "Fixed Size" -value "fixed" \
	    -variable $this-scaleModeGui_ -command "$this-c scale_mode"
	radiobutton $w.f.r.normalize -text "Normalize Largest" \
	    -value "normalize" -variable $this-scaleModeGui_ \
	    -command "$this-c scale_mode"
	radiobutton $w.f.r.scale -text "Scale Size" -value "scale" \
	    -variable $this-scaleModeGui_ -command "$this-c scale_mode"

	pack $w.f.r.fixed $w.f.r.normalize $w.f.r.scale -side top -anchor w
	global $this-showLastVecGui_
	checkbutton $w.f.v -text "Show Last As Vector" -variable $this-showLastVecGui_ -command "$this-c show_last_vec"
	global $this-showLinesGui_
	checkbutton $w.f.l -text "Show Lines" -variable $this-showLinesGui_ -command "$this-c show_lines"

	pack $w.f.s -side top -anchor w -padx 4 -pady 4
	pack $w.f.r -side top -anchor w -padx 4 -pady 4 -expand true -fill x
	pack $w.f.v $w.f.l -side top -anchor w -padx 4 -pady 4

        pack $w.f -side top -fill x -expand yes

	makeSciButtonPanel $w $w $this "\"Reset\" \"$this-c reset\" \"Reset to input.\""
	moveToCursor $w
    }
}
