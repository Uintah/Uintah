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


itcl_class SCIRun_Visualization_RescaleColorMap { 
    inherit Module 

    constructor {config} { 
        set name RescaleColorMap 
        set_defaults 
    } 
  
    method set_defaults {} { 
	global $this-main_frame
	set $this-main_frame ""

	global $this-isFixed
	global $this-min
	global $this-max
	global $this-makeSymmetric

	set $this-isFixed 0
	set $this-min 0
	set $this-max 1
	set $this-makeSymmetric 0
    }   

    method ui {} { 
	set w .ui[modname]
	
	if {[winfo exists $w]} { 
	    return
	} 
	
	toplevel $w 

	build_ui $w

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	# Don't let the GUI be smaller than it originally starts as.
	# (This works because moveToCursor forces the GUI to size
	#  itself.  Without the "update idletasks" in moveToCursor
        #  winfo would return 0.)
	set guiWidth [winfo reqwidth $w]
	set guiHeight [winfo reqheight $w]
	wm minsize $w $guiWidth $guiHeight

	if { [set $this-isFixed] } {
	    $w.bf.f3.fs select
	    $this fixedScale
	} else {
	    $w.bf.f1.as select
	    $this autoScale
	}
    }

    method build_ui { w } {
	global $this-main_frame
	set $this-main_frame $w

	global $this-isFixed
	global $this-min
	global $this-max
	global $this-makeSymmetric

	# Base Frame
	frame $w.bf
	pack $w.bf -padx 4 -pady 4 -fill both -expand y

	# Auto Scale Frame
	frame $w.bf.f1
	radiobutton $w.bf.f1.as -text "Auto Scale" -variable $this-isFixed \
	    -value 0 -command "$this autoScale"
	checkbutton $w.bf.f1.sas -text "Symmetric Auto Scale" -variable $this-makeSymmetric

	TooltipMultiline $w.bf.f1.as \
	    "Auto Scale uses the min/max values of the data (from the input field)\n" \
	    "and maps the color map to that range."
	TooltipMultiline $w.bf.f1.sas \
	    "Symmetric auto scaling of the color map will make the median data value\n" \
            "correspond to the the middle of the color map.  For example, if the maximum\n" \
            "data value is 80 and minimum is -20, the min/max range will be set to +/- 80\n" \
            "(and thus the median data value is set to 0)."

	pack $w.bf.f1.as  -side top -anchor w -padx 2
	pack $w.bf.f1.sas -side top -anchor w -padx 2

	# Fixed Scale Frame
	frame $w.bf.f3 -relief groove -borderwidth 2

	radiobutton $w.bf.f3.fs -text "Fixed Scale"  -variable $this-isFixed \
	    -value 1 -command "$this fixedScale"

	TooltipMultiline $w.bf.f3.fs \
	    "Fixed Scale allows the user to select the min and max\n" \
	    "values of the data that will correspond to the color map."

	frame $w.bf.f3.min
	label $w.bf.f3.min.l1 -text " Min:"
	entry $w.bf.f3.min.e1 -textvariable $this-min -width 10

	frame $w.bf.f3.max
	label $w.bf.f3.max.l2 -text "Max:"
	entry $w.bf.f3.max.e2 -textvariable $this-max -width 10

	pack $w.bf.f3.fs -anchor w -padx 2
	pack $w.bf.f3.min.l1 -anchor e -side left
	pack $w.bf.f3.min.e1 -expand yes -fill x -anchor e -side left
	pack $w.bf.f3.max.l2 -anchor e -side left
	pack $w.bf.f3.max.e2 -expand yes -fill x -anchor e -side left 

	pack $w.bf.f3.min -side top -anchor e -padx 2 -pady 2 -fill x 
	pack $w.bf.f3.max -side top -anchor e -padx 2 -pady 2 -fill x

	# pack in the auto scale and the fixed scale frames
	pack $w.bf.f1 -side left -anchor n
	pack $w.bf.f3 -side left -expand yes -fill both -anchor n

	bind $w.bf.f3.min.e1 <Return> "$this-c needexecute"
	bind $w.bf.f3.max.e2 <Return> "$this-c needexecute"
    }

    method autoScale { } {
	global $this-isFixed
	global $this-main_frame
	
	set w [set $this-main_frame]

	set lightgray "#444444"

	$w.bf.f1.sas    configure -state normal
	$w.bf.f3.min.l1 configure -foreground $lightgray
	$w.bf.f3.min.e1 configure -state disabled -foreground $lightgray
	$w.bf.f3.max.l2 configure -foreground $lightgray
	$w.bf.f3.max.e2 configure -state disabled -foreground $lightgray
    }

    method fixedScale { } {
	global $this-isFixed
	global $this-main_frame

	set w [set $this-main_frame]

	$w.bf.f1.sas     configure -state disabled
	$w.bf.f3.min.l1  configure -foreground black
	$w.bf.f3.min.e1  configure -state normal -foreground black
	$w.bf.f3.max.l2  configure -foreground black
	$w.bf.f3.max.e2  configure -state normal -foreground black
    }
}
