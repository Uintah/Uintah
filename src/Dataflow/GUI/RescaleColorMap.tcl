#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

itcl_class SCIRun_Visualization_RescaleColorMap { 
    inherit Module 

    constructor {config} { 
        set name RescaleColorMap 
        set_defaults 
    } 
  
    method set_defaults {} { 
	global $this-isFixed
	global $this-min
	global $this-max
	global $this-makeSymmetric
	set bVar 0
	set $this-isFixed 0
	set $this-min 0
	set $this-max 1
	set $this-makeSymmetric 0
    }   

    method ui {} { 
	global $this-isFixed
	global $this-min
	global $this-max
	global $this-makeSymmetric

	set w .ui[modname]
	
	if {[winfo exists $w]} { 
	    return
	} 
	
	toplevel $w 
	wm minsize $w 200 50 
 
	# Base Frame
	frame $w.bf
	pack $w.bf -padx 4 -pady 4

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
	label $w.bf.f3.min.l1 -text "Min:"
	entry $w.bf.f3.min.e1 -textvariable $this-min -width 10

	frame $w.bf.f3.max
	label $w.bf.f3.max.l2 -text "Max:"
	entry $w.bf.f3.max.e2 -textvariable $this-max -width 10

	pack $w.bf.f3.fs -anchor w -padx 2
	pack $w.bf.f3.min.l1 $w.bf.f3.min.e1 -expand yes -fill x -anchor e -side left
	pack $w.bf.f3.max.l2 $w.bf.f3.max.e2 -expand yes -fill x -anchor e -side left

	pack $w.bf.f3.min -side top -anchor e -padx 2 -pady 2
	pack $w.bf.f3.max -side top -anchor e -padx 2 -pady 2

	# pack in the auto scale and the fixed scale frames
	pack $w.bf.f1 $w.bf.f3 -side left -expand yes -fill x -anchor n

	bind $w.bf.f3.min.e1 <Return> "$this-c needexecute"
	bind $w.bf.f3.max.e2 <Return> "$this-c needexecute"

	if { [set $this-isFixed] } {
	    $w.bf.f3.fs select
	    $this fixedScale
	} else {
	    $w.bf.f1.as select
	    $this autoScale
	}

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

    method autoScale { } {
	global $this-isFixed
	set w .ui[modname]
	
	set lightgray "#999999"

	$w.bf.f1.sas    configure -state normal
	$w.bf.f3.min.l1 configure -foreground $lightgray
	$w.bf.f3.min.e1 configure -state disabled -foreground $lightgray
	$w.bf.f3.max.l2 configure -foreground $lightgray
	$w.bf.f3.max.e2 configure -state disabled -foreground $lightgray
    }

    method fixedScale { } {
	global $this-isFixed
	set w .ui[modname]

	$w.bf.f1.sas     configure -state disabled
	$w.bf.f3.min.l1  configure -foreground black
	$w.bf.f3.min.e1  configure -state normal -foreground black
	$w.bf.f3.max.l2  configure -foreground black
	$w.bf.f3.max.e2  configure -state normal -foreground black
    }
}
