#
#  BoxClipSField.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   July 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class PSECommon_Visualization_BoxClipSField {
    inherit Module
    constructor {config} {
	set name BoxClipSField
	set_defaults
    }
    method set_defaults {} {
    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 50

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2

	button $w.f.find_field -text "Find Field" -command "$this-c find_field"
	global $this-axis
	checkbutton $w.f.axis_allign -text "Axis Allign" -command \
		"$this-c axis_allign" -variable $this-axis
	$w.f.axis_allign select
	global $this-interp
	checkbutton $w.f.interpolate -text "Interpolate" -command \
		"$this-c need_execute" -variable $this-interp
	pack $w.f.find_field -side top -pady 5
	pack $w.f.axis_allign $w.f.interpolate -side top
    }	
}
