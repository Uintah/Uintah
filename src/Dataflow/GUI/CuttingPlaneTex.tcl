#
#  CuttingPlaneTex.tcl
#
#  Written by:
#   Colette Mullenhoff
#   Department of Computer Science
#   University of Utah
#   May. 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class PSECommon_Visualization_CuttingPlaneTex {
    inherit Module
    constructor {config} {
	set name CuttingPlaneTex
	set_defaults
    }
    method set_defaults {} {
	global $this-cutting_plane_type
	global $this-scale
	set $this-scale 1.0
	global $this-offset
	set $this-offset 0
	global $this-num_contours
	set $this-num_contours 20
	$this-c needexecute

    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 20
	frame $w.f -width 30
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	button $w.f.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.f.findxy -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.f.findyz -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.f.findxz -pady 2 -side top -ipadx 3 -anchor e

#	expscale $w.scale -orient horizontal -label "Scale:" \
#		-variable $this-scale -command $n
#	$w.scale-win- configure
#	pack $w.scale -fill x -pady 2
#	expscale $w.offset -orient horizontal -label "Offset:" \
#		-variable $this-offset -command $n
#	pack $w.offset -fill x -pady 2
	scale $w.num_contours -orient horizontal -label "Contours:" \
		-from 2 -to 500 -length 5c \
                -showvalue true \
                -resolution 1 \
		-variable $this-num_contours -command $n 
	pack $w.num_contours -fill x -pady 2
		
    }
}
