#
#  CuttingPlane.tcl
#
#  Written by:
#   Colette Mullenhoff
#   Department of Computer Science
#   University of Utah
#   May. 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class Uintah_Visualization_CuttingPlane {
    inherit Module
    constructor {config} {
	set name CuttingPlane
	set_defaults
    }
    method set_defaults {} {
	global $this-cutting_plane_type
	set $this-cutting_plane_type 0
	global $this-scale
	set $this-scale 1.0
	global $this-offset
	set $this-offset 0
	global $this-num_contours
	set $this-num_contours 20
	global $this-where
	set $this-where 0.5
	global $this-localMinMaxTCL
	set $this-localMinMaxTCL 0
	global $this-fullRezTCL
	set $this-fullRezTCL 0
	global $this-exhaustiveTCL
	set $this-exhaustiveTCL 0
#	$this-c needexecute
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
	pack $w.f -padx 2 -pady 2 -fill x -side top
	set n "$this-c needexecute "

	make_labeled_radio $w.f.wids "Plane Type:" $n \
		top $this-cutting_plane_type \
		{{Plane 0} {Surface 1} \
		{ContourLines 2}}
	pack $w.f.wids -side left -padx 5 -anchor w

	button $w.f.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.f.findxy -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.f.findyz -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.f.findxz -pady 2 -side top -ipadx 3 -anchor e

	frame $w.g -bd 2 -relief groove
	frame $w.g.x
	button $w.g.x.plus -text "+" -command "$this-c plusx"
	label $w.g.x.label -text " X "
	button $w.g.x.minus -text "-" -command "$this-c minusx"
	pack $w.g.x.plus $w.g.x.label $w.g.x.minus -side left -fill x -expand 1 -padx 10
	pack $w.g.x -side top

	frame $w.g.y
	button $w.g.y.plus -text "+" -command "$this-c plusy"
	label $w.g.y.label -text " Y "
	button $w.g.y.minus -text "-" -command "$this-c minusy"
	pack $w.g.y.plus $w.g.y.label $w.g.y.minus -side left -fill x -expand 1 -padx 10
	pack $w.g.y -side top

	frame $w.g.z
	button $w.g.z.plus -text "+" -command "$this-c plusz"
	label $w.g.z.label -text " Z "
	button $w.g.z.minus -text "-" -command "$this-c minusz"
	pack $w.g.z.plus $w.g.z.label $w.g.z.minus -side left -fill x -expand 1 -padx 10
	pack $w.g.z -side top
	pack $w.g -side top -fill x

	expscale $w.scale -orient horizontal -label "Scale:" \
		-variable $this-scale -command $n
#	$w.scale-win- configure
	pack $w.scale -fill x -pady 2
	expscale $w.offset -orient horizontal -label "Offset:" \
		-variable $this-offset -command $n
	pack $w.offset -fill x -pady 2
	scale $w.num_contours -orient horizontal -label "Contours:" \
		-from 2 -to 500 -length 5c \
                -showvalue true \
                -resolution 1 \
		-variable $this-num_contours -command $n 
	pack $w.num_contours -fill x -pady 2
	scale $w.where -orient horizontal -label "Where:" \
		-from 0 -to 1 -length 5c \
                -showvalue true \
                -resolution 0.01 \
		-variable $this-where

	pack $w.where -fill x -pady 2
	checkbutton $w.local -text "Local Relative Values" -variable \
		$this-localMinMaxTCL
	checkbutton $w.full -text "Full Resolution" -variable \
		$this-fullRezTCL
	checkbutton $w.exh -text "Exhaustive Search" -variable \
		$this-exhaustiveTCL
	pack $w.local $w.full $w.exh -fill x -pady 2 -side top
    }
}
