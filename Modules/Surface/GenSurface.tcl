
#
#  GenSurface.tcl
#
#  Written by:
#   Steven Parker
#   Department of Computer Science
#   University of Utah
#   June 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class GenSurface {
    inherit Module
    constructor {config} {
	set name GenSurface
	set_defaults
    }
    method set_defaults {} {
	global $this-surfacetype
	set $this-surfacetype point
	global $this-cyl_p1-x $this-cyl_p1-y $this-cyl_p1-z
	set $this-cyl_p1-x 0
	set $this-cyl_p1-y 0
	set $this-cyl_p1-z 0
	global $this-cyl_p2-x $this-cyl_p2-y $this-cyl_p2-z
	set $this-cyl_p1-x 0
	set $this-cyl_p1-y 0
	set $this-cyl_p1-z 1
	global $this-cyl_rad
	set $this-cyl_rad .1
	global $this-cyl_nu $this-cyl_nv $this-cyl_ndiscu
	set $this-cyl_nu 10
	set $this-cyl_nv 20
	set $this-cyl_ndiscu 4
	global $this-point_pos-x $this-point_pos-y $this-point_pos-z
	set $this-point_pos-x 0
	set $this-point_pos-y 0
	set $this-point_pos-z 0
	global $this-point_rad
	set $this-point_rad .1
	global $this-widget_color-r $this-widget_color-g $this-widget_color-b
	set $this-widget_color-r 1
	set $this-widget_color-g 0
	set $this-widget_color-b 0
	$this-c needexecute

    }
    method ui {} {
	set w .ui$this
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 20
	set n "$this-c needexecute "

	make_labeled_radio $w.type "Surface Type:" $n \
		top $this-cutting_plane_type \
		{{Point point} {Cylinder cylinder}}
	pack $w.type -side top -padx 5 -anchor w
    }
}
